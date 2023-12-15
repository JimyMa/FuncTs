import ast
import ctypes
import os
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Optional

import torch

ltprof_path = os.path.join(os.path.dirname(__file__), "../lib/libltprof.so")
_lib = ctypes.cdll.LoadLibrary(ltprof_path)

_enabled = False


def metrics_enabled():
    return os.getenv("ENABLE_METRICS") is not None


def initialize_metrics():
    _lib.initializeMetrics()


def finalize_metrics():
    _lib.finalizeMetrics()


def begin_profiler_pass():
    _lib.beginProfilerPass()


def end_profiler_pass():
    _lib.endProfilerPass()


def all_passes_submitted():
    return _lib.allPassesSubmitted()


def enable_profiling():
    global _enabled
    _lib.enableProfiling()
    _enabled = True


def disable_profiling():
    global _enabled
    _lib.enableProfiling()
    _enabled = False


@dataclass
class Record:
    total: float = 0
    min: float = float("inf")
    max: float = 0
    count: int = 0
    begin: Optional[float] = None


_records: Dict[str, Record] = {}


def prof_begin(label: str):
    if not _enabled:
        return
    torch.cuda.synchronize()
    if label not in _records:
        _records[label] = Record()
    _records[label].begin = perf_counter()


def prof_end(label: str):
    if not _enabled:
        return
    torch.cuda.synchronize()
    record = _records[label]
    assert record.begin is not None
    dur = perf_counter() - record.begin
    record.begin = None
    record.count += 1
    record.total += dur
    record.min = min(record.min, dur)
    record.max = max(record.max, dur)


def fmt_duration(dur: float):
    units = ["s", "ms", "us", "ns"]
    idx = 0
    while idx < len(units) - 1 and dur < 1:
        dur *= 1e3
        idx += 1
    return "{:.4}{}".format(dur, units[idx])


_record_fmt = "{:<16}{:>10}{:>10}{:>10}{:>10}{:>10}"


def print_profiling_results(count: int):
    _lib.printProfilingResults(ctypes.c_size_t(int(count)))
    if len(_records) == 0:
        return
    print("\nRanges:")
    print(_record_fmt.format("Label", "Count", "Total", "Mean", "Min", "Max"))
    for label, record in _records.items():
        print(
            _record_fmt.format(
                label,
                record.count,
                fmt_duration(record.total),
                fmt_duration(record.total / record.count),
                fmt_duration(record.min),
                fmt_duration(record.max),
            )
        )


class ProfileRewriter(ast.NodeTransformer):
    def __init__(self) -> None:
        super().__init__()

    def visit_Call(self, node: ast.Call):
        # Filter profiling prints
        func = node.func
        if not isinstance(func, ast.Name):
            return node
        if func.id != "print":
            return node
        if len(node.args) != 2:
            return node
        label = node.args[0]
        if not isinstance(label, ast.Constant) or not isinstance(label.value, str):
            return node
        begin = node.args[1]
        if not isinstance(begin, ast.Constant) or not isinstance(begin.value, bool):
            return node

        # Replace with profiling function call
        if begin.value:
            prof_func_id = prof_begin.__name__
        else:
            prof_func_id = prof_end.__name__
        new_node = ast.Call(
            func=ast.Name(id=prof_func_id, ctx=ast.Load()), args=[label], keywords=[]
        )

        return ast.fix_missing_locations(new_node)

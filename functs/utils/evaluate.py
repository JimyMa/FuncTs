from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import torch

# from prof import (all_passes_submitted, begin_profiler_pass, disable_profiling,
#                   enable_profiling, end_profiler_pass, finalize_metrics,
#                   initialize_metrics)


def to_cuda(val):
    if isinstance(val, list):
        return [to_cuda(elem) for elem in val]
    elif isinstance(val, tuple):
        return tuple(to_cuda(elem) for elem in val)
    elif isinstance(val, torch.Tensor):
        return val.cuda()
    else:
        return val


warmup_runs = 16
run_duration = 10.


@dataclass
class EvalRecord:
    total: float
    count: int

    def mean(self):
        return self.total / self.count


def fmt_duration(dur: float):
    units = ['s', 'ms', 'us', 'ns']
    idx = 0
    while idx < len(units) - 1 and dur < 1:
        dur *= 1e3
        idx += 1
    return '{:.4}{}'.format(dur, units[idx])


def evaluate(task: Callable[[int], None]):
    for i in range(warmup_runs):
        task(i)

    # enable_profiling()
    torch.cuda.synchronize()
    count = 0
    begin = perf_counter()
    while perf_counter() - begin < run_duration:
        task(count)
        torch.cuda.synchronize()
        count += 1
    # disable_profiling()

    return EvalRecord(total=perf_counter() - begin, count=count)


# def eval_metrics(task: Callable[[int], None], num_samples: int):
#     initialize_metrics()

#     for i in range(warmup_runs):
#         task(i)
#     torch.cuda.synchronize()
#     torch.cuda.profiler.start()
#     count = 0
#     while True:
#         begin_profiler_pass()
#         for i in range(num_samples):
#             task(i)
#             torch.cuda.synchronize()
#         end_profiler_pass()
#         if count > 0 and all_passes_submitted():
#             break
#         count += 1

#     finalize_metrics()
    
import functools
import json
import os
import shutil
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity

from .prof import (
    all_passes_submitted,
    begin_profiler_pass,
    disable_profiling,
    enable_profiling,
    end_profiler_pass,
    finalize_metrics,
    initialize_metrics,
    print_profiling_results,
)


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


@dataclass
class EvalRecord:
    total: float
    count: int

    def mean(self):
        return self.total / self.count


def fmt_duration(dur: float, round_to=None, split=False):
    units = ["s", "ms", "us", "ns"]
    round_to = round_to or "ns"
    end_idx = units.index(round_to)
    idx = 0
    while idx < len(units) - 1 and dur < 1:
        dur *= 1e3
        idx += 1
        if idx == end_idx:
            break
    if not split:
        return "{:.4}{}".format(dur, units[idx])
    else:
        return "{:.4}".format(dur), "{}".format(units[idx])


class Timer:
    def __init__(self, name="", color=True):
        self.name = name
        self.clear()
        self.color = color

    def clear(self):
        self.start_time = None
        self.end_time = None
        self.observation = None
        self.min = 1e9
        self.max = 0
        self.sum = 0
        self.cnt = 0

    def start(self):
        self.start_time = perf_counter()
        self.observation = self.start_time
        return self.start_time

    def end(self):
        self.end_time = perf_counter()
        duration = self.end_time - self.start_time
        return duration

    def time(self):
        return perf_counter()

    def unobserve(self):
        new_observation = self.time()
        self.observation = new_observation

    def observe(self):
        new_observation = self.time()
        duration = new_observation - self.observation
        self.min = min(self.min, duration)
        self.max = max(self.max, duration)
        self.sum += duration
        self.cnt += 1
        self.observation = new_observation

    def avg(self, round_to=None):
        return fmt_duration(self.sum / self.cnt, round_to=round_to, split=True)[0]

    def report(self, color=None, clear=True):
        if color is None:
            color = self.color
        if color:
            print(
                "{}: \033[31m{} iters, min = {}, max = {}, avg = {}\033[m".format(
                    self.name,
                    self.cnt,
                    fmt_duration(self.min),
                    fmt_duration(self.max),
                    fmt_duration(self.sum / self.cnt),
                )
            )
        else:
            print(
                "{}: {} iters, min = {} {}, max = {:.4f} {}, avg = {} {}".format(
                    self.name,
                    self.cnt,
                    fmt_duration(self.min),
                    fmt_duration(self.max),
                    fmt_duration(self.sum / self.cnt),
                )
            )
        if clear:
            self.clear()


WARMUP_RUNS_DEFAULT = 16
ITER_RUNS_DEFUALT = 100
ITER_PER_CAPTURE_DEFAULT = 5
RUN_DURATION_DEFAULT = 10.0
CUDA_GRAPH_CAPTURE_POOL_NUM_DEFAULT = 1


def evaluate_task(
    task: Callable[[int], None],
    name="",
    warmup_runs=WARMUP_RUNS_DEFAULT,
    run_duration=RUN_DURATION_DEFAULT,
    device="cuda",
) -> Timer:
    for i in range(warmup_runs):
        task(i)
    for i in range(warmup_runs):
        task(i)
    torch.cuda.synchronize()
    timer = Timer(name)
    begin = timer.start()
    enable_profiling()
    cnt = 0
    while timer.time() - begin < run_duration:
        task(cnt)
        cnt += 1
        timer.observe()
        torch.cuda.synchronize()
    timer.report(clear=False)
    disable_profiling()
    print_profiling_results(timer.cnt)

    return timer


def evaluate_func(
    func,
    args,
    name="",
    warmup_runs=WARMUP_RUNS_DEFAULT,
    run_duration=RUN_DURATION_DEFAULT,
    enable_cudagraph=False,
    iter_per_capture=ITER_PER_CAPTURE_DEFAULT,
    cuda_graph_caputure_pool_num=CUDA_GRAPH_CAPTURE_POOL_NUM_DEFAULT,
    device="cuda",
) -> Timer:
    for _ in range(warmup_runs):
        func(*args)
    torch.cuda.synchronize()
    timer = Timer(name)
    begin = timer.start()
    enable_profiling()
    while timer.time() - begin < run_duration:
        if not enable_cudagraph:
            func(*args)
            timer.observe()
        else:
            # g = torch_cuda_graph_pool[graph_cnt] if graph_cnt < cuda_graph_caputure_pool_num else torch.cuda.CUDAGraph()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for i in range(iter_per_capture):
                    func(*args)
                    observation = timer.observation
                    if i == 0:
                        timer.unobserve()
                    else:
                        timer.observe()
            torch.cuda.synchronize()
    timer.report(clear=False)
    disable_profiling()
    print_profiling_results(timer.cnt)
    return timer


def profiler_task(
    task: Callable[[int], None],
    name="",
    warmup_runs=WARMUP_RUNS_DEFAULT,
    run_duration=RUN_DURATION_DEFAULT,
    device="cuda",
    export_json=None,
) -> "ProfilerOberservation":
    for i in range(warmup_runs):
        task(i)
    profiler = torch.profiler.profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]
    )
    torch.cuda.synchronize()
    timer = Timer(name)
    profiler.start()
    begin = timer.start()
    count = 0
    while timer.time() - begin < run_duration:
        task(count)
        if device == "cuda":
            torch.cuda.synchronize()
        count += 1
        timer.observe()
    timer.report(clear=False)

    profiler.stop()
    if export_json:
        profiler.export_chrome_trace("nvfuser.json")
    return ProfilerOberservation(timer, profiler)


def proifler_func(
    func,
    args,
    name="",
    warmup_runs=WARMUP_RUNS_DEFAULT,
    run_duration=RUN_DURATION_DEFAULT,
    device="cuda",
    export_json=None,
) -> "ProfilerOberservation":
    for _ in range(warmup_runs):
        func(*args)
    profiler = torch.profiler.profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], use_cuda=True
    )
    torch.cuda.synchronize()
    timer = Timer(name)
    profiler.start()
    begin = timer.start()
    while timer.time() - begin < run_duration:
        func(*args)
        torch.cuda.synchronize()
        timer.observe()
    timer.report(clear=False)
    profiler.stop()
    return ProfilerOberservation(timer, profiler, export_json=export_json)


# def eval_metrics_func(func,
#                   args,
#                   name="",
#                   run_duration=RUN_DURATION_DEFAULT,
#                   warmup_runs=WARMUP_RUNS_DEFAULT,
#                   iter_runs=ITER_RUNS_DEFUALT,
#                   device="cuda"):
#     initialize_metrics()

#     for i in range(warmup_runs):
#         func(*args)
#     torch.cuda.synchronize()
#     torch.cuda.profiler.start()
#     count = 0
#     timer = Timer(0)
#     timer.start()
#     while True:
#         begin_profiler_pass()
#         for _ in range(iter_runs):
#             func(*args)
#             timer.observe()
#         end_profiler_pass()
#         if count > 0 and all_passes_submitted():
#             break
#         count += 1
#     timer.report()
#     finalize_metrics()


class ProfilerOberservation(object):
    def __init__(
        self, timer: Timer, prof: torch.profiler.profile, export_json=None, **kwargs
    ) -> None:
        self.timer = timer
        self.name = self.timer.name
        self.prof = prof

        json_path = export_json or tempfile.mktemp()
        self.prof.export_chrome_trace(json_path)
        with open(json_path, "r") as f:
            profiler_json = json.load(f)
        self.prof_json = profiler_json
        # os.remove(json_path)

    @property
    def total_cuda_memory_allocation(self) -> None:
        print(self.prof_json.keys())
        return (
            functools.reduce(
                lambda x, y: x + y,
                [
                    event["args"]["bytes"]
                    for event in self.prof_json["traceEvents"]
                    if event.get("cat") == "gpu_memcpy"
                    and "Memcpy DtoD" in event.get("name")
                ],
                0,
            )
            / self.timer.cnt
        )

    @property
    def total_kernel_calls(self) -> None:
        return (
            functools.reduce(
                lambda x, y: x + y,
                [
                    1
                    for event in self.prof_json["traceEvents"]
                    if event.get("cat") == "kernel"
                ],
            )
            / self.timer.cnt
        )

    @property
    def total_kernel_durations(self) -> None:
        return (
            functools.reduce(
                lambda x, y: x + y,
                [
                    event.get("dur")
                    for event in self.prof_json["traceEvents"]
                    if event.get("cat") == "kernel"
                ],
            )
            / self.timer.cnt
        )

    @property
    def key_metrics(self) -> dict:
        return OrderedDict(
            [
                ["name", self.name],
                ["kenel counts", self.total_kernel_calls],
                ["kernel dur", self.total_kernel_durations],
                ["cuda memory allocation", self.total_cuda_memory_allocation],
            ]
        )


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

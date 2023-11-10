import os
import functools
import tempfile
import shutil
import json
from dataclasses import dataclass
from time import perf_counter
from typing import Callable
from collections import OrderedDict


import torch
from torch.profiler import ProfilerActivity
from torch.autograd import DeviceType

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


def fmt_duration(dur: float):
    units = ['s', 'ms', 'us', 'ns']
    idx = 0
    while idx < len(units) - 1 and dur < 1:
        dur *= 1e3
        idx += 1
    return '{:.4}{}'.format(dur, units[idx])


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

    def observe(self):
        new_observation = self.time()
        duration = new_observation - self.observation
        self.min = min(self.min, duration)
        self.max = max(self.max, duration)
        self.sum += duration
        self.cnt += 1
        self.observation = new_observation

    def report(self, color = None, clear=True):
        if color is None: color = self.color
        if color:
            print("{}: \033[31m{} iters, min = {}, max = {}, avg = {}\033[m".format(
                self.name,
                self.cnt,
                fmt_duration(self.min),
                fmt_duration(self.max),
                fmt_duration(self.sum / self.cnt),
            ))
        else:
            print("{}: {} iters, min = {} {}, max = {:.4f} {}, avg = {} {}".format(
                self.name,
                self.cnt,
                fmt_duration(self.min),
                fmt_duration(self.max),
                fmt_duration(self.sum / self.cnt),
            ))
        if clear:
            self.clear()


WARMUP_RUNS_DEFAULT = 16
RUN_DURATION_DEFAULT = 10.

def evaluate_task(task: Callable[[int], None],
                  name="",
                  warmup_runs=WARMUP_RUNS_DEFAULT,
                  run_duration=RUN_DURATION_DEFAULT,
                  device="cuda") -> Timer:
    for i in range(warmup_runs):
        task(i)

    # enable_profiling()
    if device == "cuda":
        torch.cuda.synchronize()
    count = 0
    timer = Timer(name)
    begin = timer.start()
    while timer.time() - begin < run_duration:
        task(count)
        if (device == "cuda"):
            torch.cuda.synchronize()
        count += 1
        timer.observe()
    # disable_profiling()
    timer.report(clear=False)

    return timer


def evaluate_func(func, 
                  args,
                  name="",
                  warmup_runs=WARMUP_RUNS_DEFAULT,
                  run_duration=RUN_DURATION_DEFAULT,
                  device="cuda") -> Timer:
    for _ in range(warmup_runs):
        func(*args)
    torch.cuda.synchronize() 
    timer = Timer(name)
    begin = timer.start()
    while timer.time() - begin < run_duration:
        func(*args)
        timer.observe()
    timer.report(clear=False)
    return timer

def profiler_task(task: Callable[[int], None],
                  name="",
                  warmup_runs=WARMUP_RUNS_DEFAULT,
                  run_duration=RUN_DURATION_DEFAULT,
                  device="cuda") -> "ProfilerOberservation":
    for i in range(warmup_runs):
        task(i)
    profiler = torch.profiler.profile(activities=[ProfilerActivity.CUDA])
    torch.cuda.synchronize()
    timer = Timer(name)
    profiler.start()
    begin = timer.start()
    count = 0
    while timer.time() - begin < run_duration:
        task(count)
        if (device == "cuda"):
            torch.cuda.synchronize()
        count += 1
        timer.observe()
    timer.report(clear=False)

    profiler.stop()
    json_path = tempfile.mktemp()
    profiler.export_chrome_trace(json_path)
    with open(json_path, "r") as f:
        profiler_json = json.load(f)
    os.remove(json_path)
    return ProfilerOberservation(timer, profiler_json)

def proifler_func(func, 
                  args,
                  name="",
                  warmup_runs=WARMUP_RUNS_DEFAULT,
                  run_duration=RUN_DURATION_DEFAULT,
                  device="cuda") -> "ProfilerOberservation":
    for _ in range(warmup_runs):
        func(*args)
    profiler = torch.profiler.profile(activities=[ProfilerActivity.CUDA])
    torch.cuda.synchronize()
    timer = Timer(name)
    profiler.start()
    begin = timer.start()
    while timer.time() - begin < run_duration:
        func(*args)
        timer.observe()
    timer.report(clear=False)

    profiler.stop()
    json_path = tempfile.mktemp()
    profiler.export_chrome_trace(json_path)
    with open(json_path, "r") as f:
        profiler_json = json.load(f)
    os.remove(json_path)
    return ProfilerOberservation(timer, profiler_json)



class ProfilerOberservation(object):
    def __init__(self, timer: Timer,
                 prof: torch.autograd.profiler_util.EventList,
                 **kwargs) -> None:
        self.timer = timer
        self.name = self.timer.name
        self.prof = prof

    @property
    def total_kernel_calls(self) -> None:
        return functools.reduce(lambda x, y: x + y, [1 for event in self.prof["traceEvents"] if event.get("cat") == "kernel"]) / self.timer.cnt

    @property
    def key_metrics(self) -> dict:
        return OrderedDict([ ["name", self.name], ["kenel counts", self.total_kernel_calls] ])

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
    
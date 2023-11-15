import time

import functools

from typing import List

import torch
from torch.profiler import profile, ProfilerActivity
import functs

import ssd_bbox_static as ssd_bbox

torch.cuda.init()


# self.prior_generator = SSDAnchorGenerator(
#             strides=[16, 32, 64, 107, 160, 320],
#             ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
#             min_sizes=[48, 100, 150, 202, 253, 304],
#             max_sizes=[100, 150, 202, 253, 304, 320],
#             scale_major=False
#         )

cls_score = torch.rand([1, 486, 20, 20]).cuda()
bbox_pred = torch.rand([1, 24, 20, 20]).cuda()

feats = torch.load("ssd_feat.pt")

eager_timer: List[functs.utils.evaluate.Timer] = []
jit_timer: List[functs.utils.evaluate.Timer] = []
functs_timer: List[functs.utils.evaluate.Timer] = []

model_0 = ssd_bbox.SSDBBox(16, [2, 3], 48, 100, False).eval()
jit_model_0 = torch.jit.script(model_0)
functs_model_0 = functs.jit.script(model_0)

args_0 = feats[0][0][0], feats[0][1][0]

eager_timer.append(functs.utils.evaluate_func(model_0, args_0, "ssd eager", run_duration=2.0))
jit_timer.append(functs.utils.evaluate_func(jit_model_0, args_0, "ssd jit", run_duration=2.0))
functs_timer.append(functs.utils.evaluate_func(functs_model_0, args_0, "ssd functs", run_duration=2.0))


model_1 = ssd_bbox.SSDBBox(32, [2, 3], 100, 150, False).eval()
jit_model_1 = torch.jit.script(model_1)
functs_model_1 = functs.jit.script(model_1)

args_1 = feats[0][0][1], feats[0][1][1]

eager_timer.append(functs.utils.evaluate_func(model_1, args_1, "ssd eager", run_duration=2.0))
jit_timer.append(functs.utils.evaluate_func(jit_model_1, args_1, "ssd jit", run_duration=2.0))
functs_timer.append(functs.utils.evaluate_func(functs_model_1, args_1, "ssd functs", run_duration=2.0))


model_2 = ssd_bbox.SSDBBox(64, [2, 3], 150, 202, False).eval()
jit_model_2 = torch.jit.script(model_2)
functs_model_2 = functs.jit.script(model_2)

args_2 = feats[0][0][2], feats[0][1][2]

eager_timer.append(functs.utils.evaluate_func(model_2, args_2, "ssd eager", run_duration=2.0))
jit_timer.append(functs.utils.evaluate_func(jit_model_2, args_2, "ssd jit", run_duration=2.0))
functs_timer.append(functs.utils.evaluate_func(functs_model_2, args_2, "ssd functs", run_duration=2.0))

model_3 = ssd_bbox.SSDBBox(107, [2, 3], 202, 253, False).eval()
jit_model_3 = torch.jit.script(model_3)
functs_model_3 = functs.jit.script(model_3)

args_3 = feats[0][0][3], feats[0][1][3]

eager_timer.append(functs.utils.evaluate_func(model_3, args_3, "ssd eager", run_duration=2.0))
jit_timer.append(functs.utils.evaluate_func(jit_model_3, args_3, "ssd jit", run_duration=2.0))
functs_timer.append(functs.utils.evaluate_func(functs_model_3, args_3, "ssd functs", run_duration=2.0))

model_4 = ssd_bbox.SSDBBox(160, [2, 3], 253, 304, False).eval()
jit_model_4 = torch.jit.script(model_4)
functs_model_4 = functs.jit.script(model_4)

args_4 = feats[0][0][4], feats[0][1][4]

eager_timer.append(functs.utils.evaluate_func(model_4, args_4, "ssd eager", run_duration=2.0))
jit_timer.append(functs.utils.evaluate_func(jit_model_4, args_4, "ssd jit", run_duration=2.0))
functs_timer.append(functs.utils.evaluate_func(functs_model_4, args_4, "ssd functs", run_duration=2.0))

model_5 = ssd_bbox.SSDBBox(320, [2, 3], 304, 320, False).eval()
jit_model_5 = torch.jit.script(model_5)
functs_model_5 = functs.jit.script(model_5)

args_5 = feats[0][0][5], feats[0][1][5]

eager_timer.append(functs.utils.evaluate_func(model_5, args_5, "ssd eager", run_duration=2.0))
jit_timer.append(functs.utils.evaluate_func(jit_model_5, args_5, "ssd jit", run_duration=2.0))
functs_timer.append(functs.utils.evaluate_func(functs_model_5, args_5, "ssd functs", run_duration=2.0))

eager_time_avg = functools.reduce(lambda x, y: x + y, [x.sum / x.cnt  for x in eager_timer])
print("eager_time_avg: ", eager_time_avg)


jit_time_avg = functools.reduce(lambda x, y: x + y, [x.sum / x.cnt  for x in jit_timer])
print("jit_time_avg: ", jit_time_avg)


functs_time_avg = functools.reduce(lambda x, y: x + y, [x.sum / x.cnt  for x in functs_timer])
print("functs_time_avg: ", functs_time_avg)

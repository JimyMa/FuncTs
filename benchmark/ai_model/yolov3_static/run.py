import time

from typing import List, Tuple

import torch
from torch.profiler import profile, ProfilerActivity
import functs

import yolov3_bbox_static as yolov3_bbox

torch.cuda.init()

# pred_maps = [torch.Size([1, 255, 10, 10]), torch.Size([1, 255, 20, 20]), torch.Size([1, 255, 40, 40])]

strides=[32, 16, 8]
base_sizes=[[(116, 90), (156, 198), (373, 326)],
            [(30, 61), (62, 45), (59, 119)],
            [(10, 13), (16, 30), (33, 23)]]


class YOLOV3MultiLevel(torch.nn.Module):
    def __init__(self, strides, 
                       base_sizes):
        super().__init__()
        self.bboxes_module_0 = yolov3_bbox.YOLOV3BBox(strides[0], base_sizes[0])
        self.bboxes_module_1 = yolov3_bbox.YOLOV3BBox(strides[0], base_sizes[0])
        self.bboxes_module_2 = yolov3_bbox.YOLOV3BBox(strides[0], base_sizes[0])
    
    def forward(self, feats: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        preds = feats[0]

        pred_0 = preds[0].clone()
        result_0 = self.bboxes_module_0(pred_0)

        pred_1 = preds[1].clone()
        result_1 = self.bboxes_module_1(pred_1)

        pred_2 = preds[2].clone()
        result_2 = self.bboxes_module_2(pred_2)
        return result_0, result_1, result_2


feats = torch.load("yolov3_feat.pt")
num_samples = len(feats)


model = YOLOV3MultiLevel(strides, base_sizes).cuda().eval()
jit_module = torch.jit.script(model)
functs_module = functs.jit.script(model)
dynamo_module = torch.compile(model, dynamic=True)

fait_model = functs.jit.script(model, backend="fait")
type_hint = [torch.TupleType([torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 255, 10, 10]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 255, 20, 20]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 255, 40, 40]).with_device(torch.device("cuda"))],)]

functs._C._jit_pass_fait_pipeline(fait_model.graph, type_hint)

print(fait_model.graph)

code = torch._C._jit_get_code(fait_model.graph)


model(feats[0])
jit_module(feats[0])
functs_module(feats[0])

functs.utils.evaluate_func(model, [feats[0]], "model", run_duration=0.5)
functs.utils.evaluate_func(jit_module, [feats[0]], "jit model", run_duration=0.5)
functs.utils.evaluate_func(dynamo_module, [feats[0]], "dynamo model", run_duration=0.5)
functs.utils.evaluate_func(functs_module, [feats[0]], "functs model", run_duration=0.5)





# model = yolov3_bbox.YOLOV3BBox(strides[0], base_sizes[0]).cuda().eval()
# model_0 = yolov3_bbox.YOLOV3BBox(strides[0], base_sizes[0]).cuda().eval()
# jit_model_0 = torch.jit.script(model_0)
# dynamo_model_0 = torch.compile(model_0, dynamic=True)
# functs_model_0 = functs.jit.script(torch.jit.freeze(torch.jit.script(model_0)))
# nvfuser_model_0 = torch.jit.freeze(torch.jit.script(model_0))

# model_1 = yolov3_bbox.YOLOV3BBox(strides[1], base_sizes[1]).cuda().eval()
# jit_model_1 = torch.jit.script(model_1)
# dynamo_model_1 = torch.compile(model_1, dynamic=True)
# functs_model_1 = functs.jit.script(torch.jit.freeze(torch.jit.script(model_1)))
# nvfuser_model_1 = torch.jit.freeze(torch.jit.script(model_0))

# model_2 = yolov3_bbox.YOLOV3BBox(strides[2], base_sizes[2]).cuda().eval()
# jit_model_2 = torch.jit.script(model_2)
# dynamo_model_2 = torch.compile(model_2, dynamic=True)
# functs_model_2 = functs.jit.script(torch.jit.freeze(torch.jit.script(model_2)))
# nvfuser_model_2 = torch.jit.freeze(torch.jit.script(model_0))



# model = [model_0, model_1, model_2]
# jit_model = [jit_model_0, jit_model_1, jit_model_2]
# dynamo_model = [dynamo_model_0, dynamo_model_1, dynamo_model_2]
# functs_model = [functs_model_0, functs_model_1, functs_model_2]
# nvfuser_model = [nvfuser_model_0, nvfuser_model_1, nvfuser_model_2]


# # inp = torch.rand([1, 255, 10, 10]).cuda()
# # functs.utils.evaluate_func(model, (feats[1][0][1], ), "eager", run_duration=1.)
# # functs.utils.evaluate_func(jit_model, (feats[1][0][1], ), "jit", run_duration=1.)
# # functs.utils.evaluate_func(functs_model, (feats[1][0][1], ), "functs", run_duration=1.)

# def multi_level_func(model, feats, idx):
#     feats = feats[0 % num_samples][0]
#     # for i, feat in  enumerate(feats):
#     #     _ = model[i](feat)
#     torch.cuda.synchronize()
#     model[0](feats[0])
#     torch.cuda.synchronize()
#     model[1](feats[1])
#     torch.cuda.synchronize()
#     model[2](feats[2])

# def eager_task(idx: int):
#     multi_level_func(model, feats, idx)

# def jit_task(idx: int):
#     multi_level_func(jit_model, feats, idx)

# def dynamo_task(idx: int):
#     multi_level_func(dynamo_model, feats, idx)

# def functs_task(idx: int):
#     multi_level_func(functs_model, feats, idx)

# def nvfuser_task(idx: int):
#     multi_level_func(nvfuser_model, feats, idx)


# functs.utils.evaluate_task(eager_task, "eager", run_duration=2.)
# functs.utils.evaluate_task(jit_task, "jit", run_duration=2.)
# functs.utils.evaluate_task(functs_task, "functs", run_duration=2.)
# functs.utils.evaluate_task(dynamo_task, "dynamo+inductor", run_duration=2.)

# torch._C._jit_set_nvfuser_enabled(True)
# functs.utils.evaluate_task(nvfuser_task, "nvfuser", run_duration=2.)
# torch._C._jit_set_nvfuser_enabled(False)

# # nvfuser
# torch._C._jit_set_nvfuser_enabled(True)
# functs.utils.evaluate_task(nvfuser_task, "nvfuser", run_duration=2.)
# torch._C._jit_set_nvfuser_enabled(False)

# # print(functs.utils.profiler_task(eager_task, "eager", run_duration=2.).key_metrics)
# # print(functs.utils.profiler_task(jit_task, "jit", run_duration=2.).key_metrics)
# # print(functs.utils.profiler_task(fait_task, "fait", run_duration=2.).key_metrics)




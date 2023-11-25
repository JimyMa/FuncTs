import time

import torch
from torch.profiler import profile, ProfilerActivity
import functs

import argparse

import yolov3_bbox

torch.cuda.init()

parser = argparse.ArgumentParser()
parser.add_argument("--bs", default=1, type=int)
arguments = parser.parse_args()

def process_feat(feat):
    new_feat = []
    for data_tuple in feat:
        new_data_tuple =[data.repeat(arguments.bs, 1, 1, 1) for data in data_tuple]
        new_feat.append(new_data_tuple)
    return tuple(new_feat)

def process_feat_batch(feats):
    new_feats = []
    for feat in feats:
        new_feats.append(process_feat(feat))
    return new_feats

# pred_maps = [torch.Size([1, 255, 10, 10]), torch.Size([1, 255, 20, 20]), torch.Size([1, 255, 40, 40])]

model = yolov3_bbox.YOLOV3BBox().cuda().eval()

# fait
cls_scores: [torch.Size([1, 486, 20, 20]), torch.Size([1, 486, 10, 10]), torch.Size([1, 486, 5, 5]), torch.Size([1, 486, 3, 3]), torch.Size([1, 486, 2, 2]), torch.Size([1, 486, 1, 1])]
bbox_preds: [torch.Size([1, 24, 20, 20]), torch.Size([1, 24, 10, 10]), torch.Size([1, 24, 5, 5]), torch.Size([1, 24, 3, 3]), torch.Size([1, 24, 2, 2]), torch.Size([1, 24, 1, 1])]
type_hint = [torch.TupleType([torch.TensorType.get().with_dtype(torch.float32).with_sizes([arguments.bs, 255, 10, 10]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([arguments.bs, 255, 20, 20]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([arguments.bs, 255, 40, 40]).with_device(torch.device("cuda"))],)]

feats = torch.load("yolov3_feat.pt")
feats = process_feat_batch(feats)
num_samples = len(feats)

# functs_model = functs.jit.script(torch.jit.freeze(torch.jit.script(model)))

with torch.no_grad():
    # torchscript
    jit_model = torch.jit.freeze(torch.jit.script(model))
    
    # nvfuser
    nvfuser_model = torch.jit.freeze(torch.jit.script(model))
    
    # torch dynamo + inductor
    torch._dynamo.reset()
    tracing_model = torch.compile(model)

    # functs
    functs_model = functs.jit.script(model)

    # aot backend
    fait_model = functs.jit.script(model, backend="fait")
    functs._C._jit_pass_fait_pipeline(fait_model.graph, type_hint)
    fait_model = functs._C._create_function_from_graph("forward", fait_model.graph)

    def eager_task(idx: int):
        model(*feats[idx % num_samples])

    def jit_task(idx: int):
        jit_model(*feats[idx % num_samples])

    def functs_task(idx: int):
        functs_model(*feats[idx % num_samples])

    def tracing_task(idx: int):
        tracing_model(*feats[idx % num_samples])

    def nvfuser_task(idx: int):
        nvfuser_model(*feats[idx % num_samples])

    def fait_task(idx: int):
        fait_model("",  *feats[idx % num_samples])

    functs.utils.evaluate_task(eager_task, "eager", run_duration=2.)
    functs.utils.evaluate_task(jit_task, "jit", run_duration=2.)
    functs.utils.evaluate_task(functs_task, "functs", run_duration=2.)
    functs.utils.evaluate_task(tracing_task, "dynamo+inductor", run_duration=2.)
    functs.utils.evaluate_task(fait_task, "fait", run_duration=2.)

    # nvfuser
    torch._C._jit_set_nvfuser_enabled(True)
    functs.utils.evaluate_task(nvfuser_task, "nvfuser", run_duration=2.)
    torch._C._jit_set_nvfuser_enabled(False)

    # print(functs.utils.profiler_task(eager_task, "eager", run_duration=2.).key_metrics)
    # print(functs.utils.profiler_task(jit_task, "jit", run_duration=2.).key_metrics)
    # print(functs.utils.profiler_task(fait_task, "fait", run_duration=2.).key_metrics)





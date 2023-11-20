import time

import torch
from torch.profiler import profile, ProfilerActivity
import functs

import yolact_mask
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--bs", default=1, type=int)
arguments = parser.parse_args()

torch.cuda.init()

# type hint
cls_scores: [torch.Size([1, 255, 10, 10]), torch.Size([1, 255, 20, 20]), torch.Size([1, 255, 40, 40])]
type_hint = [
    torch.TupleType([torch.TensorType.get().with_dtype(torch.float32).with_sizes([ arguments.bs, 12, 40, 40 ]).with_device(torch.device("cuda")),
                     torch.TensorType.get().with_dtype(torch.float32).with_sizes([ arguments.bs, 12, 20, 20 ]).with_device(torch.device("cuda")),
                     torch.TensorType.get().with_dtype(torch.float32).with_sizes([ arguments.bs, 12, 10, 10 ]).with_device(torch.device("cuda")),
                     torch.TensorType.get().with_dtype(torch.float32).with_sizes([ arguments.bs, 12, 5, 5 ]).with_device(torch.device("cuda")),
                     torch.TensorType.get().with_dtype(torch.float32).with_sizes([ arguments.bs, 12, 3, 3 ]).with_device(torch.device("cuda")),]),         
    torch.TupleType([torch.TensorType.get().with_dtype(torch.float32).with_sizes([ arguments.bs, 243, 40, 40 ]).with_device(torch.device("cuda")),
                     torch.TensorType.get().with_dtype(torch.float32).with_sizes([ arguments.bs, 243, 20, 20 ]).with_device(torch.device("cuda")),
                     torch.TensorType.get().with_dtype(torch.float32).with_sizes([ arguments.bs, 243, 10, 10 ]).with_device(torch.device("cuda")),
                     torch.TensorType.get().with_dtype(torch.float32).with_sizes([ arguments.bs, 243, 5, 5 ]).with_device(torch.device("cuda")),
                     torch.TensorType.get().with_dtype(torch.float32).with_sizes([ arguments.bs, 243, 3, 3 ]).with_device(torch.device("cuda")),]),
    torch.TupleType([torch.TensorType.get().with_dtype(torch.float32).with_sizes([ arguments.bs, 96, 40, 40 ]).with_device(torch.device("cuda")),
                     torch.TensorType.get().with_dtype(torch.float32).with_sizes([ arguments.bs, 96, 20, 20 ]).with_device(torch.device("cuda")),
                     torch.TensorType.get().with_dtype(torch.float32).with_sizes([ arguments.bs, 96, 10, 10 ]).with_device(torch.device("cuda")),
                     torch.TensorType.get().with_dtype(torch.float32).with_sizes([ arguments.bs, 96, 5, 5 ]).with_device(torch.device("cuda")),
                     torch.TensorType.get().with_dtype(torch.float32).with_sizes([ arguments.bs, 96, 3, 3 ]).with_device(torch.device("cuda")),]),
    torch.TensorType.get().with_dtype(torch.float32).with_sizes([ arguments.bs, 32, 80, 80 ]).with_device(torch.device("cuda")),
]

def process_feat(feat):
    new_feat = []
    for data in feat:
        if isinstance(data, torch.Tensor):
            new_feat.append(data.repeat(arguments.bs, 1, 1, 1))
        else:
            new_data_tuple =[tensor_data.repeat(arguments.bs, 1, 1, 1) for tensor_data in data]
            new_feat.append(new_data_tuple)
    return tuple(new_feat)

def process_feat_batch(feats):
    new_feats = []
    for feat in feats:
        new_feats.append(process_feat(feat))
    return new_feats

# data load
model = yolact_mask.YolactBBoxMask().cuda().eval()

# torchscript
jit_model = torch.jit.freeze(torch.jit.script(model))

# nvfuser
nvfuser_model = torch.jit.freeze(torch.jit.script(model))

# functs
# functs_model = functs.jit.script(torch.jit.freeze(torch.jit.script(model)))

# torch dynamo + inductor
tracing_model = torch.compile(model)

# fait
fait_model = functs.jit.script(model, backend="fait")
functs._C._jit_pass_fait_pipeline(fait_model.graph, type_hint)
code = torch._C._jit_get_code(fait_model.graph)
print("done")

feats = torch.load("yolact_feat.pt")
feats = process_feat_batch(feats)
num_samples = len(feats)

code = torch._C._jit_get_code(fait_model.graph)

def eager_task(idx: int):
    model(*feats[idx % num_samples])

def jit_task(idx: int):
    jit_model(*feats[idx % num_samples])

# def functs_task(idx: int):
#     functs_model(*feats[idx % num_samples])

def tracing_task(idx: int):
    tracing_model(*feats[idx % num_samples])

def nvfuser_task(idx: int):
    nvfuser_model(*feats[idx % num_samples])

def fait_task(idx: int):
    torch._C._jit_run_code(code, ("", ) + feats[idx % num_samples])

for i in range(num_samples):
    eager_task(i)
    jit_task(i)
    # functs_task(i)
    tracing_task(i)
    fait_task(i)


functs.utils.evaluate_task(eager_task, "eager", run_duration=2.)
functs.utils.evaluate_task(jit_task, "jit", run_duration=2.)
functs.utils.evaluate_task(tracing_task, "dynamo", run_duration=2.)
# functs.utils.evaluate_task(functs_task, "functs", run_duration=2.)
functs.utils.evaluate_task(fait_task, "fait", run_duration=2.)

# print(functs_model.graph_for(*feats[0]))

torch._C._jit_set_nvfuser_enabled(True)
functs.utils.evaluate_task(nvfuser_task, "nvfuser", run_duration=2.)
torch._C._jit_set_nvfuser_enabled(False)

# print(functs.utils.profiler_task(eager_task, "eager", run_duration=2.).key_metrics)
# print(functs.utils.profiler_task(jit_task, "jit", run_duration=2.).key_metrics)
# print(functs.utils.profiler_task(fait_task, "fait", run_duration=2.).key_metrics)
import time

import torch
from torch.profiler import profile, ProfilerActivity
import functs

import yolact_mask

torch.cuda.init()

# type hint
cls_scores: [torch.Size([1, 255, 10, 10]), torch.Size([1, 255, 20, 20]), torch.Size([1, 255, 40, 40])]
type_hint = [
    torch.TupleType([torch.TensorType.get().with_dtype(torch.float32).with_sizes([ 1, 12, 40, 40 ]).with_device(torch.device("cuda")),
                     torch.TensorType.get().with_dtype(torch.float32).with_sizes([ 1, 12, 20, 20 ]).with_device(torch.device("cuda")),
                     torch.TensorType.get().with_dtype(torch.float32).with_sizes([ 1, 12, 10, 10 ]).with_device(torch.device("cuda")),
                     torch.TensorType.get().with_dtype(torch.float32).with_sizes([ 1, 12, 5, 5 ]).with_device(torch.device("cuda")),
                     torch.TensorType.get().with_dtype(torch.float32).with_sizes([ 1, 12, 3, 3 ]).with_device(torch.device("cuda")),]),         
    torch.TupleType([torch.TensorType.get().with_dtype(torch.float32).with_sizes([ 1, 243, 40, 40 ]).with_device(torch.device("cuda")),
                     torch.TensorType.get().with_dtype(torch.float32).with_sizes([ 1, 243, 20, 20 ]).with_device(torch.device("cuda")),
                     torch.TensorType.get().with_dtype(torch.float32).with_sizes([ 1, 243, 10, 10 ]).with_device(torch.device("cuda")),
                     torch.TensorType.get().with_dtype(torch.float32).with_sizes([ 1, 243, 5, 5 ]).with_device(torch.device("cuda")),
                     torch.TensorType.get().with_dtype(torch.float32).with_sizes([ 1, 243, 3, 3 ]).with_device(torch.device("cuda")),]),
    torch.TupleType([torch.TensorType.get().with_dtype(torch.float32).with_sizes([ 1, 96, 40, 40 ]).with_device(torch.device("cuda")),
                     torch.TensorType.get().with_dtype(torch.float32).with_sizes([ 1, 96, 20, 20 ]).with_device(torch.device("cuda")),
                     torch.TensorType.get().with_dtype(torch.float32).with_sizes([ 1, 96, 10, 10 ]).with_device(torch.device("cuda")),
                     torch.TensorType.get().with_dtype(torch.float32).with_sizes([ 1, 96, 5, 5 ]).with_device(torch.device("cuda")),
                     torch.TensorType.get().with_dtype(torch.float32).with_sizes([ 1, 96, 3, 3 ]).with_device(torch.device("cuda")),]),
    torch.TensorType.get().with_dtype(torch.float32).with_sizes([ 1, 32, 80, 80 ]).with_device(torch.device("cuda")),
]

# data load
model = yolact_mask.YolactBBoxMask().cuda().eval()

# torchscript
jit_model = torch.jit.freeze(torch.jit.script(model))

# functs
functs_model = functs.jit.script(torch.jit.freeze(torch.jit.script(model)))

# torch dynamo + inductor
tracing_model = torch.compile(model)

# fait
fait_model = functs.jit.script(model, backend="fait")
functs._C._jit_pass_fait_pipeline(fait_model.graph, type_hint)
code = torch._C._jit_get_code(fait_model.graph)
print("done")

feats = torch.load("yolact_feat.pt")
num_samples = len(feats)

code = torch._C._jit_get_code(fait_model.graph)

def eager_task(idx: int):
    model(*feats[idx % num_samples])

def jit_task(idx: int):
    jit_model(*feats[idx % num_samples])

def functs_task(idx: int):
    functs_model(*feats[idx % num_samples])

def tracing_task(idx: int):
    tracing_model(*feats[idx % num_samples])

def fait_task(idx: int):
    torch._C._jit_run_code(code, ("", ) + feats[idx % num_samples])

for i in range(num_samples):
    eager_task(i)
    jit_task(i)
    functs_task(i)
    tracing_task(i)
    fait_task(i)


functs.utils.evaluate_task(eager_task, "eager", run_duration=2.)
functs.utils.evaluate_task(jit_task, "jit", run_duration=2.)
functs.utils.evaluate_task(fait_task, "fait", run_duration=2.)

print(functs.utils.profiler_task(eager_task, "eager", run_duration=2.).key_metrics)
print(functs.utils.profiler_task(jit_task, "jit", run_duration=2.).key_metrics)
print(functs.utils.profiler_task(fait_task, "fait", run_duration=2.).key_metrics)

import time

import torch
import functs

import ssd_bbox

torch.cuda.init()

# type hint
cls_scores: [torch.Size([1, 486, 20, 20]), torch.Size([1, 486, 10, 10]), torch.Size([1, 486, 5, 5]), torch.Size([1, 486, 3, 3]), torch.Size([1, 486, 2, 2]), torch.Size([1, 486, 1, 1])]
bbox_preds: [torch.Size([1, 24, 20, 20]), torch.Size([1, 24, 10, 10]), torch.Size([1, 24, 5, 5]), torch.Size([1, 24, 3, 3]), torch.Size([1, 24, 2, 2]), torch.Size([1, 24, 1, 1])]
type_hint = [torch.TupleType([torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 486, 20, 20]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 486, 10, 10]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 486, 5, 5]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 486, 3, 3]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 486, 2, 2]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 486, 1, 1]).with_device(torch.device("cuda"))],),
             torch.TupleType([torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 24, 20, 20]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 24, 10, 10]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 24, 5, 5]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 24, 3, 3]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 24, 2, 2]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 24, 1, 1]).with_device(torch.device("cuda"))])]

# data load
model = ssd_bbox.SSDBBox().cuda().eval()

# torchscript nvfuser
jit_model = torch.jit.freeze(torch.jit.script(model))

# torchscript nnc
nvfuser_model = torch.jit.freeze(torch.jit.script(model))

# functs
functs_model = functs.jit.script(torch.jit.freeze(torch.jit.script(model)))


# torch dynamo + inductor
tracing_model = torch.compile(model, dynamic=True)

# fait
# fait_model = functs.jit.script(model, backend="fait")
# functs._C._jit_pass_fait_pipeline(fait_model.graph, type_hint)
# code = torch._C._jit_get_code(fait_model.graph)
print("done")

feats = torch.load("ssd_feat.pt")
num_samples = len(feats)

# code = torch._C._jit_get_code(fait_model.graph)

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
    torch._C._jit_run_code(code, ("", ) + feats[idx % num_samples])

functs.utils.evaluate_task(eager_task, "eager", run_duration=2.)
functs.utils.evaluate_task(jit_task, "jit", run_duration=2.)
functs.utils.evaluate_task(functs_task, "functs", run_duration=2.)
functs.utils.evaluate_task(tracing_task, "dynamo", run_duration=2.)
functs.utils.evaluate_task(fait_task, "fait", run_duration=2.)

# # nvfuser
torch._C._jit_set_nvfuser_enabled(True)
functs.utils.evaluate_task(nvfuser_task, "nvfuser", run_duration=2.)
torch._C._jit_set_nvfuser_enabled(False)

# print(functs.utils.profiler_task(eager_task, "eager", run_duration=2.).key_metrics)
# print(functs.utils.profiler_task(jit_task, "jit", run_duration=2.).key_metrics)
# print(functs.utils.profiler_task(fait_task, "fait", run_duration=2.).key_metrics)


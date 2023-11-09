import time

import torch
import functs

import yolov3_bbox

torch.cuda.init()

# pred_maps = [torch.Size([1, 255, 10, 10]), torch.Size([1, 255, 20, 20]), torch.Size([1, 255, 40, 40])]

model = yolov3_bbox.YOLOV3BBox().cuda().eval()


# torchscript
jit_model = torch.jit.freeze(torch.jit.script(model))

# torch dynamo + inductor
torch._dynamo.reset()
tracing_model = torch.compile(model)

# fait
cls_scores: [torch.Size([1, 486, 20, 20]), torch.Size([1, 486, 10, 10]), torch.Size([1, 486, 5, 5]), torch.Size([1, 486, 3, 3]), torch.Size([1, 486, 2, 2]), torch.Size([1, 486, 1, 1])]
bbox_preds: [torch.Size([1, 24, 20, 20]), torch.Size([1, 24, 10, 10]), torch.Size([1, 24, 5, 5]), torch.Size([1, 24, 3, 3]), torch.Size([1, 24, 2, 2]), torch.Size([1, 24, 1, 1])]
type_hint = [torch.TupleType([torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 255, 10, 10]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 255, 20, 20]).with_device(torch.device("cuda")),
                              torch.TensorType.get().with_dtype(torch.float32).with_sizes([1, 255, 40, 40]).with_device(torch.device("cuda"))],)]

feats = torch.load("yolov3_feat.pt")
num_samples = len(feats)

functs_model = functs.jit.script(torch.jit.freeze(torch.jit.script(model)))
fait_model = functs.jit.script(model, backend="fait")
functs._C._jit_pass_fait_pipeline(fait_model.graph, type_hint)

code = torch._C._jit_get_code(fait_model.graph)

torch._C._jit_set_texpr_dynamic_shape_enabled(True)

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

def dump_proflier(task, name):
    functs.utils.evaluate_task(task, name=name)

torch.cuda.profiler.start()
dump_proflier(eager_task, "eager")
dump_proflier(jit_task, "jit")
dump_proflier(functs_task, "functs")
dump_proflier(tracing_task, "tracing jit")
dump_proflier(fait_task, "fait")
torch.cuda.profiler.stop()





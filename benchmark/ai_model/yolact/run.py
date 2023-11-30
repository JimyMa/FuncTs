import os
import time

import torch
import functs

import yolact_mask

import argparse

torch.cuda.init()

parser = argparse.ArgumentParser()
parser.add_argument("--bs", default=1, type=int)
arguments = parser.parse_args()

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

feats = torch.load(os.path.join(os.path.dirname(__file__), "yolact_feat.pt"))
feats = process_feat_batch(feats)
num_samples = len(feats)

with torch.no_grad():
    model = yolact_mask.YolactBBoxMask().cuda().eval()
    # torchscript
    jit_model = torch.jit.freeze(torch.jit.script(model))

    # nvfuser
    nvfuser_model = torch.jit.freeze(torch.jit.script(model))

    # torch dynamo + inductor
    torch._dynamo.reset()
    dynamo_model = torch.compile(model, dynamic=True)

    # functs
    functs_model = functs.jit.script(model)

    # aot backend
    fait_model = functs.jit.build(functs.jit.script(model, backend="aot"), feats[0]) 

    task = lambda fn: lambda _: fn(*feats[0 % num_samples])

    functs.utils.evaluate_task(task(model), "eager", run_duration=2.)
    functs.utils.evaluate_task(task(jit_model), "jit", run_duration=2.)
    functs.utils.evaluate_task(task(functs_model), "functs", run_duration=2.)
    functs.utils.evaluate_task(
        task(dynamo_model), "dynamo+inductor", run_duration=2.)
    functs.utils.evaluate_task(task(fait_model), "fait", run_duration=2.)

    # nvfuser
    torch._C._jit_set_nvfuser_enabled(True)
    functs.utils.evaluate_task(task(nvfuser_model), "nvfuser", run_duration=2.)
    torch._C._jit_set_nvfuser_enabled(False)


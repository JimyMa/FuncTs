import time
from typing import List

import torch

import functs


class BBoxTargetExpand(torch.nn.Module):
    def forward(self, bbox_targets, bbox_weights, labels):
        bbox_targets_expand = bbox_targets.clone()
        bbox_weights_expand = bbox_weights.clone()
        valid_label: List[int] = torch.nonzero(labels > 0).squeeze(-1).tolist()
        for i in valid_label:
            bbox_targets_expand[i] = bbox_targets[i]
            bbox_weights_expand[i] = bbox_weights[i]
        return bbox_targets_expand.clone(), bbox_weights_expand.clone()


eager_fn = BBoxTargetExpand()
jit_fn = torch.jit.script(BBoxTargetExpand())
functs_fn = functs.jit.script(BBoxTargetExpand())

M = 300
N = 4

bbox_targets = torch.rand(M, N).float().cuda()
bbox_weights = torch.rand(M, N).float().cuda()
labels = torch.ones(M, ).long().cuda()


o_functs = functs_fn(bbox_targets, bbox_weights, labels)
o_jit = jit_fn(bbox_targets, bbox_weights, labels)
o_eager = eager_fn(bbox_targets, bbox_weights, labels)

print(torch.allclose(o_functs[0], o_eager[0], atol=1e-3))
print(torch.allclose(o_functs[1], o_eager[1], atol=1e-3))

print(functs_fn.graph_for(bbox_targets, bbox_weights, labels))

# warm up 0
for i in range(10):
    o_functs = functs_fn(bbox_targets, bbox_weights, labels)
    o_jit = jit_fn(bbox_targets, bbox_weights, labels)
    o_eager = eager_fn(bbox_targets, bbox_weights, labels)

# warm up 1
for i in range(10):
    o_functs = functs_fn(bbox_targets, bbox_weights, labels)
    o_jit = jit_fn(bbox_targets, bbox_weights, labels)
    o_eager = eager_fn(bbox_targets, bbox_weights, labels)


begin = time.time()
for i in range(100):
    o_functs = functs_fn(bbox_targets, bbox_weights, labels)
mid_0 = time.time()
for i in range(100):
    o_jit = jit_fn(bbox_targets, bbox_weights, labels)
mid_1 = time.time()
for i in range(100):
    o_eager = eager_fn(bbox_targets, bbox_weights, labels)
end = time.time()

print("functs: ", mid_0 - begin)
print("torchscript: ", mid_1 - mid_0)
print("eager: ", end - mid_1)



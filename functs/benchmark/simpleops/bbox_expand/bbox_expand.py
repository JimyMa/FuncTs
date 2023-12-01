import time
from typing import List

import torch

import functs


class BBoxTargetExpand(torch.nn.Module):
    def forward(self, bbox_targets, bbox_weights, labels: List[int]):
        bbox_targets_expand = bbox_targets.clone()
        bbox_weights_expand = bbox_weights.clone()
        valid_label: List[int] = labels
        for i in valid_label:
            bbox_targets_expand[i, :] = bbox_targets[i, :]
            bbox_weights_expand[i, :] = bbox_weights[i, :]
        return bbox_targets_expand.clone(), bbox_weights_expand.clone()


eager_fn = BBoxTargetExpand()
jit_fn = torch.jit.script(BBoxTargetExpand())
functs_fn = functs.jit.script(BBoxTargetExpand())

M = 300
N = 4

bbox_targets = torch.rand(M, N).float().cuda()
bbox_weights = torch.rand(M, N).float().cuda()
labels = [1, 2, 3, 4]


o_functs = functs_fn(bbox_targets, bbox_weights, labels)
o_jit = jit_fn(bbox_targets, bbox_weights, labels)
o_eager = eager_fn(bbox_targets, bbox_weights, labels)

print(torch.allclose(o_functs[0], o_eager[0], atol=1e-3))
print(torch.allclose(o_functs[1], o_eager[1], atol=1e-3))

functs.utils.evaluate_func(eager_fn, (bbox_targets, bbox_weights, labels), "eager", run_duration=2.0)
functs.utils.evaluate_func(jit_fn, (bbox_targets, bbox_weights, labels), "jit", run_duration=2.0)
functs.utils.evaluate_func(functs_fn, (bbox_targets, bbox_weights, labels), "functs", run_duration=2.0)

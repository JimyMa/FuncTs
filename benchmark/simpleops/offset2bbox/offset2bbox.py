from typing import Tuple

import torch


def offset2bbox(boxes: torch.Tensor, offset: torch.Tensor):
    """
    Forward transform that maps proposal boxes to predicted ground-truth
    boxes using bounding-box regression deltas(offset). See bbox_transform_inv
    for a description of the weights argument.
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    widths = (x2 - x1 + 1)
    heights = (y2 - y1 + 1)
    ctr_x = x1 + 0.5 * widths
    ctr_y = y1 + 0.5 * heights

    dx = offset[:, 0::4] / 1.0
    dy = offset[:, 1::4] / 1.0
    dw = offset[:, 2::4] / 1.0
    dh = offset[:, 3::4] / 1.0

    # Prevent sending too large values into np.exp()
    dw = torch.clamp(dw, max=4.135)
    dh = torch.clamp(dh, max=4.135)

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    pred_boxes = offset.clone()
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

    return pred_boxes


import functs._C
import torch
import torch._C._te as te

torch.cuda.init()

jit_func = torch.jit.script(offset2bbox)
g = jit_func.graph
torch._C._jit_pass_inline(g)

# profiling
# Step 1. Generate GraphIR and Functionalized GraphIR
functs._C._jit_pass_convert_to_tensorssa(g)
functs._C._jit_pass_tensorssa_remove_update(g)
torch._C._jit_pass_dce(g)
torch._C._jit_pass_cse(g)

a = torch.rand(3000, 4).float().cuda()
b = torch.rand(3000, 4).float().cuda()

# correctness
print(torch.allclose(jit_func(a, b), offset2bbox(a, b)))

torch._C._jit_pass_complete_shape_analysis(g, (a, b), False)

# dump graph to annotate shape information.
with open("offset2bbox.torchscript", "w") as f:
    f.write(repr(g))

print(jit_func.graph_for(a, b))





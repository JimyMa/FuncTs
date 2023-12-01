from typing import Tuple

import torch

import functs


class Offset2bbox(torch.nn.Module):
    def forward(self, boxes: torch.Tensor, offset: torch.Tensor):
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
        pred_boxes[:] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
        # y2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1
        return pred_boxes


eager_fn = Offset2bbox()
jit_fn = torch.jit.script(Offset2bbox())
functs_fn = functs.jit.script(Offset2bbox())
# print(functs_fn.graph)

a = torch.randn(3000, 4).float().cuda()
b = torch.randn(3000, 4).float().cuda()

o_functs = functs_fn(a, b)
o_jit = jit_fn(a, b)
o_eager = eager_fn(a, b)

print(torch.allclose(o_functs, o_eager, atol=1e-3))

functs.utils.evaluate_func(eager_fn, (a, b), "eager", run_duration=2.)
functs.utils.evaluate_func(jit_fn, (a, b), "jit", run_duration=2.)
functs.utils.evaluate_func(functs_fn, (a, b), "functs", run_duration=2.)



from typing import Tuple

import pydot

import torch
import functs._C


def xyxy2xywh(boxes):
    """(x1, y1, x2, y2) -> (x, y, w, h)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = (x2 - x1 + 1)
    h = (y2 - y1 + 1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return cx, cy, w, h


def offset2bbox(boxes, offset, weights: Tuple[float, float, float, float]=(1.0, 1.0, 1.0, 1.0)):
    """
    Forward transform that maps proposal boxes to predicted ground-truth
    boxes using bounding-box regression deltas(offset). See bbox_transform_inv
    for a description of the weights argument.
    """
    ctr_x, ctr_y, widths, heights = xyxy2xywh(boxes)

    wx, wy, ww, wh = weights
    dx = offset[:, 0::4] / wx
    dy = offset[:, 1::4] / wy
    dw = offset[:, 2::4] / ww
    dh = offset[:, 3::4] / wh

    # Prevent sending too large values into np.exp()
    dw = torch.clamp(dw, max=torch.log(torch.tensor(1000. / 16.)))
    dh = torch.clamp(dh, max=torch.log(torch.tensor(1000. / 16.)))

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    pred_boxes = offset.new_zeros(offset.shape)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

    return pred_boxes


def dump_graph_to_file(graph:torch.Graph, file_name:str):
    with open(file_name, "w") as f:
        f.write(repr(g))
    return


func = offset2bbox
jit_func = torch.jit.script(func)
g = jit_func.graph
torch._C._jit_pass_inline(g)
print("\033[1;32;40mOrigin Graph:")
print(g)
print("\033[0m")
dump_graph_to_file(g, "ori.rb")

dot_graph, = pydot.graph_from_dot_data(g.alias_db().to_graphviz_str())
dot_graph.write_png("alias_vis/func_alias.png")

functs._C._jit_pass_convert_to_tensorssa(g)
print("\033[1;36;40mGraph after tensorssa")
print(g)
print("\033[0m")

dump_graph_to_file(g, "tssa.rb")

functs._C._jit_pass_tensorssa_remove_update(g)
torch._C._jit_pass_dce(g)
print("\033[1;34;40mGraph after dce")
print(g)
print("\033[0m")
dump_graph_to_file(g, "dce.rb")

dot_graph, = pydot.graph_from_dot_data(g.alias_db().to_graphviz_str())
dot_graph.write_png("alias_vis/func_alias_after_tensorssa.png")

a = torch.rand([1024, 1024])
b = torch.rand([1024, 1024])
weight: Tuple[float, float, float, float] = (1., 1., 1., 1.)

torch._C._jit_pass_complete_shape_analysis(g, (a, b, weight), False)
torch._C._jit_pass_fuse_tensorexprs(g)
print(g)
# torch._C._jit_pass_propagate_shapes_on_graph(g)

# torch._C._jit_pass_fuse_tensorexprs(g)



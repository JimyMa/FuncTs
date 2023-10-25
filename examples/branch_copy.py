import pydot

import torch
from torch.testing import FileCheck

import functs._C


def func(a: torch.Tensor, c: torch.Tensor, d: torch.Tensor):
    tmp_0 = d
    if (a.nonzero() > 1):
        x = tmp_0 + d
        
        tmp_0.add_(tmp_0[0])
    else:
        
        tmp_0.copy_(tmp_0[1])
    tmp_0.add_(d[0])
    return tmp_0


func = torch.jit.script(func)
g = func.graph

print(g)

# functs._C._jit_pass_remove_inplace(func.graph)
# print(g)

# functs._C._jit_pass_dump_remove_inter_precedure_mutation(func.graph)
# print(g)

dot_graph, = pydot.graph_from_dot_data(g.alias_db().to_graphviz_str())
dot_graph.write_png("alias_vis/branch_alias.png")

functs._C._jit_pass_convert_to_tensorssa(func.graph)

buffer_tree, = pydot.graph_from_dot_file("buffer_tree.dot")
buffer_tree.write_png("alias_vis/branch_alias_buffer_tree.png")

dot_graph, = pydot.graph_from_dot_data(g.alias_db().to_graphviz_str())
dot_graph.write_png("alias_vis/branch_alias_after_alias_removal.png")

torch._C._jit_pass_dce(g)

g.alias_db().dump()
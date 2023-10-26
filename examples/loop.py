import pydot
import torch

import functs._C

def loop(a: torch.Tensor, b: torch.Tensor):
    for i in range(10):
        a[i].add_(b[i+1])
    return a


func = torch.jit.script(loop)
g = func.graph

print(g)

dot_graph, = pydot.graph_from_dot_data(g.alias_db().to_graphviz_str())
dot_graph.write_png("alias_vis/loop_alias.png")

dot_graph, = pydot.graph_from_dot_data(g.alias_db().to_graphviz_str())
dot_graph.write_png("alias_vis/branch_alias.png")

functs._C._jit_pass_convert_to_tensorssa(func.graph)
# print(g)
g.alias_db().dump()

buffer_tree, = pydot.graph_from_dot_file("buffer_tree.dot")
buffer_tree.write_png("alias_vis/branch_alias_buffer_tree.png")

dot_graph, = pydot.graph_from_dot_data(g.alias_db().to_graphviz_str())
dot_graph.write_png("alias_vis/branch_alias_after_alias_removal.png")

torch._C._jit_pass_dce(g)

g.alias_db().dump()





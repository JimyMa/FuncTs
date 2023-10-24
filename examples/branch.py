import pydot

import torch

import functs._C


def func(a: torch.Tensor, c: torch.Tensor):
    tmp_0 = a + c
    tmp_1 = tmp_0[1]
    if (a.nonzero() > 1):
        b = a[0]
    else:
        b = c[1]
    tmp_1.copy_(b)
    return b + tmp_1


func = torch.jit.script(func)
print(func.graph)
func.graph.dump_alias_db()
dot_graph, = pydot.graph_from_dot_data(func.graph.alias_db().to_graphviz_str())
dot_graph.write_png("branch_alias.png")

functs._C._jit_pass_convert_to_tensorssa(func.graph)

buffer_tree, = pydot.graph_from_dot_file("buffer_tree.dot")
buffer_tree.write_png("branch_alias_buffer_tree.png")

import pydot

import torch

import functs._C


def func(a: torch.Tensor, c: torch.Tensor):
    x = a + c
    y = x[0]
    if (a.nonzero() > 1):
        b = a[0]
    else:
        b = c[1]
    b.add_(0.5)
    q = []
    q.append(y)
    q.append(b)
    return q


func = torch.jit.script(func)
print(func.graph)
func.graph.dump_alias_db()
dot_graph, = pydot.graph_from_dot_data(func.graph.alias_db().to_graphviz_str())
dot_graph.write_png("branch_alias.png")

functs._C._jit_pass_convert_to_tensorssa(func.graph)


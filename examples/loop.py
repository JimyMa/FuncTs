import pydot
import torch

def loop(a: torch.Tensor, b: torch.Tensor):
    list_ = []
    for i in range(10):
        a += b
        list_.append(a)
    list_.append(a)
    return list_


func = torch.jit.script(loop)
g = func.graph

print(g)

dot_graph, = pydot.graph_from_dot_data(g.alias_db().to_graphviz_str())
dot_graph.write_png("alias_vis/loop_alias.png")







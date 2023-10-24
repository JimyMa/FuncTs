import pydot
import torch


def list_object(a: torch.Tensor, b: torch.Tensor):
    list_ = []
    list_.append(a)
    list_.append(b)
    return list_


func = torch.jit.script(list_object)
g = func.graph

print(g)

dot_graph, = pydot.graph_from_dot_data(g.alias_db().to_graphviz_str())
dot_graph.write_png("alias_vis/container_alias.png")

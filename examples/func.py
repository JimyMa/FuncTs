import pydot

import torch
from torch.testing import FileCheck

import functs._C

def normal(a:torch.Tensor, b: torch.Tensor):
    a[0].copy_(b[0])
    a[1].copy_(b[1])
    return a

def loop(a: torch.Tensor, b: torch.Tensor):
    for i in range(10):
        if (b.sum() <= 0):
            a[i].add_(b[i+1])
        else:
            a[i].add_(b[i])
        a[i+1].add_(b[i+1])
    return a

def branch(a: torch.Tensor, c: torch.Tensor, d: torch.Tensor):
    tmp_0 = d
    tmp_1 = tmp_0[1]
    if (a.nonzero() > 1):
        tmp_0 = tmp_0 + 1
    else:
        tmp_1.copy_(tmp_0[2])
    return tmp_1

def branch_loop(a: torch.Tensor, c: torch.Tensor, d: torch.Tensor):
    tmp_0 = d
    tmp_1 = tmp_0[1]
    if (a.nonzero() > 1):
        tmp_0 = tmp_0 + 1
        for i in range(10):
            tmp_1.copy_(tmp_0[i+1])
    else:
        tmp_0 = tmp_0 +tmp_1[1]
        tmp_1.copy_(tmp_0[2])
    return tmp_1 + tmp_0


def branch_with_dependency(a: torch.Tensor, c: torch.Tensor, d: torch.Tensor):
    tmp_0 = d
    tmp_1 = tmp_0[1]
    if (a.nonzero() > 1):
        tmp_0 = tmp_0 + 1
    else:
        tmp_0 = tmp_1 + tmp_0
        tmp_1.copy_(tmp_0[2])
    return tmp_1 + tmp_0

def list_object(a: torch.Tensor, b: torch.Tensor):
    list_ = []
    list_.append(a)
    list_.append(b)
    return list_


def dump_graph_to_file(graph:torch.Graph, file_name:str):
    with open(file_name, "w") as f:
        f.write(repr(g))
    return


func = branch_with_dependency
jit_func = torch.jit.script(func)
g = jit_func.graph
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


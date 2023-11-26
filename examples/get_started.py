import torch
import torch._C._te as te
import functs._C

from typing import List, Tuple


# substitute your own function here~
def func(a: torch.Tensor, b: torch.Tensor, n: int):
  a = a.clone()
  b = b.clone()
  for i in range(n):
    b[i] = b[i] + 1
  return b

def func_data_control_flow_dependency(a_list: List[Tuple[torch.Tensor, torch.Tensor]], 
                                      idx: int):
  a = a_list[0][0].clone()
  b = a_list[0][1].clone()
  b += 1
  b = b.copy_(b + 1)
  if idx >= 0:
    a += 1
    b[idx].copy_(a[idx])
  else:
    a -= 1
    b[-idx].copy_(a[-idx])
  return a + b

# func = func_data_control_flow_dependency
func = func
jit_func = torch.jit.script(func)
functs_func = functs.jit.script(func)

# torchscript
print("graph before functionalization")
jit_func.graph.alias_db().dump()

# functs
print("graph after functionalization")
functs_func.graph.alias_db().dump()

# check equal
a: torch.Tensor = torch.randn([1024, 1024])
b: torch.Tensor = torch.randn([1024, 1024])
n = 3

print(torch.allclose(jit_func(a, b, 3), func(a, b, 3)))











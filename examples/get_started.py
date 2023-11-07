import torch
import torch._C._te as te
import functs._C



# substitute your own function here~
def func(a: torch.Tensor, b: torch.Tensor, n: int):
  a = a.clone()
  b = b.clone()
  b_list = []
  for i in range(n):
    b[i] = b[i] + 1
    b[i].copy_(b[i] + 1)
    b_list.append(b[i])
  return b_list

def func_data_control_flow_dependency(a: torch.Tensor, 
                                      b: torch.Tensor, 
                                      idx: int):
  a = a.clone()
  b = b.clone()
  b += 1
  b = b.copy_(b + 1)
  if idx >= 0:
    a += 1
    b[idx].copy_(a[idx])
  else:
    a -= 1
    b[-idx].copy_(a[-idx])
  return a + b

func = func_data_control_flow_dependency


jit_func = torch.jit.script(func)
functs._C._jit_pass_remove_inplace(jit_func.graph)
c = jit_func.code
g = jit_func.graph
g.alias_db().dump()
print(g.str(False))

# to torchscript
jit_func = functs.jit.script(func)
c = jit_func.code
g = jit_func.graph
print(c)
g.alias_db().dump()

# check equal
a: torch.Tensor = torch.randn([1024, 1024])
b: torch.Tensor = torch.randn([1024, 1024])
n = 3

print(torch.allclose(jit_func(a, b, 3), func(a, b, 3)))









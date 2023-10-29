import torch
import torch._C._te as te
import functs._C



# substitute your own function here~
def func(a: torch.Tensor, b: torch.Tensor):
  for i in range(10):
    b[i] += 1
    a[i].copy_(b[i+1])
  return a

# to torchscript
jit_func = torch.jit.script(func)
g = jit_func.graph

# Note: only intra-procedure is supported!!!
torch._C._jit_pass_inline(g)

# tensorssa alias removal
functs._C._jit_pass_convert_to_tensorssa(g)
functs._C._jit_pass_tensorssa_remove_update(g)

# dce
torch._C._jit_pass_dce(g)
print(g)

# check equal
a: torch.Tensor = torch.randn([1024, 1024])
b: torch.Tensor = torch.randn([1024, 1024])

print(torch.allclose(jit_func(a, b), func(a, b)))
import pprint
from typing import List, Tuple

import torch
import torch._C._te as te
import functs._C


# substitute your own function here~
def func(a: torch.Tensor, b: torch.Tensor, n: int):
  a = a.clone()
  b = b.clone()
  for i in range(n):
    b[i] = b[i] + 1
  return b

def func_data_control_flow_dependency(a: torch.Tensor, b: torch.Tensor, 
                                      idx: int):
  a = a.clone()
  b = b.clone()
  if idx >= 0:
    a = a + 1
    b[idx].copy_(a[idx])
  else:
    a = a - 1
    b[-idx].copy_(a[-idx])
  return a + b

func = func
# func = func_data_control_flow_dependency

# torchscript
jit_func = torch.jit.script(func)
print("graph before functionalization")
print("original graph: ")
jit_func.graph.alias_db().dump()

# step 1: rewrite mutation
mutate_info = functs._C.TensorSSAMutateInfo()
functs._C._jit_pass_rewrite_mutation(jit_func.graph, mutate_info)
print("graph after rewrite mutation")
print(jit_func.graph)
print("mutated values: ") 
print(mutate_info.mutValues)
print("mutated nodes: ")
print(mutate_info.mutNodes)

# step 2: block propagation
functs._C._jit_pass_block_propagation(jit_func.graph, mutate_info)
print("graph after block propagation")
print(jit_func.graph)

# step 3: rename
functs._C._jit_pass_rename(jit_func.graph)
print("graph after rename according tensorssa::Update")
print(jit_func.graph)

# step 4: remove update
functs._C._jit_pass_tensorssa_remove_update(jit_func.graph)
print("graph after remove update")
print(jit_func.graph)

# step 5: cse, dce, constant_propagation
torch._C._jit_pass_cse(jit_func.graph)
torch._C._jit_pass_dce(jit_func.graph)
torch._C._jit_pass_constant_propagation(jit_func.graph)
print("after csd, dce and constant propagation")
jit_func.graph.alias_db().dump()

# check equal
a: torch.Tensor = torch.randn([1024, 1024])
b: torch.Tensor = torch.randn([1024, 1024])
n = 3

print(torch.allclose(jit_func(a, b, 3), func(a, b, 3)))











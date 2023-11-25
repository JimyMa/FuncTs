import torch
import torch._C._te as te
import functs._C

from typing import List, Tuple



# substitute your own function here~
# def func(a: torch.Tensor, b: torch.Tensor, n: int):
#   a = a.clone()
#   b = b.clone()
#   for i in range(n):
#     b[i] = b[i] + 1
#   return b
class Func(torch.nn.Module):
  def forward(self, a, b, idx: int):
    a = a.clone()
    b = b.clone()
    # a = a_list[0][0].clone()
    # b = a_list[0][1].clone()
    if idx >= 0:
      a += 1
      b[0] = a[0]
    else:
      a -= 1
      b[0] = a[0]
    return a + b

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

func = Func()
jit_func = functs.jit.script(func.cuda().eval())

aot_function = functs._C._create_function_from_graph("forward", jit_func.graph)

a = torch.randn([10]).cuda()
b = torch.randn([10]).cuda()

print(func(a, b, 1))
print(aot_function("", a, b, 1))






# type_hint = functs._C._jit_get_code(jit_func.graph, [torch.TensorType.get().with_dtype(torch.float32).with_sizes([0, 255, 10, 10]).with_device(torch.device("cuda")),
#                                                      torch.TensorType.get().with_dtype(torch.float32).with_sizes([0, 255, 20, 20]).with_device(torch.device("cuda")),
#                                                      torch.int32])

# functs._C._jit_pass_fait_pipeline(jit_func, type_hint)

# code = functs._C._jit_get_code(jit_func.graph)
# a = functs._C._jit_run_code(code, ("", [(torch.randn([10]), torch.randn([10]))], 1))

# print(a)
# functs._C._jit_pass_remove_inplace(jit_func.graph)
# c = jit_func.code
# g = jit_func.graph
# g.alias_db().dump()
# print(g.str(False))

# # to torchscript
# jit_func = functs.jit.script(func)
# c = jit_func.code
# g = jit_func.graph
# print(c)
# g.alias_db().dump()

# # check equal
# a: torch.Tensor = torch.randn([1024, 1024])
# b: torch.Tensor = torch.randn([1024, 1024])
# n = 3

# func = functs._C._create_function_from_graph("func", g)
# # print(func)
# # print(func(a, b, n))
# print(func.graph)


# print(torch.allclose(jit_func(a, b, 3), func(a, b, 3)))











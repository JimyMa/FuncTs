import time

import torch

import functs

import tvm.te as te


class Normalize(torch.nn.Module):
    def forward(self, 
                src: torch.Tensor,
                mean: float, scale: float):
        # RGB to BGR
        dup = src.clone()
        dup[..., 0] = src[..., 2]
        dup[..., 2] = src[..., 0]
        return (dup - mean) * scale



eager_fn = Normalize()
jit_fn = torch.jit.script(Normalize())
functs_fn = functs.jit.script(Normalize())
# fait
fait_fn = functs.jit.script(Normalize())
tensor_type = torch.TensorType.get().with_device(torch.device("cuda")).with_sizes([800, 1333, 3]).with_dtype(torch.float32)
functs._C._jit_pass_fait_pipeline(fait_fn.graph, [tensor_type, torch.FloatType.get(), torch.FloatType.get()])
fait_code = torch._C._jit_get_code(fait_fn.graph)
print(jit_fn.graph.str(False))
print(functs_fn.graph.str(False))
print(fait_fn.graph.str(False))

a = torch.randn(800, 1333, 3).float().cuda()
mean = 0.0
scale = 1.0

import torch._C._te as te
tensorexpr_str = """
graph(%src.1 : Float(800, 1333, 3, strides=[3999, 3, 1], device=cuda:0),
      %mean.1 : float,
      %scale.1 : float):
  %9 : int = prim::Constant[value=0]() # /home/jimyma/project/TensorSSA/FuncTs/benchmark/simpleops/normalize/normalize.py:14:17
  %8 : int = prim::Constant[value=2]() # /home/jimyma/project/TensorSSA/FuncTs/benchmark/simpleops/normalize/normalize.py:14:31
  %7 : NoneType = prim::Constant()
  %6 : int = prim::Constant[value=-1]() # /home/jimyma/project/TensorSSA/FuncTs/benchmark/simpleops/normalize/normalize.py:14:22
  %4 : int = prim::Constant[value=1]()
  %dup.1 : Float(800, 1333, 3, strides=[3999, 3, 1], device=cuda:0) = aten::clone(%src.1, %7) # /home/jimyma/project/TensorSSA/FuncTs/benchmark/simpleops/normalize/normalize.py:13:14
  %11 : Float(800, 1333, strides=[1333, 1], device=cuda:0) = immut::select(%src.1, %6, %8) # /home/jimyma/project/TensorSSA/FuncTs/benchmark/simpleops/normalize/normalize.py:14:22
  %24 : Float(800, 1333, 3, strides=[3999, 3, 1], device=cuda:0) = immut::select_rev(%dup.1, %11, %6, %9)
  %14 : Float(800, 1333, strides=[1333, 1], device=cuda:0) = immut::select(%src.1, %6, %9) # /home/jimyma/project/TensorSSA/FuncTs/benchmark/simpleops/normalize/normalize.py:15:22
  %28 : Float(800, 1333, 3, strides=[3999, 3, 1], device=cuda:0) = immut::select_rev(%24, %14, %6, %8)
  %17 : Float(800, 1333, 3, strides=[3999, 3, 1], device=cuda:0) = aten::sub(%28, %mean.1, %4) # /home/jimyma/project/TensorSSA/FuncTs/benchmark/simpleops/normalize/normalize.py:16:16
  %18 : Float(800, 1333, 3, strides=[3999, 3, 1], device=cuda:0) = aten::mul(%17, %scale.1) # /home/jimyma/project/TensorSSA/FuncTs/benchmark/simpleops/normalize/normalize.py:16:16
  return (%18)
"""

torch.cuda.init()
te_g = torch.parse_ir(tensorexpr_str)
kernel = te.TensorExprKernel(te_g)
print(kernel.get_code_text())

o_functs = functs_fn(a, mean, scale)
o_jit = jit_fn(a, mean, scale)
o_eager = eager_fn(a, mean, scale)

stack = torch._C._jit_run_code(fait_code, ("", a, mean, scale))

print(torch.allclose(o_functs, o_eager, atol=1e-3))
print(torch.allclose(stack, o_eager, atol=1e-3))


# warm up 0
for i in range(10):
    o_functs = functs_fn(a, mean, scale)
    o_jit = jit_fn(a, mean, scale)
    o_eager = eager_fn(a, mean, scale)
    torch._C._jit_run_code(fait_code, ("", a, mean, scale))

# warm up 1
for i in range(10):
    o_functs = functs_fn(a, mean, scale)
    o_jit = jit_fn(a, mean, scale)
    o_eager = eager_fn(a, mean, scale)
    torch._C._jit_run_code(fait_code, ("", a, mean, scale))

torch.cuda.profiler.start()
begin = time.time()
for i in range(1000):
    o_functs = functs_fn(a, mean, scale)
mid_0 = time.time()
for i in range(1000):
    torch._C._jit_run_code(fait_code, ("", a, mean, scale))
mid_1 = time.time()
for i in range(1000):
    o_jit = jit_fn(a, mean, scale)
mid_2 = time.time()
for i in range(1000):
    o_eager = eager_fn(a, mean, scale)
end = time.time()
torch.cuda.profiler.stop()

print("functs: ", mid_0 - begin)
print("fait: ", mid_1 - mid_0)
print("torchscript: ", mid_2 - mid_1)
print("eager: ", end - mid_2)


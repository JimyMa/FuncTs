import torch
from torch.profiler import profile, ProfilerActivity

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
dynamo_fn = torch.compile(Normalize(), dynamic=True)
nvfuser_fn = torch.jit.freeze(torch.jit.script(Normalize()).cuda().eval())

# fait
fait_fn = functs.jit.script(Normalize())
tensor_type = torch.TensorType.get().with_device(torch.device("cuda")).with_sizes([200, 1333, 3]).with_dtype(torch.float32)
functs._C._jit_pass_fait_pipeline(fait_fn.graph, [tensor_type, torch.FloatType.get(), torch.FloatType.get()])
fait_code = torch._C._jit_get_code(fait_fn.graph)
print(jit_fn.graph.str(False))
print(functs_fn.graph.str(False))
print(fait_fn.graph.str(False))

a = torch.randn(200, 1333, 3).float().cuda()
mean = 0.0
scale = 1.0

# import torch._C._te as te
# tensorexpr_str = """
# graph(%src.1 : Float(200, 1333, 3, strides=[3999, 3, 1], device=cuda:0),
#       %mean.1 : float,
#       %scale.1 : float):
#   %9 : int = prim::Constant[value=0]()
#   %8 : int = prim::Constant[value=2]()
#   %7 : NoneType = prim::Constant()
#   %6 : int = prim::Constant[value=-1]()
#   %4 : int = prim::Constant[value=1]()
#   %dup.1 : Float(200, 1333, 3, strides=[3999, 3, 1], device=cuda:0) = aten::clone(%src.1, %7)
#   %11 : Float(200, 1333, strides=[1333, 1], device=cuda:0) = immut::select(%src.1, %6, %8)
#   %24 : Float(200, 1333, 3, strides=[3999, 3, 1], device=cuda:0) = immut::select_rev(%dup.1, %11, %6, %9)
#   %14 : Float(200, 1333, strides=[1333, 1], device=cuda:0) = immut::select(%src.1, %6, %9)
#   %28 : Float(200, 1333, 3, strides=[3999, 3, 1], device=cuda:0) = immut::select_rev(%24, %14, %6, %8)
#   %17 : Float(200, 1333, 3, strides=[3999, 3, 1], device=cuda:0) = aten::sub(%28, %mean.1, %4)
#   %18 : Float(200, 1333, 3, strides=[3999, 3, 1], device=cuda:0) = aten::mul(%17, %scale.1)
#   return (%18)
# """

# torch.cuda.init()
# te_g = torch.parse_ir(tensorexpr_str)
# kernel = te.TensorExprKernel(te_g)
# print(kernel.get_code_text())

o_functs = functs_fn(a, mean, scale)
o_jit = jit_fn(a, mean, scale)
o_eager = eager_fn(a, mean, scale)

stack = torch._C._jit_run_code(fait_code, ("", a, mean, scale))

print(torch.allclose(o_functs, o_eager, atol=1e-3))
# print(torch.allclose(stack, o_eager, atol=1e-3))

functs.utils.evaluate_func(eager_fn, (a, mean, scale), "normalize eager", run_duration=2.0)
functs.utils.evaluate_func(jit_fn, (a, mean, scale), "normalize jit", run_duration=2.0)
functs.utils.evaluate_func(functs_fn, (a, mean, scale), "normalize functs", run_duration=2.0)
functs.utils.evaluate_func(dynamo_fn, (a, mean, scale), "normalize dynamo", run_duration=2.0)

torch._C._jit_set_nvfuser_enabled(True)
functs.utils.evaluate_func(nvfuser_fn, (a, mean, scale), "normalize nvfuser", run_duration=2.0)
torch._C._jit_set_nvfuser_enabled(False)

# print(nvfuser_fn.graph_for((a, mean, scale)))


# functs.utils.eval_metrics_func(eager_fn, (a, mean, scale), "normalize eager", run_duration=2.0)
# functs.utils.eval_metrics_func(jit_fn, (a, mean, scale), "normalize jit", run_duration=2.0)
# functs.utils.eval_metrics_func(functs_fn, (a, mean, scale), "normalize functs", run_duration=2.0)

# print(functs.utils.proifler_func(eager_fn, (a, mean, scale), "normalize eager", run_duration=2.0).key_metrics)
# print(functs.utils.proifler_func(jit_fn, (a, mean, scale), "normalize jit", run_duration=2.0).key_metrics)
# print(functs.utils.proifler_func(functs_fn, (a, mean, scale), "normalize functs", run_duration=2.0).key_metrics)

torch.onnx.export(functs_fn, (a, mean, scale), "x.onnx")



import torch
import functs
import torch._C._te as te


class Normalize(torch.nn.Module):
    def forward(self,
                src: torch.Tensor,
                mean: float, scale: float):
        # only inner-procedure is supported bynow.
        src = src.clone()
        # RGB to BGR
        dup = src.clone()
        dup[..., 0] = src[..., 2]
        dup[..., 2] = src[..., 0]
        return (dup - mean) * scale


jit_fn = torch.jit.freeze(torch.jit.script(Normalize().eval().cuda()))
functs_fn = functs.jit.script(Normalize().eval().cuda())

functs.jit.shape_infer(jit_fn, [torch.rand(800, 1333, 3).cuda(), 0.0, 1.0])
functs.jit.shape_infer(functs_fn, [torch.rand(800, 1333, 3).cuda(), 0.0, 1.0])

# jit function
jit_g = jit_fn.graph
print(f"jit graph:\n{jit_g}")

functs._C._jit_pass_fuse_tensorexpr(jit_g)
print(f"torch.jit.script fused graph:\n{jit_g}")

# functs function
functs_g = functs_fn.graph
print(f"functs graph:\n{functs_g}")

functs._C._jit_pass_fuse_tensorexpr(functs_g)
print(f"functs.jit.script fused graph:\n{functs_g}")

fusion_subgraph = list(functs_g.nodes())[0].g("Subgraph")
kernel = te.TensorExprKernel(fusion_subgraph)
print(kernel.get_code_text())

functs.utils.evaluate_func(Normalize(),
                           [torch.rand(800, 1333, 3).cuda(), 0.0, 1.0],
                           name="eager",
                           run_duration=2.)

functs.utils.evaluate_func(torch.jit.script(Normalize()),
                           [torch.rand(800, 1333, 3).cuda(), 0.0, 1.0],
                           name="jit",
                           run_duration=2.)

functs.utils.evaluate_func(functs.jit.script(Normalize()),
                           [torch.rand(800, 1333, 3).cuda(), 0.0, 1.0],
                           name="functs",
                           run_duration=2.)

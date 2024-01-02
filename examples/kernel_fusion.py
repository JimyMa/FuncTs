import torch
import functs


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


jit_fn = torch.jit.script(Normalize().eval().cuda())
functs_fn = functs.jit.script(Normalize().eval().cuda())


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

import time

import torch

import functs


class NormalizeTs(torch.nn.Module):
    def forward(self, src: torch.Tensor, mean: float, scale: float):
        # RGB to BGR
        dup = src.clone()
        dup[:, :, 0] = src[:, :, 2]
        dup[:, :, 2] = src[:, :, 0]
        return (dup - mean) * scale


class NormalizeFuncTs(torch.nn.Module):
    def forward(self, src: torch.Tensor, mean: float, scale: float):
        # RGB to BGR
        dup = src.clone()
        dup[:, :, 0] = src[:, :, 2]
        dup[:, :, 2] = src[:, :, 0]
        return (dup - mean) * scale


eager_fn = NormalizeTs()
jit_fn = torch.jit.script(NormalizeTs())
functs_fn = functs.jit.script(NormalizeFuncTs())

a = torch.randn(800, 1333, 3).float().cuda()
mean = 0.0
scale = 1.0

o_functs = functs_fn(a, mean, scale)
o_jit = jit_fn(a, mean, scale)
o_eager = eager_fn(a, mean, scale)

print(torch.allclose(o_functs, o_eager, atol=1e-3))

# warm up 0
for i in range(10):
    o_functs = functs_fn(a, mean, scale)
    o_jit = jit_fn(a, mean, scale)
    o_eager = eager_fn(a, mean, scale)

# warm up 1
for i in range(10):
    o_functs = functs_fn(a, mean, scale)
    o_jit = jit_fn(a, mean, scale)
    o_eager = eager_fn(a, mean, scale)

begin = time.time()
for i in range(1000):
    o_functs = functs_fn(a, mean, scale)
mid_0 = time.time()
for i in range(1000):
    o_jit = jit_fn(a, mean, scale)
mid_1 = time.time()
for i in range(1000):
    o_eager = eager_fn(a, mean, scale)
end = time.time()

print("functs: ", mid_0 - begin)
print("torchscript: ", mid_1 - mid_0)
print("eager: ", end - mid_1)

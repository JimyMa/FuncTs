import time

import torch

import functs


class Normalize(torch.nn.Module):
    def forward(self, 
                src: torch.Tensor,
                mean: float, scale: float):
        # RGB to BGR
        dup = src.clone()
        dup[:, :, 0] = src[:, :, 2]
        dup[:, :, 2] = src[:, :, 0]
        return (dup - mean) * scale


eager_fn = Normalize()
jit_fn = torch.jit.script(Normalize())
functs_fn = functs.jit.script(Normalize())
# fait
fait_fn = functs.jit.script(Normalize())
tensor_type = torch.TensorType.get().with_device(torch.device("cuda")).with_sizes([800, 1333, 3]).with_dtype(torch.float32)
functs._C._jit_pass_fait_pipeline(fait_fn.graph, [tensor_type, torch.FloatType.get(), torch.FloatType.get()])
fait_code = torch._C._jit_get_code(fait_fn.graph)
print(fait_fn.graph)

a = torch.randn(800, 1333, 3).float().cuda()
mean = 0.0
scale = 1.0

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
# for i in range(1000):
#     o_functs = functs_fn(a, mean, scale)
mid_0 = time.time()
# for i in range(1000):
#     torch._C._jit_run_code(fait_code, ("", a, mean, scale))
mid_1 = time.time()
for i in range(1000):
    o_jit = jit_fn(a, mean, scale)
mid_2 = time.time()
# for i in range(1000):
#     o_eager = eager_fn(a, mean, scale)
end = time.time()
torch.cuda.profiler.stop()

print("functs: ", mid_0 - begin)
print("fait: ", mid_1 - mid_0)
print("torchscript: ", mid_2 - mid_1)
print("eager: ", end - mid_2)



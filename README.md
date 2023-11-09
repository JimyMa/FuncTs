# FuncTs：TorchScript Functionalization

## Bulid from source

- PyTorch is all you need:

```python
python -c "import torch"; echo $?
>>> 0
python setup.py develop --user
```

## Use FuncTs to perform functionalization

you can directly run python file `python example/get_started.py`. You can also define a script as follow in a  `python` file and run it.

```python

import torch
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

# check equal
a: torch.Tensor = torch.randn([1024, 1024])
b: torch.Tensor = torch.randn([1024, 1024])

print(torch.allclose(jit_func(a, b), func(a, b)))

```

## Latency benchmark

![latency](./docs/imgs/latency.jpg)

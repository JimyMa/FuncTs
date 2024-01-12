# try ParallelMap
- pytorch code

```python
class HomoConv(torch.nn.Module):
    def __init__(self, parallel_level, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.parallel_level = parallel_level
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(self, inputs: List[torch.Tensor]):
        assert len(inputs) == self.parallel_level
        outs: List[torch.Tensor] = []
        for in_ in inputs:
            out = self.conv(in_)
            outs.append(out)
        return outs
```

- TorchScript code
```python
graph(%self : __torch__.___torch_mangle_0.HomoConv,
      %inputs.1 : Tensor[]):
  %26 : int[] = prim::Constant[value=[1, 1]]()
  %15 : int = prim::Constant[value=1]() # /mnt/workspace/LongTail/src/pytorch/pytorch/torch/nn/modules/conv.py:456:45
  %4 : str = prim::Constant[value="AssertionError: "]()
  %3 : NoneType = prim::Constant()
  %2 : bool = prim::Constant[value=1]() # examples/homo_conv.py:24:8
  %self.conv.bias : Float(8, strides=[1], requires_grad=0, device=cuda:0) = prim::Constant[value=0.01 * -5.6091  0.9351  3.7898  3.0649  2.2585 -3.9355  5.4143 -3.2854 [ CUDAFloatType{8} ]]()
  %self.conv.weight : Float(8, 8, 3, 3, strides=[72, 9, 3, 1], requires_grad=0, device=cuda:0) = prim::Constant[value=<Tensor>]()
  %self.parallel_level : int = prim::Constant[value=2]()
  %5 : int = aten::len(%inputs.1) # examples/homo_conv.py:22:15
  %7 : bool = aten::eq(%5, %self.parallel_level) # examples/homo_conv.py:22:15
   = prim::If(%7) # examples/homo_conv.py:22:8
    block0():
      -> ()
    block1():
       = prim::RaiseException(%4, %3) # examples/homo_conv.py:22:8
      -> ()
  %outs.1 : Tensor[] = prim::ListConstruct()
   = prim::Loop(%5, %2) # examples/homo_conv.py:24:8
    block0(%10 : int):
      %in_.1 : Tensor = aten::__getitem__(%inputs.1, %10) # examples/homo_conv.py:24:8
      %out.1 : Tensor = aten::conv2d(%in_.1, %self.conv.weight, %self.conv.bias, %26, %26, %26, %15) # /mnt/workspace/LongTail/src/pytorch/pytorch/torch/nn/modules/conv.py:456:15
      %14 : Tensor[] = aten::append(%outs.1, %out.1) # examples/homo_conv.py:26:12
      -> (%2)
  return (%outs.1)

graph(%self : __torch__.___torch_mangle_0.HomoConv,
      %inputs.1 : Tensor[]):
  %26 : int[] = prim::Constant[value=[1, 1]]()
  %15 : int = prim::Constant[value=1]() # /mnt/workspace/LongTail/src/pytorch/pytorch/torch/nn/modules/conv.py:456:45
  %self.conv.bias : Float(8, strides=[1], requires_grad=0, device=cuda:0) = prim::Constant[value=0.01 * -5.6091  0.9351  3.7898  3.0649  2.2585 -3.9355  5.4143 -3.2854 [ CUDAFloatType{8} ]]()
  %self.conv.weight : Float(8, 8, 3, 3, strides=[72, 9, 3, 1], requires_grad=0, device=cuda:0) = prim::Constant[value=<Tensor>]()
  %33 : int = prim::Constant[value=2]()
  %27 : Float(4, 8, 32, 32, strides=[8192, 1024, 32, 1], device=cuda:0)[] = prim::ParallelMap(%33, %inputs.1) # examples/homo_conv.py:24:8
    block0(%28 : int, %32 : Float(4, 8, 32, 32, device=cuda:0)):
      %out.4 : Float(4, 8, 32, 32, strides=[8192, 1024, 32, 1], device=cuda:0) = aten::conv2d(%32, %self.conv.weight, %self.conv.bias, %26, %26, %26, %15) # /mnt/workspace/LongTail/src/pytorch/pytorch/torch/nn/modules/conv.py:456:15
      -> (%out.4)
  return (%27)

graph(%self : __torch__.___torch_mangle_0.HomoConv,
      %inputs.4 : Float(4, 8, 32, 32, device=cuda:0)[]):
  %26 : int[] = prim::Constant[value=[1, 1]]()
  %15 : int = prim::Constant[value=1]() # /mnt/workspace/LongTail/src/pytorch/pytorch/torch/nn/modules/conv.py:456:45
  %self.conv.bias : Float(8, strides=[1], requires_grad=0, device=cuda:0) = prim::Constant[value=0.01 * -5.6091  0.9351  3.7898  3.0649  2.2585 -3.9355  5.4143 -3.2854 [ CUDAFloatType{8} ]]()
  %self.conv.weight : Float(8, 8, 3, 3, strides=[72, 9, 3, 1], requires_grad=0, device=cuda:0) = prim::Constant[value=<Tensor>]()
  %33 : int = prim::Constant[value=2]()
  %43 : Float(4, 8, 32, 32, strides=[8192, 1024, 32, 1], device=cuda:0)[] = functs_parallel::homo_conv(%33, %inputs.4, %self.conv.weight, %self.conv.bias, %26, %26, %26, %15)
  return (%43)
```

```
functs_homo: 88202 iters, min = 20.27us, max = 4.198ms, avg = 22.68us
pytorch_homo: 39291 iters, min = 47.49us, max = 4.707ms, avg = 50.9us
```

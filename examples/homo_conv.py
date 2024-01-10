from typing import List

import functs

import torch


class HomoConv(torch.nn.Module):
    def __init__(self, parallel_level, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.parallel_level = parallel_level
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=[1, 1],
            stride=[2, 2],
            dilation=[3, 3],
            bias=False,
        )

    def forward(self, inputs: List[torch.Tensor]):
        assert len(inputs) == self.parallel_level
        outs: List[torch.Tensor] = []
        for in_ in inputs:
            out = self.conv(in_)
            outs.append(out)
        return outs


parallel_level = 3

# conv config
channels = 8
kernel_size = [3, 3]
pads = [1, 1]

# batch size
bs = 4
feat_size = [32, 32]

with torch.no_grad():
    homo_conv = (
        HomoConv(parallel_level, channels, channels, kernel_size, pads).cuda().eval()
    )

    ins = (
        torch.rand(bs, channels, *feat_size).cuda(),
        torch.rand(bs, channels, *feat_size).cuda(),
        torch.rand(bs, channels, *feat_size).cuda(),
    )

    homo_conv_functs = functs.jit.script(
        functs.jit.freeze(torch.jit.script(homo_conv)), backend="aot"
    )
    print(homo_conv_functs.graph)

    type_hint = functs.jit.extract_type_hint([ins])
    functs._C._jit_pass_fait_gen_parallel_map(homo_conv_functs.graph, type_hint)
    print(homo_conv_functs.graph)

    functs._C._jit_pass_fait_gen_homo_conv(homo_conv_functs.graph, type_hint)
    print(homo_conv_functs.graph)

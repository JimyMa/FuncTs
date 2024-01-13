from typing import List

import functs

import torch
import torch.fx


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
        self.relu = torch.nn.ReLU()

    def forward(self, inputs: List[torch.Tensor]):
        assert len(inputs) == self.parallel_level
        outs: List[torch.Tensor] = []
        for in_ in inputs:
            out = self.conv(in_)
            outs.append(self.relu(out))
        return outs


parallel_level = 2

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

    # homo_conv_functs = functs.jit.script(
    #     functs.jit.freeze(torch.jit.script(homo_conv)), backend="aot"
    # )
    # print(homo_conv_functs.graph)
    in_0 = torch.ones([4, 8, 32, 32]).cuda()
    in_1 = torch.ones([4, 8, 32, 32]).cuda()

    out_0 = torch.ones([4, 8, 32, 32]).cuda()
    out_1 = torch.ones([4, 8, 32, 32]).cuda()

    weight_0 = torch.ones([8, 8, 3, 3]).cuda()
    weight_1 = torch.ones([8, 8, 3, 3]).cuda()

    bias_0 = torch.ones([8]).cuda()
    bias_1 = torch.ones([8]).cuda()

    # type_hint = functs.jit.extract_type_hint([[in_0, in_1]])
    # functs._C._jit_pass_fait_gen_parallel_map(homo_conv_functs.graph, type_hint)
    # print(homo_conv_functs.graph)

    # functs._C._jit_pass_fait_gen_homo_conv(homo_conv_functs.graph, type_hint)
    # print(homo_conv_functs.graph)

    # run cuda kernel
    print(homo_conv.conv.weight.shape)
    print(homo_conv.conv.bias.shape)

    def functs_homo_conv(
        ins: List[torch.Tensor],
        outs: List[torch.Tensor],
        weights: List[torch.Tensor],
        bias: List[torch.Tensor],
    ):
        functs._C.invoke_homo_conv(
            [ins[0], ins[1]],
            [out_0, out_1],
            [weight_0, weight_1],
            [bias_0, bias_1],
        )
        return out_0, out_1

    torch.cuda.synchronize()
    functs.utils.evaluate_func(
        functs_homo_conv,
        [[in_0, in_1], [out_0, out_1], [weight_0, weight_1], [bias_0, bias_1]],
        run_duration=2.0,
        name="functs_try_brt_homo_conv",
    )

    torch.cuda.synchronize()
    functs.utils.evaluate_func(
        homo_conv,
        [[in_0, in_1]],
        run_duration=2.0,
        name="pytorch_homo_conv",
    )

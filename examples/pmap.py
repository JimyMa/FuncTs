from typing import List

import functs

import torch


class MultiScaleBboxProcessForJIT(torch.nn.Module):
    def decode_bboxes(self, bboxes, pred_bboxes, stride: float):
        # assert pred_bboxes.size(-1) == bboxes.size(-1) == 4
        xy_centers = (bboxes[..., :2] + bboxes[..., 2:]) * 0.5 + (
            pred_bboxes[..., :2] - 0.5
        ) * stride
        whs = (bboxes[..., 2:] - bboxes[..., :2]) * 0.5 * pred_bboxes[..., 2:].exp()
        decoded_bboxes = torch.stack(
            (
                xy_centers[..., 0] - whs[..., 0],
                xy_centers[..., 1] - whs[..., 1],
                xy_centers[..., 0] + whs[..., 0],
                xy_centers[..., 1] + whs[..., 1],
            ),
            dim=-1,
        )
        return decoded_bboxes.clone()

    def forward(
        self,
        bboxes_list: List[torch.Tensor],
        pred_bboxes_list: List[torch.Tensor],
        stride_list: List[float],
    ):
        outs = []
        for bboxes, pred_bboxes, stride in zip(
            bboxes_list, pred_bboxes_list, stride_list
        ):
            out = self.decode_bboxes(bboxes, pred_bboxes, stride)
            outs.append(out)
        return outs


class MultiScaleBboxProcessUnroll(torch.nn.Module):
    def decode_bboxes(self, bboxes, pred_bboxes, stride: float):
        # assert pred_bboxes.size(-1) == bboxes.size(-1) == 4
        xy_centers = (bboxes[..., :2] + bboxes[..., 2:]) * 0.5 + (
            pred_bboxes[..., :2] - 0.5
        ) * stride
        whs = (bboxes[..., 2:] - bboxes[..., :2]) * 0.5 * pred_bboxes[..., 2:].exp()
        decoded_bboxes = torch.stack(
            (
                xy_centers[..., 0] - whs[..., 0],
                xy_centers[..., 1] - whs[..., 1],
                xy_centers[..., 0] + whs[..., 0],
                xy_centers[..., 1] + whs[..., 1],
            ),
            dim=-1,
        )
        return decoded_bboxes.clone()

    def forward(
        self,
        bboxes_0,
        bboxes_1,
        bboxes_2,
        pred_bboxes_0,
        pred_bboxes_1,
        pred_bboxes_2,
        stride_0: float,
        stride_1: float,
        stride_2: float,
    ):
        outs = []
        out_0 = self.decode_bboxes(bboxes_0, pred_bboxes_0, stride_0)
        outs.append(out_0)
        out_1 = self.decode_bboxes(bboxes_1, pred_bboxes_1, stride_1)
        outs.append(out_1)
        out_2 = self.decode_bboxes(bboxes_2, pred_bboxes_2, stride_2)
        outs.append(out_2)
        return outs


class MultiScaleBboxProcess(torch.nn.Module):
    def decode_bboxes(self, bboxes, pred_bboxes, stride: float):
        # assert pred_bboxes.size(-1) == bboxes.size(-1) == 4
        xy_centers = (bboxes[..., :2] + bboxes[..., 2:]) * 0.5 + (
            pred_bboxes[..., :2] - 0.5
        ) * stride
        whs = (bboxes[..., 2:] - bboxes[..., :2]) * 0.5 * pred_bboxes[..., 2:].exp()
        decoded_bboxes = torch.stack(
            (
                xy_centers[..., 0] - whs[..., 0],
                xy_centers[..., 1] - whs[..., 1],
                xy_centers[..., 0] + whs[..., 0],
                xy_centers[..., 1] + whs[..., 1],
            ),
            dim=-1,
        )
        return decoded_bboxes.clone()

    def forward(
        self,
        bboxes_list: List[torch.Tensor],
        pred_bboxes_list: List[torch.Tensor],
        stride_list: List[float],
    ):
        outs = []
        for bboxes, pred_bboxes, stride in zip(
            bboxes_list, pred_bboxes_list, stride_list
        ):
            out = self.decode_bboxes(bboxes, pred_bboxes, stride)
            outs.append(out)
        return outs


bboxes_0 = torch.rand(128, 4).cuda()
bboxes_1 = torch.rand(64, 4).cuda()
bboxes_2 = torch.rand(32, 4).cuda()

pred_bboxes_0 = torch.rand(128, 4).cuda()
pred_bboxes_1 = torch.rand(64, 4).cuda()
pred_bboxes_2 = torch.rand(32, 4).cuda()

bboxes_list = [bboxes_0, bboxes_1, bboxes_2]
pred_bboxes_list = [pred_bboxes_0, pred_bboxes_1, pred_bboxes_2]
stride_list = [8.0, 16.0, 32.0]

multi_scale_bboxes_process = MultiScaleBboxProcess().cuda().eval()
multi_scale_bboxes_process_jit = MultiScaleBboxProcessForJIT().cuda().eval()
multi_scale_bboxes_process_unroll = MultiScaleBboxProcessUnroll().cuda().eval()

jit_module = torch.jit.script(multi_scale_bboxes_process_jit)

multi_scale_bboxes_process_functs_pmap = functs.jit.script(
    multi_scale_bboxes_process, backend="aot"
)

functs_pmap_module = functs.jit.build(
    multi_scale_bboxes_process_functs_pmap, [bboxes_list, pred_bboxes_list, stride_list]
)

multi_scale_bboxes_process_functs_unroll = functs.jit.script(
    multi_scale_bboxes_process_unroll,
)

functs_unroll_module = functs.jit.build(
    multi_scale_bboxes_process_functs_unroll,
    [*bboxes_list, *pred_bboxes_list, *stride_list],
)

print(jit_module.graph)
print(functs_pmap_module.graph)
print(functs_unroll_module.graph)

functs.utils.evaluate_func(
    jit_module,
    [bboxes_list, pred_bboxes_list, stride_list],
    run_duration=2.0,
    name="jit",
)

functs.utils.evaluate_func(
    functs_unroll_module,
    [*bboxes_list, *pred_bboxes_list, *stride_list],
    run_duration=2.0,
    name="functs unroll",
)

functs.utils.evaluate_func(
    functs_pmap_module,
    [bboxes_list, pred_bboxes_list, stride_list],
    run_duration=2.0,
    name="functs pmap",
)

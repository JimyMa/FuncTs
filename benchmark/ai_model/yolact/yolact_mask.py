from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor


def select_single_mlvl(mlvl_tensors: List[Tensor], batch_id: int):
    num_levels = len(mlvl_tensors)
    mlvl_tensor_list = [
        mlvl_tensors[i][batch_id] for i in range(num_levels)
    ]
    return mlvl_tensor_list


def delta2bbox(rois: torch.Tensor,
               deltas: torch.Tensor,
               ):
    means = (0., 0., 0., 0.)
    stds = (0.1, 0.1, 0.2, 0.2)
    wh_ratio_clip = 0.016
    max_shape = (320, 320)

    num_bboxes, num_classes = deltas.size(0), deltas.size(1) // 4
    deltas = deltas.reshape(-1, 4)
    means = torch.tensor(means, dtype=deltas.dtype,
                         device=deltas.device).view(1, -1)
    stds = torch.tensor(stds, dtype=deltas.dtype,
                        device=deltas.device).view(1, -1)
    denorm_deltas = deltas * stds + means
    dxy = denorm_deltas[:, :2]
    dwh = denorm_deltas[:, 2:]

    rois_ = rois.repeat(1, num_classes).reshape(-1, 4)
    pxy = ((rois_[:, :2] + rois_[:, 2:]) * 0.5)
    pwh = (rois_[:, 2:] - rois_[:, :2])
    dxy_wh = pwh * dxy
    max_ratio = float(torch.abs(torch.log(torch.tensor(wh_ratio_clip))))
    dwh = dwh.clamp(min=-max_ratio, max=max_ratio)

    gxy = pxy + dxy_wh
    gwh = pwh * dwh.exp()
    x1y1 = gxy - (gwh * 0.5)
    x2y2 = gxy + (gwh * 0.5)
    bboxes = torch.cat([x1y1, x2y2], dim=-1)
    # bboxes[..., 0::2].clamp_(min=0, max=max_shape[1])
    # bboxes[..., 1::2].clamp_(min=0, max=max_shape[0])

    torch.clamp_(bboxes, min=0, max=max_shape[1])

    return bboxes


def fast_nms(multi_bboxes: Tensor,
             multi_scores: Tensor,
             multi_coeffs: Tensor):
    score_thr = 0.05
    iou_thr = 0.5
    top_k = 200
    max_num = 100

    scores = multi_scores[:, :-1].t()  # [#class, n]
    scores, idx = scores.sort(1, descending=True)
    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]  # [#class, topk]
    num_classes, num_dets = idx.size(0), idx.size(1)
    boxes = multi_bboxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    coeffs = multi_coeffs[idx.view(-1), :].view(num_classes, num_dets, 32)
    iou = bbox_overlaps(boxes, boxes)  # [#class, topk, topk]
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    keep = (iou_max <= iou_thr) & (scores > score_thr)
    classes = torch.arange(
        num_classes, device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]
    boxes = boxes[keep]
    coeffs = coeffs[keep]
    scores = scores[keep]

    scores, idx = scores.sort(0, descending=True)
    idx = idx[:max_num]
    scores = scores[:max_num]
    classes = classes[idx]
    boxes = boxes[idx]
    coeffs = coeffs[idx]

    cls_dets = torch.cat([boxes, scores[:, None]], dim=1)
    return cls_dets, classes, coeffs


def bbox_overlaps(bboxes1: Tensor, bboxes2: Tensor):
    # Either the boxes are empty or the length of boxes' last dimension is 4
    # assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    # assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)
    # assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])
    lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
    rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
    wh = torch.clamp(rb - lt, min=0)
    overlap = wh[..., 0] * wh[..., 1]
    union = area1[..., None] + area2[..., None, :] - overlap
    eps = torch.tensor([1e-6], dtype=union.dtype, device=union.device)
    union = torch.max(union, eps)
    ious = overlap / union
    return ious


class AnchorGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_sizes = [8, 16, 32, 64, 128]
        self.strides = [(s, s) for s in self.base_sizes]
        self.octave_base_scale = 3
        self.scales_per_octave = 1
        octave_scales = np.array(
            [2**(i / self.scales_per_octave) for i in range(self.scales_per_octave)])
        scales = octave_scales * self.octave_base_scale
        self.scales = torch.Tensor(scales)
        self.ratios = torch.Tensor([0.5, 1.0, 2.0])
        self.scale_major = True
        self.centers = [(s / 2, s / 2) for s in self.base_sizes]
        self.center_offset = 0
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_levels(self):
        return len(self.strides)

    def gen_base_anchors(self):
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=self.scales,
                    ratios=self.ratios,
                    center=center))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      ratios,
                                      center=None):
        w = base_size
        h = base_size
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors.cuda()

    def forward(self, featmap_sizes: List[Tuple[int, int]], dtype: torch.dtype, device: torch.device):
        # assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_priors(
                featmap_sizes[i], level_idx=i, dtype=dtype, device=device)
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_priors(self,
                                 featmap_size: Tuple[int, int],
                                 level_idx: int,
                                 dtype: torch.dtype,
                                 device: torch.device):

        base_anchors = self.base_anchors[level_idx].to(
            device=device, dtype=dtype)
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = torch.arange(0, feat_w, dtype=dtype,
                               device=device) * stride_w
        shift_y = torch.arange(0, feat_h, dtype=dtype,
                               device=device) * stride_h
        
        shifts = torch.zeros(feat_h * feat_w, 4, device="cuda", dtype=torch.float32)
        shift_xx = shift_x.repeat(feat_h)
        shift_yy = shift_y.reshape(-1, 1).repeat(1, feat_w).reshape(-1)
        # shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts[:, 0].copy_(shift_xx)
        shifts[:, 1].copy_(shift_yy)
        shifts[:, 2].copy_(shift_xx)
        shifts[:, 3].copy_(shift_yy)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        return all_anchors

    def _meshgrid(self, x: Tensor, y: Tensor):
        xx = x.repeat(y.size(0))
        yy = y.view(-1, 1).repeat(1, x.size(0)).view(-1)
        return xx, yy


class YolactBBoxMask(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_sizes = [8, 16, 32, 64, 128]
        self.strides = [(s, s) for s in self.base_sizes]
        self.octave_base_scale = 3
        self.scales_per_octave = 1
        octave_scales = np.array(
            [2**(i / self.scales_per_octave) for i in range(self.scales_per_octave)])
        scales = octave_scales * self.octave_base_scale
        self.scales = torch.Tensor(scales)
        self.ratios = torch.Tensor([0.5, 1.0, 2.0])
        self.scale_major = True
        self.centers = [(s / 2, s / 2) for s in self.base_sizes]
        self.center_offset = 0
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_levels(self):
        return len(self.strides)

    def gen_base_anchors(self):
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=self.scales,
                    ratios=self.ratios,
                    center=center))
        return multi_level_base_anchors
    
    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      ratios,
                                      center=None):
        w = base_size
        h = base_size
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors.cuda()

    def forward(self,
                bbox_preds: List[Tensor],
                cls_scores: List[Tensor],
                coeff_preds: List[Tensor],
                proto: Tensor):
        # assert len(cls_scores) == len(bbox_preds)
        # bbox_preds: [torch.Size([1, 12, 40, 40]), torch.Size([1, 12, 20, 20]), torch.Size([1, 12, 10, 10]), torch.Size([1, 12, 5, 5]), torch.Size([1, 12, 3, 3])]
        # cls_scores: [torch.Size([1, 243, 40, 40]), torch.Size([1, 243, 20, 20]), torch.Size([1, 243, 10, 10]), torch.Size([1, 243, 5, 5]), torch.Size([1, 243, 3, 3])]
        # coeff_preds: [torch.Size([1, 96, 40, 40]), torch.Size([1, 96, 20, 20]), torch.Size([1, 96, 10, 10]), torch.Size([1, 96, 5, 5]), torch.Size([1, 96, 3, 3])]
        # proto: torch.Size([1, 32, 80, 80])
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        dtype = cls_scores[0].dtype
        featmap_sizes = [(cls_score.size(-2), cls_score.size(-1))
                         for cls_score in cls_scores]

        mlvl_anchors = []
        # for level_idx in range(5):
        #     base_anchors = self.base_anchors[level_idx].to(
        #     device=device, dtype=dtype)
        #     feat_h, feat_w = featmap_sizes[level_idx]
        #     stride_w, stride_h = self.strides[level_idx]
        #     shift_x = torch.arange(0, feat_w, dtype=dtype,
        #                         device=device) * stride_w
        #     shift_y = torch.arange(0, feat_h, dtype=dtype,
        #                         device=device) * stride_h
            
        #     shifts = torch.zeros(feat_h * feat_w, 4, device="cuda", dtype=torch.float32)
        #     shift_xx = shift_x.repeat(feat_h)
        #     shift_yy = shift_y.reshape(-1, 1).repeat(1, feat_w).reshape(-1)
        #     # shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        #     shifts[:, 0].copy_(shift_xx)
        #     shifts[:, 1].copy_(shift_yy)
        #     shifts[:, 2].copy_(shift_xx)
        #     shifts[:, 3].copy_(shift_yy)

        #     anchors = base_anchors[None, :, :] + shifts[:, None, :]
        #     anchors = anchors.view(-1, 4)
        #     mlvl_anchors.append(anchors)
        
        base_anchors = self.base_anchors[0].to(device=device, dtype=dtype).clone()
        feat_h, feat_w = featmap_sizes[0]
        stride_w, stride_h = self.strides[0]
        shift_x = torch.arange(0, 40, dtype=dtype,
                            device=device) * stride_w
        shift_y = torch.arange(0, 40, dtype=dtype,
                            device=device) * stride_h
        
        shifts = torch.zeros(40 * 40, 4, device="cuda", dtype=torch.float32)
        shift_xx = shift_x.repeat(40)
        shift_yy = shift_y.reshape(-1, 1).repeat(1, 40).reshape(-1)
        # shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts[:, 0].copy_(shift_xx)
        shifts[:, 1].copy_(shift_yy)
        shifts[:, 2].copy_(shift_xx)
        shifts[:, 3].copy_(shift_yy)

        anchors = base_anchors[None, :, :] + shifts[:, None, :]
        anchors = anchors.view(-1, 4).clone()
        mlvl_anchors.append(anchors)

        base_anchors = self.base_anchors[1].to(device=device, dtype=dtype).clone()
        feat_h, feat_w = featmap_sizes[1]
        stride_w, stride_h = self.strides[1]
        shift_x = torch.arange(0, 20, dtype=dtype,
                            device=device) * stride_w
        shift_y = torch.arange(0, 20, dtype=dtype,
                            device=device) * stride_h
        
        shifts = torch.zeros(20 * 20, 4, device="cuda", dtype=torch.float32)
        shift_xx = shift_x.repeat(20)
        shift_yy = shift_y.reshape(-1, 1).repeat(1, 20).reshape(-1)
        # shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts[:, 0].copy_(shift_xx)
        shifts[:, 1].copy_(shift_yy)
        shifts[:, 2].copy_(shift_xx)
        shifts[:, 3].copy_(shift_yy)

        anchors = base_anchors[None, :, :] + shifts[:, None, :]
        anchors = anchors.view(-1, 4).clone()
        mlvl_anchors.append(anchors)


        base_anchors = self.base_anchors[2].to(device=device, dtype=dtype).clone()
        feat_h, feat_w = featmap_sizes[2]
        stride_w, stride_h = self.strides[2]
        shift_x = torch.arange(0, 10, dtype=dtype,
                            device=device) * stride_w
        shift_y = torch.arange(0, 10, dtype=dtype,
                            device=device) * stride_h
        
        shifts = torch.zeros(10 * 10, 4, device="cuda", dtype=torch.float32)
        shift_xx = shift_x.repeat(10)
        shift_yy = shift_y.reshape(-1, 1).repeat(1, 10).reshape(-1)
        # shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts[:, 0].copy_(shift_xx)
        shifts[:, 1].copy_(shift_yy)
        shifts[:, 2].copy_(shift_xx)
        shifts[:, 3].copy_(shift_yy)

        anchors = base_anchors[None, :, :] + shifts[:, None, :]
        anchors = anchors.view(-1, 4).clone()
        mlvl_anchors.append(anchors)


        base_anchors = self.base_anchors[3].to(device=device, dtype=dtype).clone()
        feat_h, feat_w = featmap_sizes[3]
        stride_w, stride_h = self.strides[3]
        shift_x = torch.arange(0, 5, dtype=dtype,
                            device=device) * stride_w
        shift_y = torch.arange(0, 5, dtype=dtype,
                            device=device) * stride_h
        
        shifts = torch.zeros(5 * 5, 4, device="cuda", dtype=torch.float32)
        shift_xx = shift_x.repeat(5)
        shift_yy = shift_y.reshape(-1, 1).repeat(1, 5).reshape(-1)
        # shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts[:, 0].copy_(shift_xx)
        shifts[:, 1].copy_(shift_yy)
        shifts[:, 2].copy_(shift_xx)
        shifts[:, 3].copy_(shift_yy)

        anchors = base_anchors[None, :, :] + shifts[:, None, :]
        anchors = anchors.view(-1, 4).clone()
        mlvl_anchors.append(anchors)


        base_anchors = self.base_anchors[4].to(device=device, dtype=dtype).clone()
        feat_h, feat_w = featmap_sizes[4]
        stride_w, stride_h = self.strides[4]
        shift_x = torch.arange(0, 3, dtype=dtype,
                            device=device) * stride_w
        shift_y = torch.arange(0, 3, dtype=dtype,
                            device=device) * stride_h
        
        shifts = torch.zeros(3 * 3, 4, device="cuda", dtype=torch.float32)
        shift_xx = shift_x.repeat(3)
        shift_yy = shift_y.reshape(-1, 1).repeat(1, 3).reshape(-1)
        # shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts[:, 0].copy_(shift_xx)
        shifts[:, 1].copy_(shift_yy)
        shifts[:, 2].copy_(shift_xx)
        shifts[:, 3].copy_(shift_yy)

        anchors = base_anchors[None, :, :] + shifts[:, None, :]
        anchors = anchors.view(-1, 4).clone()
        mlvl_anchors.append(anchors)


        det_bboxes = []
        det_labels = []
        det_coeffs = []
        num_imgs = cls_scores[0].size(0)
        for img_id in range(num_imgs):
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            coeff_pred_list = select_single_mlvl(coeff_preds, img_id)
            bbox_res = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                               coeff_pred_list, mlvl_anchors)
            det_bboxes.append(bbox_res[0])
            det_labels.append(bbox_res[1])
            det_coeffs.append(bbox_res[2])

        proto = proto.permute(0, 2, 3, 1).contiguous()
        mask_pred_list = []
        for idx in range(num_imgs):
            cur_prototypes = proto[idx]
            cur_coeff_pred = det_coeffs[idx]
            cur_bboxes = det_bboxes[idx]
            bboxes_for_cropping = cur_bboxes
            mask_pred = cur_prototypes @ cur_coeff_pred.t()
            mask_pred = torch.sigmoid(mask_pred)

            mask_pred = self.crop(mask_pred, bboxes_for_cropping)
            mask_pred = mask_pred.permute(2, 0, 1).contiguous()
            mask_pred_list.append(mask_pred)

        return det_bboxes, det_labels, det_coeffs, mask_pred_list

    def _get_bboxes_single(self,
                           cls_score_list: List[Tensor],
                           bbox_pred_list: List[Tensor],
                           coeff_preds_list: List[Tensor],
                           mlvl_anchors: List[Tensor]):
        # assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        nms_pre = 1000
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_coeffs = []
        # for cls_score, bbox_pred, coeff_pred, anchors in \
        #         zip(cls_score_list, bbox_pred_list,
        #             coeff_preds_list, mlvl_anchors):
        #     # assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        cls_score = cls_score_list[0].clone()
        bbox_pred = bbox_pred_list[0].clone()
        coeff_pred = coeff_preds_list[0].clone()
        anchors = mlvl_anchors[0].clone()
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, 81)
        scores = cls_score.softmax(-1)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
        coeff_pred = coeff_pred.permute(1, 2, 0).reshape(-1, 32)
        max_scores, _ = scores[:, :-1].max(dim=1)

        _, topk_inds = max_scores.topk(min(nms_pre, max_scores.size(0)))
        anchors = anchors[topk_inds, :]
        bbox_pred = bbox_pred[topk_inds, :]
        scores = scores[topk_inds, :]
        coeff_pred = coeff_pred[topk_inds, :]

        bboxes = delta2bbox(anchors, bbox_pred)

        mlvl_bboxes.append(bboxes.clone())
        mlvl_scores.append(scores.clone())
        mlvl_coeffs.append(coeff_pred.clone())

        cls_score = cls_score_list[1].clone()
        bbox_pred = bbox_pred_list[1].clone()
        coeff_pred = coeff_preds_list[1].clone()
        anchors = mlvl_anchors[1].clone()
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, 81)
        scores = cls_score.softmax(-1)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
        coeff_pred = coeff_pred.permute(1, 2, 0).reshape(-1, 32)
        max_scores, _ = scores[:, :-1].max(dim=1)

        _, topk_inds = max_scores.topk(min(nms_pre, max_scores.size(0)))
        anchors = anchors[topk_inds, :]
        bbox_pred = bbox_pred[topk_inds, :]
        scores = scores[topk_inds, :]
        coeff_pred = coeff_pred[topk_inds, :]

        bboxes = delta2bbox(anchors, bbox_pred)

        mlvl_bboxes.append(bboxes.clone())
        mlvl_scores.append(scores.clone())
        mlvl_coeffs.append(coeff_pred.clone())

        cls_score = cls_score_list[2].clone()
        bbox_pred = bbox_pred_list[2].clone()
        coeff_pred = coeff_preds_list[2].clone()
        anchors = mlvl_anchors[2].clone()
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, 81)
        scores = cls_score.softmax(-1)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
        coeff_pred = coeff_pred.permute(1, 2, 0).reshape(-1, 32)
        max_scores, _ = scores[:, :-1].max(dim=1)

        _, topk_inds = max_scores.topk(min(nms_pre, max_scores.size(0)))
        anchors = anchors[topk_inds, :]
        bbox_pred = bbox_pred[topk_inds, :]
        scores = scores[topk_inds, :]
        coeff_pred = coeff_pred[topk_inds, :]

        bboxes = delta2bbox(anchors, bbox_pred)

        mlvl_bboxes.append(bboxes.clone())
        mlvl_scores.append(scores.clone())
        mlvl_coeffs.append(coeff_pred.clone())

        cls_score = cls_score_list[3].clone()
        bbox_pred = bbox_pred_list[3].clone()
        coeff_pred = coeff_preds_list[3].clone()
        anchors = mlvl_anchors[3].clone()
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, 81)
        scores = cls_score.softmax(-1)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
        coeff_pred = coeff_pred.permute(1, 2, 0).reshape(-1, 32)
        max_scores, _ = scores[:, :-1].max(dim=1)

        _, topk_inds = max_scores.topk(min(nms_pre, max_scores.size(0)))
        anchors = anchors[topk_inds, :]
        bbox_pred = bbox_pred[topk_inds, :]
        scores = scores[topk_inds, :]
        coeff_pred = coeff_pred[topk_inds, :]

        bboxes = delta2bbox(anchors, bbox_pred)

        mlvl_bboxes.append(bboxes.clone())
        mlvl_scores.append(scores.clone())
        mlvl_coeffs.append(coeff_pred.clone())

        cls_score = cls_score_list[4].clone()
        bbox_pred = bbox_pred_list[4].clone()
        coeff_pred = coeff_preds_list[4].clone()
        anchors = mlvl_anchors[4].clone()
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, 81)
        scores = cls_score.softmax(-1)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
        coeff_pred = coeff_pred.permute(1, 2, 0).reshape(-1, 32)
        max_scores, _ = scores[:, :-1].max(dim=1)

        _, topk_inds = max_scores.topk(min(nms_pre, max_scores.size(0)))
        anchors = anchors[topk_inds, :]
        bbox_pred = bbox_pred[topk_inds, :]
        scores = scores[topk_inds, :]
        coeff_pred = coeff_pred[topk_inds, :]

        bboxes = delta2bbox(anchors, bbox_pred)

        mlvl_bboxes.append(bboxes.clone())
        mlvl_scores.append(scores.clone())
        mlvl_coeffs.append(coeff_pred.clone())

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_coeffs = torch.cat(mlvl_coeffs)
        det_bboxes, det_labels, det_coeffs = fast_nms(mlvl_bboxes, mlvl_scores,
                                                      mlvl_coeffs)

        return det_bboxes, det_labels, det_coeffs

    def crop(self, masks: Tensor, boxes: Tensor):
        h, w, n = masks.size(0), masks.size(1), masks.size(2)
        x1, x2 = self.sanitize_coordinates(boxes[:, 0], boxes[:, 2], w)
        y1, y2 = self.sanitize_coordinates(boxes[:, 1], boxes[:, 3], h)

        rows = torch.arange(
            w, device=masks.device, dtype=x1.dtype).view(1, -1,
                                                         1).expand(h, w, n)
        cols = torch.arange(
            h, device=masks.device, dtype=x1.dtype).view(-1, 1,
                                                         1).expand(h, w, n)

        masks_left = rows >= x1.view(1, 1, -1)
        masks_right = rows < x2.view(1, 1, -1)
        masks_up = cols >= y1.view(1, 1, -1)
        masks_down = cols < y2.view(1, 1, -1)

        crop_mask = masks_left & masks_right & masks_up & masks_down

        return masks * crop_mask.float()

    def sanitize_coordinates(self, x1, x2, img_size: int):
        padding = 1
        x1 = torch.min(x1, x2)
        x2 = torch.max(x1, x2)
        x1 = torch.clamp(x1 - padding, min=0)
        x2 = torch.clamp(x2 + padding, max=img_size)
        return x1, x2


if __name__ == '__main__':
    mask = YolactBBoxMask().cuda().eval()
    mask = torch.jit.script(mask)
    print(mask.graph)
    torch.jit.save(mask, 'yolact_mask.pt')
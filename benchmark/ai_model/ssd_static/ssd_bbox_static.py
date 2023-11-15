from typing import List, Tuple

import numpy as np

import torch
from torch import Tensor


class SSDAnchorGenerator(torch.nn.Module):
    def __init__(self, stride:int, ratio:Tuple[int, int], min_size: int, max_size: int, scale_major: bool):
        super().__init__()
        from torch.nn.modules.utils import _pair
        self.stride: Tuple[int, int] = _pair(stride)
        self.center: Tuple[float, float] = (stride / 2., stride / 2.)
        # self.base_size = base_size
        # self.base_anchor = self.gen_base_anchors()

        scales = [1., np.sqrt(max_size / min_size)]
        anchor_ratio = [1.]
        for r in ratio:
            anchor_ratio += [1 / r, r]  # 4 or 6 ratio
        anchor_ratio = Tensor(anchor_ratio).cuda()
        anchor_scale = Tensor(scales).cuda()

        self.base_size = min_size
        self.scale = anchor_scale
        self.ratio = anchor_ratio
        self.scale_major = scale_major
        self.center_offset = 0
        self.base_anchors = self.gen_base_anchors()
    
    def gen_base_anchors(self):
        base_anchors = self.gen_single_level_base_anchors(
            self.base_size,
            scale=self.scale,
            ratio=self.ratio,
            center=self.center)
        indices = list(range(len(self.ratio)))
        indices.insert(1, len(indices))
        base_anchors = torch.index_select(base_anchors, 0,
                                              torch.LongTensor(indices).cuda())
        return base_anchors
    
    def gen_single_level_base_anchors(self, base_size: Tuple[int, int], scale, ratio, center: Tuple[float, float]):
        w = base_size
        h = base_size
        x_center, y_center = center

        h_ratios = torch.sqrt(ratio)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scale[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scale[None, :]).view(-1)
        else:
            ws = (w * scale[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scale[:, None] * h_ratios[None, :]).view(-1)

        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = torch.stack(base_anchors, dim=-1)
        

        return base_anchors
    
    def forward(self, featmap_size: Tuple[int, int], dtype: torch.dtype, device: torch.device):
        # assert self.num_levels == len(featmap_sizes)
        anchors = self.single_level_grid_priors(featmap_size, dtype=dtype, device=device)
        return anchors
    
    def single_level_grid_priors(self,
                                 featmap_size: Tuple[int, int],
                                 dtype: torch.dtype,
                                 device: torch.device):
        base_anchor = self.base_anchors.to(device, dtype)
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.stride
        shift_x = torch.arange(0, feat_w, dtype=dtype,
                               device=device) * stride_w
        shift_y = torch.arange(0, feat_h, dtype=dtype,
                               device=device) * stride_h

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.zeros([feat_w * feat_h, 4], device="cuda", dtype=torch.float32)
        shifts[:, 0].copy_(shift_xx) 
        shifts[:, 1].copy_(shift_yy)
        shifts[:, 2].copy_(shift_xx)
        shifts[:, 3].copy_(shift_yy)
        # shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        # print(shifts.shape)
        all_anchors = base_anchor[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.reshape(-1, 4)

        return all_anchors
    
    def _meshgrid(self, x: Tensor, y: Tensor):
        xx = x.repeat(y.size(0))
        yy = y.reshape(-1, 1).repeat(1, x.size(0)).reshape(-1)
        return xx, yy


def select_single_mlvl(mlvl_tensors: List[Tensor], batch_id: int):
    num_levels = len(mlvl_tensors)
    mlvl_tensor_list = [mlvl_tensors[i][batch_id] for i in range(num_levels)]
    return mlvl_tensor_list


def filter_scores_and_topk(scores: Tensor, score_thr: float, topk: int):
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)
    num_topk = min(topk, valid_idxs.size(0))
    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs[:, 0], topk_idxs[:, 1]
    return scores, labels, keep_idxs


def delta2bbox(rois: Tensor,
               deltas: Tensor
               ):
    means = (0., 0., 0., 0.)
    stds = (0.1, 0.1, 0.2, 0.2)
    wh_ratio_clip = 0.016
    max_shape = (320, 320)

    num_classes = deltas.size(1) // 4
    deltas = deltas.reshape(-1, 4)
    means = torch.tensor(means, dtype=deltas.dtype,
                         device=deltas.device)
    stds = torch.tensor(stds, dtype=deltas.dtype,
                        device=deltas.device)
    denorm_deltas = deltas * stds + means
    dxy = denorm_deltas[:, :2]
    dwh = denorm_deltas[:, 2:]

    rois_ = torch.zeros([rois.shape[0], rois.shape[1] * num_classes], dtype=torch.float32, device="cuda")
    rois_[:, 0:4].copy_(rois_)

    # rois_ = rois.repeat(1, num_classes).reshape(-1, 4)
    pxy = ((rois_[:, :2] + rois_[:, 2:]) * 0.5)
    pwh = (rois_[:, 2:] - rois_[:, :2])
    dxy_wh = pwh * dxy
    max_ratio = float(torch.abs(torch.log(torch.tensor(wh_ratio_clip))))
    dwh = dwh.clamp(min=-max_ratio, max=max_ratio)

    bboxes = torch.zeros_like(rois)
    gxy = pxy + dxy_wh
    gwh = pwh * dwh.exp()
    x1y1 = gxy - (gwh * 0.5)
    x2y2 = gxy + (gwh * 0.5)
    bboxes[..., 0:2].copy_(x1y1)
    bboxes[..., 0:2].copy_(x2y2)
    
    bboxes.clamp_(min=0, max=max_shape[1])

    return bboxes + 0.0


def nms_wrapper(boxes: Tensor,
                scores: Tensor,
                iou_threshold: float):
    # assert boxes.size(1) == 4
    # assert boxes.size(0) == scores.size(0)
    inds = torch.ops.torchvision.nms(boxes, scores, iou_threshold)
    dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)
    return dets, inds


def batched_nms(boxes: Tensor,
                scores: Tensor,
                idxs: Tensor):
    iou_threshold = 0.45
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]

    split_thr = 10000
    if boxes_for_nms.size(0) < split_thr:
        dets, keep = nms_wrapper(
            boxes_for_nms, scores, iou_threshold=iou_threshold)
        boxes = boxes[keep]
        scores = dets[:, -1]
    else:
        total_mask = torch.zeros_like(scores, dtype=torch.bool)
        scores_after_nms = torch.zeros_like(scores)
        for id in torch.unique(idxs):
            mask = torch.nonzero(idxs == id).view(-1)
            dets, keep = nms_wrapper(
                boxes_for_nms[mask], scores[mask], iou_threshold)
            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = dets[:, -1]
        keep = total_mask.nonzero().view(-1)
        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

    boxes = torch.cat([boxes, scores[:, None]], -1)
    det_bboxes = boxes[:100]
    det_labels = idxs[keep][:100]
    return det_bboxes, det_labels



class SSDBBox(torch.nn.Module):
    def __init__(self, 
                 stride: int,
                 ratio: List[int],
                 min_size: float,
                 max_size: float,
                 scale_major: bool) -> None:
        super().__init__()
        self.num_classes = 80
        self.num_attrib = 5 + self.num_classes
        self.cls_out_channels = self.num_classes + 1
        self.use_sigmoid_cls = False
        self.stride = stride
        self.prior_generator = SSDAnchorGenerator(
            stride=stride,
            ratio=ratio,
            min_size=min_size,
            max_size=max_size,
            scale_major=scale_major
        )
        # self.decode_bboxes = DecodeBBoxes()

    def forward(self, cls_score: Tensor, bbox_pred: Tensor):
        # pred_maps = [torch.Size([1, 255, 10, 10]), torch.Size([1, 255, 20, 20]), torch.Size([1, 255, 40, 40])]
        featmap_sizes = (cls_score.size(-2), cls_score.size(-1))
        
        anchors = self.prior_generator(
            featmap_sizes,
            dtype=cls_score.dtype,
            device=cls_score.device)

        result_list: List[Tuple[Tensor, Tensor]] = []
        results = self._get_bboxes_single(cls_score, bbox_pred, anchors)
        result_list.append(results)

        return result_list

    def _get_bboxes_single(self,
                           cls_score: Tensor,
                           bbox_pred: Tensor,
                           anchors: Tensor):
        nms_pre = 1000
        score_thr = 0.02
        cls_score = cls_score[0]
        bbox_pred = bbox_pred[0]
        priors = anchors

        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
        
        cls_score = cls_score.permute(
            1, 2, 0).reshape(-1, self.cls_out_channels)
        scores = cls_score.softmax(-1)[:, :-1]
        
        scores, labels, keep_idxs = filter_scores_and_topk(
            scores, score_thr, nms_pre)

        bbox_pred = bbox_pred
        priors = priors
        bboxes = delta2bbox(priors, bbox_pred)
        bboxes = bboxes[keep_idxs]

        max_per_img = 100
        if bboxes.numel() > 0:
            det_bboxes, det_labels = batched_nms(bboxes, scores, labels)
        else:
            det_bboxes, det_labels = torch.tensor(0.).cuda(), torch.tensor(0.0).cuda()

        return det_bboxes, det_labels
        




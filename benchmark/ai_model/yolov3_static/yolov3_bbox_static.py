from typing import List, Tuple

import torch
from torch import Tensor


class DecodeBBoxes(torch.nn.Module):
    def forward(self,bboxes, pred_bboxes, stride):
        # assert pred_bboxes.size(-1) == bboxes.size(-1) == 4
        decoded_bboxes = torch.zeros([1, bboxes.size(0), bboxes.size(1)], dtype=torch.float32, device="cuda")
        xy_centers = (bboxes[..., :2] + bboxes[..., 2:]) * 0.5 + (
            pred_bboxes[..., :2] - 0.5) * stride
        whs = (bboxes[..., 2:] -
            bboxes[..., :2]) * 0.5 * pred_bboxes[..., 2:].exp()
        # print(whs.shape)
        decoded_bboxes[..., 0].copy_((xy_centers[..., 0] - whs[..., 0]))
        decoded_bboxes[..., 1].copy_((xy_centers[..., 1] - whs[..., 1]))
        decoded_bboxes[..., 2].copy_((xy_centers[..., 0] + whs[..., 0]))
        decoded_bboxes[..., 3].copy_((xy_centers[..., 1] + whs[..., 1]))
        return decoded_bboxes.clone()


class YOLOAnchorGenerator(torch.nn.Module):
    def __init__(self, stride, base_size: List[Tuple[int, int]]):
        super().__init__()
        from torch.nn.modules.utils import _pair
        self.stride: Tuple[int, int] = _pair(stride)
        self.center: Tuple[float, float] = (stride / 2., stride / 2.)
        self.base_size = base_size
        self.base_anchor = self.gen_base_anchors()
    
    def gen_base_anchors(self):
        center = None
        center = self.center
        base_anchor = self.gen_single_level_base_anchors(self.base_size, center)
        return base_anchor
    
    def gen_single_level_base_anchors(self, base_size: Tuple[int, int], center: Tuple[float, float]):
        x_center, y_center = center
        base_anchors = []
        for base_size in base_size:
            w, h = base_size
            base_anchor = Tensor([
                x_center - 0.5 * w, y_center - 0.5 * h, x_center + 0.5 * w,
                y_center + 0.5 * h
            ])
            base_anchors.append(base_anchor)
        base_anchors = torch.stack(base_anchors, dim=0)

        return base_anchors.cuda()
    
    def forward(self, featmap_size: Tuple[int, int], dtype: torch.dtype, device: torch.device):
        # assert self.num_levels == len(featmap_sizes)
        anchors = self.single_level_grid_priors(featmap_size, dtype=dtype, device=device)
        return anchors
    
    def single_level_grid_priors(self,
                                 featmap_size: Tuple[int, int],
                                 dtype: torch.dtype,
                                 device: torch.device):
        base_anchor = self.base_anchor.to(device, dtype)
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.stride
        shift_x = torch.arange(0, feat_w, dtype=dtype,
                               device=device) * stride_w
        shift_y = torch.arange(0, feat_h, dtype=dtype,
                               device=device) * stride_h

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        all_anchors = base_anchor[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.reshape(-1, 4)

        return all_anchors
    
    def _meshgrid(self, x: Tensor, y: Tensor):
        xx = x.repeat(y.size(0))
        yy = y.reshape(-1, 1).repeat(1, x.size(0)).reshape(-1)
        return xx, yy


def multiclass_nms(multi_bboxes: Tensor,
                   multi_scores: Tensor,
                   score_factors: Tensor):
    score_thr = 0.05
    max_num = 100

    num_classes = multi_scores.size(1) - 1
    bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.reshape(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    valid_mask = scores > score_thr

    score_factors = score_factors.reshape(-1, 1).expand(
        multi_scores.size(0), num_classes)
    score_factors = score_factors.reshape(-1)
    scores = scores * score_factors

    inds = valid_mask.nonzero()[:, 0]
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]

    dets, keep = batched_nms(bboxes, scores, labels)
    dets = dets[:max_num]
    keep = keep[:max_num]

    return dets, labels[keep]


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
            mask = torch.nonzero(idxs == id).reshape(-1)
            dets, keep = nms_wrapper(
                boxes_for_nms[mask], scores[mask], iou_threshold)
            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = dets[:, -1]
        keep = total_mask.nonzero().reshape(-1)
        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

    boxes = torch.cat([boxes, scores[:, None]], -1)
    return boxes, keep


def nms_wrapper(boxes: Tensor,
                scores: Tensor,
                iou_threshold: float):
    # assert boxes.size(1) == 4
    # assert boxes.size(0) == scores.size(0)
    inds = torch.ops.torchvision.nms(boxes, scores, iou_threshold)
    dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)
    return dets, inds


class YOLOV3BBox(torch.nn.Module):
    def __init__(self, stride=32, base_size=[(116, 90), (156, 198), (373, 326)]) -> None:
        super().__init__()
        self.num_classes = 80
        self.num_attrib = 5 + self.num_classes
        self.stride = stride
        self.prior_generator = YOLOAnchorGenerator(
            stride = stride,
            base_size = base_size
        )
        self.decode_bboxes = DecodeBBoxes()

    def forward(self, pred_map: Tensor):
        # pred_maps = [torch.Size([1, 255, 10, 10]), torch.Size([1, 255, 20, 20]), torch.Size([1, 255, 40, 40])]
        featmap_strides = self.stride
        num_imgs = pred_map.size(0)
        # featmap_sizes = [(pred_map.size(-2), pred_map.size(-1))
        #                  for pred_map in pred_maps]
        featmap_size = (pred_map.size(-2), pred_map.size(-1))

        anchors = self.prior_generator(
            featmap_size, dtype=pred_map.dtype, device=pred_map.device)
        
        pred = pred_map.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                    self.num_attrib).clone()
        pred[..., :2].sigmoid_().clone()
        bbox_pred = pred[..., :4]
        objectness = pred[..., 4].sigmoid()
        cls_scores = pred[..., 5:].sigmoid()

        strides = torch.tensor(featmap_strides, device=pred.device).expand(pred.size(1))
        strides = strides
        bboxes = self.decode_bboxes(anchors, bbox_pred, strides.unsqueeze(-1)).clone()
        
        padding = bboxes.new_zeros(num_imgs, bboxes.size(1), 1)
        flatten_cls_scores = torch.cat([cls_scores, padding], dim=-1)

        det_results: List[Tuple[Tensor, Tensor]] = []
        conf_thr = 0.005
        conf_inds = objectness[0] >= conf_thr
        bboxes = bboxes[0, conf_inds, ]
        scores = flatten_cls_scores[0, conf_inds, ]
        objectness = objectness[0, conf_inds]
        det_results.append(multiclass_nms(bboxes, scores, objectness))

        return det_results






from typing import List, Tuple

from torch import Tensor

import torch
import torchvision

test_cfg = {'nms_pre': 1000, 'min_bbox_size': 0, 'score_thr': 0.05, 'conf_thr': 0.005,
            'nms': {'type': 'nms', 'iou_threshold': 0.45}, 'max_per_img': 100}


def decode_bboxes(bboxes, pred_bboxes, stride):
    # assert pred_bboxes.size(-1) == bboxes.size(-1) == 4
    xy_centers = (bboxes[..., :2] + bboxes[..., 2:]) * 0.5 + (
        pred_bboxes[..., :2] - 0.5) * stride
    whs = (bboxes[..., 2:] -
           bboxes[..., :2]) * 0.5 * pred_bboxes[..., 2:].exp()
    decoded_bboxes = torch.stack(
        (xy_centers[..., 0] - whs[..., 0], xy_centers[..., 1] -
            whs[..., 1], xy_centers[..., 0] + whs[..., 0],
            xy_centers[..., 1] + whs[..., 1]),
        dim=-1)
    return decoded_bboxes


class YOLOAnchorGenerator(torch.nn.Module):
    def __init__(self, strides, base_sizes):
        super().__init__()
        from torch.nn.modules.utils import _pair
        self.strides = [_pair(stride) for stride in strides]
        self.centers = [(stride[0] / 2., stride[1] / 2.)
                        for stride in self.strides]
        self.base_sizes = []
        for base_sizes_per_level in base_sizes:
            # assert len(base_sizes[0]) == len(base_sizes_per_level)
            self.base_sizes.append(
                [_pair(base_size) for base_size in base_sizes_per_level])
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_levels(self):
        return len(self.base_sizes)

    def gen_base_anchors(self):
        multi_level_base_anchors: List[Tensor] = []
        for i, base_sizes_per_level in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(base_sizes_per_level,
                                                   center))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self, base_sizes_per_level: List[Tuple[int, int]], center: Tuple[float]):
        x_center, y_center = center
        base_anchors = []
        for base_size in base_sizes_per_level:
            w, h = base_size
            base_anchor = Tensor([
                x_center - 0.5 * w, y_center - 0.5 * h, x_center + 0.5 * w,
                y_center + 0.5 * h
            ])
            base_anchors.append(base_anchor)
        base_anchors = torch.stack(base_anchors, dim=0)

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
        base_anchors = self.base_anchors[level_idx].to(device, dtype)
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = torch.arange(0, feat_w, dtype=dtype,
                               device=device) * stride_w
        shift_y = torch.arange(0, feat_h, dtype=dtype,
                               device=device) * stride_h

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
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
    inds = torchvision.ops.nms(boxes, scores, iou_threshold)
    dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)
    return dets, inds


class YOLOV3BBox(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_classes = 80
        self.num_attrib = 5 + self.num_classes
        self.prior_generator = YOLOAnchorGenerator(
            strides=[32, 16, 8],
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]],)

    def forward(self, pred_maps: List[Tensor]):
        # pred_maps = [torch.Size([1, 255, 10, 10]), torch.Size([1, 255, 20, 20]), torch.Size([1, 255, 40, 40])]
        featmap_strides = [32, 16, 8]
        num_imgs = pred_maps[0].size(0)
        featmap_sizes = [(pred_map.size(-2), pred_map.size(-1))
                         for pred_map in pred_maps]

        print('anchor', True)
        mlvl_anchors = self.prior_generator(
            featmap_sizes, dtype=pred_maps[0].dtype, device=pred_maps[0].device)
        flatten_anchors = torch.cat(mlvl_anchors)
        print('anchor', False)

        print('flatten', True)
        flatten_preds = []
        flatten_strides = []
        for pred, stride in zip(pred_maps, featmap_strides):
            pred = pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                    self.num_attrib)
            pred[..., :2].sigmoid_()
            flatten_preds.append(pred)
            flatten_strides.append(torch.tensor(
                stride, device=pred.device).expand(pred.size(1)))
        flatten_preds = torch.cat(flatten_preds, dim=1)
        print('flatten', False)
        print('decode', True)
        flatten_bbox_preds = flatten_preds[..., :4]
        flatten_objectness = flatten_preds[..., 4].sigmoid()
        flatten_cls_scores = flatten_preds[..., 5:].sigmoid()
        flatten_strides = torch.cat(flatten_strides)
        flatten_bboxes = decode_bboxes(flatten_anchors,
                                       flatten_bbox_preds,
                                       flatten_strides.unsqueeze(-1))
        padding = flatten_bboxes.new_zeros(num_imgs, flatten_bboxes.size(1), 1)
        flatten_cls_scores = torch.cat([flatten_cls_scores, padding], dim=-1)
        print('decode', False)

        print('nms', True)
        det_results: List[Tuple[Tensor, Tensor]] = []
        for (bboxes, scores, objectness) in zip(flatten_bboxes,
                                                flatten_cls_scores,
                                                flatten_objectness):
            conf_thr = 0.005
            conf_inds = objectness >= conf_thr
            bboxes = bboxes[conf_inds, :]
            scores = scores[conf_inds, :]
            objectness = objectness[conf_inds]
            det_results.append(multiclass_nms(bboxes, scores, objectness))
        print('nms', False)

        return det_results


if __name__ == '__main__':
    mod = YOLOV3BBox().cuda().eval()
    mod = torch.jit.script(mod)
    print(mod.graph)
    torch.jit.save(mod, 'yolov3_bbox_prof.pt')

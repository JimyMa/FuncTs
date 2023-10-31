from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn.modules.utils import _pair

import torchvision


class MlvlPointGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        strides = [8, 16, 32]
        self.strides = [_pair(stride) for stride in strides]
        offset = 0
        self.offset = offset

    @property
    def num_levels(self):
        return len(self.strides)

    def single_level_grid_priors(self,
                                 featmap_size: Tuple[int, int],
                                 level_idx: int,
                                 dtype: torch.dtype =torch.float32,
                                 device: torch.device ='cuda'):
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = (torch.arange(0, feat_w, device=device) +
                   self.offset) * stride_w
        shift_x = shift_x.to(dtype)
        shift_y = (torch.arange(0, feat_h, device=device) +
                   self.offset) * stride_h
        shift_y = shift_y.to(dtype)
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        return shifts

    def grid_priors(self,
                    featmap_sizes: List[Tuple[int, int]],
                    dtype: torch.dtype=torch.float32,
                    device: torch.device='cuda'):
        multi_level_priors = []
        for i in range(self.num_levels):
            priors = self.single_level_grid_priors(
                featmap_sizes[i],
                level_idx=i,
                dtype=dtype,
                device=device)
            multi_level_priors.append(priors)
        return multi_level_priors
    
    def _meshgrid(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        xx = x.repeat(y.size(0))
        yy = y.view(-1, 1).repeat(1, x.size(0)).view(-1)
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
    scores, idxs = torch.sort(scores, descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs[:, 0], topk_idxs[:, 1]
    return scores, labels, keep_idxs


def distance2bbox(
    points: Tensor, distance: Tensor, max_shape: Tuple[int, int]
) -> Tensor:
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    bboxes = torch.stack([x1, y1, x2, y2], -1)
    bboxes[:, 0::2].clamp_(min=0, max=max_shape[1])
    bboxes[:, 1::2].clamp_(min=0, max=max_shape[0])
    return bboxes


def batched_nms(boxes: Tensor, scores: Tensor, idxs: Tensor):
    iou_threshold = 0.5
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
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
            mask = (idxs == id).nonzero().view(-1)
            dets, keep = nms_wrapper(
                boxes_for_nms[mask], scores[mask], iou_threshold)
            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = dets[:, -1]
        keep = total_mask.nonzero().view(-1)
        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

    boxes = torch.cat([boxes, scores[:, None]], -1)
    return boxes, keep


def nms_wrapper(boxes: Tensor, scores: Tensor, iou_threshold: float):
    # assert boxes.size(1) == 4
    # assert boxes.size(0) == scores.size(0)
    inds = torchvision.ops.nms(boxes, scores, iou_threshold)
    dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)
    return dets, inds


class RTMDetBBox(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prior_generator = MlvlPointGenerator()

    def forward(self,
                cls_scores:List[Tensor],
                bbox_preds:List[Tensor]):
        featmap_sizes = [(score.size(-2), score.size(-1)) for score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list: List[tuple[Tensor, Tensor]] = []

        for img_id in range(cls_scores[0].size(0)):
            # img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id)

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                mlvl_priors=mlvl_priors)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                mlvl_priors: List[Tensor]):
        img_shape = (320, 320)
        nms_pre = 1000
        cls_out_channels = 80

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []

        for cls_score, bbox_pred, priors in zip(cls_score_list, bbox_pred_list, mlvl_priors):
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, cls_out_channels)
            scores = cls_score.sigmoid()

            score_thr = 0.05
            results = filter_scores_and_topk(
                scores, score_thr, nms_pre)
            scores, labels, keep_idxs = results
            bbox_pred = bbox_pred[keep_idxs]
            priors = priors[keep_idxs]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = torch.cat(mlvl_valid_priors)
        scores = torch.cat(mlvl_scores)
        labels = torch.cat(mlvl_labels)
        bboxes = distance2bbox(priors, bbox_pred, max_shape=img_shape)

        
        if bboxes.numel() == 0:
            det_bboxes = torch.cat([bboxes, scores[:, None]], -1)
            return det_bboxes, labels

        det_bboxes, keep_idxs = batched_nms(bboxes, scores, labels)
        max_per_img = 100
        det_bboxes = det_bboxes[:max_per_img]
        det_labels = labels[keep_idxs][:max_per_img]

        return det_bboxes, det_labels


if __name__ == '__main__':
    bbox = RTMDetBBox().cuda().eval()
    bbox = torch.jit.script(bbox)
    print(bbox.graph)
    torch.jit.save(bbox, 'rtmdet_bbox.pt')

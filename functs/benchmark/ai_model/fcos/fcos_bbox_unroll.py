from typing import List, Tuple

import torch
import torchvision
from torch import Tensor


class MlvlPointGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.strides = [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128)]
        self.offset = 0.5

    @property
    def num_levels(self) -> int:
        return len(self.strides)

    def forward(
        self,
        featmap_sizes: List[Tuple[int, int]],
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cuda",
    ) -> List[Tensor]:
        # assert self.num_levels == len(featmap_sizes)
        multi_level_priors = []
        for i in range(self.num_levels):
            feat_h, feat_w = featmap_sizes[i]
            stride_w, stride_h = self.strides[i]
            shift_x = (torch.arange(0, feat_w, dtype=dtype, device=device) +
                    self.offset) * stride_w
            shift_y = (torch.arange(0, feat_h, dtype=dtype, device=device) +
                    self.offset) * stride_h
            shifts = torch.zeros(feat_h * feat_w, 2, device="cuda", dtype=torch.float32)
            shift_xx = shift_x.repeat(feat_h)
            shift_yy = shift_y.reshape(-1, 1).repeat(1, feat_w).reshape(-1)
            # shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
            shifts[:, 0].copy_(shift_xx)
            shifts[:, 1].copy_(shift_yy)
            priors = shifts
            multi_level_priors.append(priors)
        return multi_level_priors

    def single_level_grid_priors(
        self,
        featmap_size: Tuple[int, int],
        level_idx: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cuda",
    ) -> Tensor:
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = (torch.arange(0, feat_w, dtype=dtype, device=device) +
                   self.offset) * stride_w
        shift_y = (torch.arange(0, feat_h, dtype=dtype, device=device) +
                   self.offset) * stride_h
        shifts = torch.zeros(feat_h * feat_w, 2, device="cuda", dtype=torch.float32)
        shift_xx = shift_x.repeat(feat_h)
        shift_yy = shift_y.reshape(-1, 1).repeat(1, feat_w).reshape(-1)
        # shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts[:, 0].copy_(shift_xx)
        shifts[:, 1].copy_(shift_yy)
        # shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        # shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        return shifts

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


def distance2bbox(
    points: Tensor, distance: Tensor, max_shape: Tuple[int, int]
) -> Tensor:
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]

    bboxes = distance.clone()
    bboxes[..., 0].copy_(x1)
    bboxes[..., 1].copy_(y1)
    bboxes[..., 2].copy_(x2)
    bboxes[..., 3].copy_(y2)
    torch.clamp_(bboxes, min=0, max=max_shape[1])
    # bboxes = torch.stack([x1, y1, x2, y2], -1)
    # bboxes[:, 0::2].clamp_(min=0, max=max_shape[1])
    # bboxes[:, 1::2].clamp_(min=0, max=max_shape[0])
    return bboxes.clone()


class FCOSBBox(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prior_generator = MlvlPointGenerator()
        self.strides = [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128)]
        self.offset = 0.5

    @property
    def num_levels(self) -> int:
        return len(self.strides)

    def forward(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        score_factors: List[Tensor],
    ):
        # cls_scores: [torch.Size([1, 80, 40, 40]), torch.Size([1, 80, 20, 20]), torch.Size([1, 80, 10, 10]), torch.Size([1, 80, 5, 5]), torch.Size([1, 80, 3, 3])]
        # bbox_preds: [torch.Size([1, 4, 40, 40]), torch.Size([1, 4, 20, 20]), torch.Size([1, 4, 10, 10]), torch.Size([1, 4, 5, 5]), torch.Size([1, 4, 3, 3])]
        # score_factors: [torch.Size([1, 1, 40, 40]), torch.Size([1, 1, 20, 20]), torch.Size([1, 1, 10, 10]), torch.Size([1, 1, 5, 5]), torch.Size([1, 1, 3, 3])]
        featmap_sizes = [(score.size(-2), score.size(-1))
                         for score in cls_scores]
        
        mlvl_priors = self.prior_generator(
            featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device
        )

        mlvl_priors = []
        # for i in range(self.num_levels):
        #     feat_h, feat_w = featmap_sizes[i]
        #     stride_w, stride_h = self.strides[i]
        #     shift_x = (torch.arange(0, feat_w, dtype=torch.float32, device="cuda") +
        #             self.offset) * stride_w
        #     shift_y = (torch.arange(0, feat_h, dtype=torch.float32, device="cuda") +
        #             self.offset) * stride_h
        #     shifts = torch.zeros(feat_h * feat_w, 2, device="cuda", dtype=torch.float32)
        #     shift_xx = shift_x.repeat(feat_h)
        #     shift_yy = shift_y.reshape(-1, 1).repeat(1, feat_w).reshape(-1)
        #     # shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        #     shifts[:, 0].copy_(shift_xx)
        #     shifts[:, 1].copy_(shift_yy)
        #     priors = shifts
        #     mlvl_priors.append(priors)

        feat_h, feat_w = featmap_sizes[0]
        stride_w, stride_h = self.strides[0]
        shift_x = (torch.arange(0, 40, dtype=torch.float32, device="cuda") + self.offset) * stride_w
        shift_y = (torch.arange(0, 40, dtype=torch.float32, device="cuda") + self.offset) * stride_h
        shifts = torch.zeros(40 * 40, 2, device="cuda", dtype=torch.float32)
        shift_xx = shift_x.repeat(40)
        shift_yy = shift_y.reshape(-1, 1).repeat(1, 40).reshape(-1)
        # shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts[:, 0].copy_(shift_xx)
        shifts[:, 1].copy_(shift_yy)
        priors = shifts.clone()
        mlvl_priors.append(priors)

        feat_h, feat_w = featmap_sizes[1]
        stride_w, stride_h = self.strides[1]
        shift_x = (torch.arange(0, 20, dtype=torch.float32, device="cuda") + self.offset) * stride_w
        shift_y = (torch.arange(0, 20, dtype=torch.float32, device="cuda") + self.offset) * stride_h
        shifts = torch.zeros(20 * 20, 2, device="cuda", dtype=torch.float32)
        shift_xx = shift_x.repeat(20)
        shift_yy = shift_y.reshape(-1, 1).repeat(1, 20).reshape(-1)
        # shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts[:, 0].copy_(shift_xx)
        shifts[:, 1].copy_(shift_yy)
        priors = shifts.clone()
        mlvl_priors.append(priors)

        feat_h, feat_w = featmap_sizes[2]
        stride_w, stride_h = self.strides[2]
        shift_x = (torch.arange(0, 10, dtype=torch.float32, device="cuda") + self.offset) * stride_w
        shift_y = (torch.arange(0, 10, dtype=torch.float32, device="cuda") + self.offset) * stride_h
        shifts = torch.zeros(10 * 10, 2, device="cuda", dtype=torch.float32)
        shift_xx = shift_x.repeat(10)
        shift_yy = shift_y.reshape(-1, 1).repeat(1, 10).reshape(-1)
        # shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts[:, 0].copy_(shift_xx)
        shifts[:, 1].copy_(shift_yy)
        priors = shifts.clone()
        mlvl_priors.append(priors)

        feat_h, feat_w = featmap_sizes[3]
        stride_w, stride_h = self.strides[3]
        shift_x = (torch.arange(0, 5, dtype=torch.float32, device="cuda") + self.offset) * stride_w
        shift_y = (torch.arange(0, 5, dtype=torch.float32, device="cuda") + self.offset) * stride_h
        shifts = torch.zeros(5 * 5, 2, device="cuda", dtype=torch.float32)
        shift_xx = shift_x.repeat(5)
        shift_yy = shift_y.reshape(-1, 1).repeat(1, 5).reshape(-1)
        # shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts[:, 0].copy_(shift_xx)
        shifts[:, 1].copy_(shift_yy)
        priors = shifts.clone()
        mlvl_priors.append(priors)

        feat_h, feat_w = featmap_sizes[4]
        stride_w, stride_h = self.strides[4]
        shift_x = (torch.arange(0, 3, dtype=torch.float32, device="cuda") + self.offset) * stride_w
        shift_y = (torch.arange(0, 3, dtype=torch.float32, device="cuda") + self.offset) * stride_h
        shifts = torch.zeros(3 * 3, 2, device="cuda", dtype=torch.float32)
        shift_xx = shift_x.repeat(3)
        shift_yy = shift_y.reshape(-1, 1).repeat(1, 3).reshape(-1)
        # shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts[:, 0].copy_(shift_xx)
        shifts[:, 1].copy_(shift_yy)
        priors = shifts.clone()
        mlvl_priors.append(priors)

        result_list: List[Tuple[Tensor, Tensor]] = []
        for img_id in range(cls_scores[0].size(0)):
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            score_factor_list = select_single_mlvl(score_factors, img_id)
            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                score_factor_list=score_factor_list,
                mlvl_priors=mlvl_priors,
            )
            result_list.append(results)

        return result_list

    def _predict_by_feat_single(
        self,
        cls_score_list: List[Tensor],
        bbox_pred_list: List[Tensor],
        score_factor_list: List[Tensor],
        mlvl_priors: List[Tensor],
    ):
        img_shape = (320, 320)
        nms_pre = 1000

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []
        mlvl_score_factors = []
        # for cls_score, bbox_pred, score_factor, priors in zip(
        #     cls_score_list, bbox_pred_list, score_factor_list, mlvl_priors
        # ):
        #     # assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        #     bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
        #     score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
        #     cls_score = cls_score.permute(1, 2, 0).reshape(-1, 80)
        #     scores = cls_score.sigmoid()

        #     score_thr = 0.025
        #     scores, labels, keep_idxs = filter_scores_and_topk(
        #         scores, score_thr, nms_pre
        #     )
        #     bbox_pred = bbox_pred[keep_idxs]
        #     priors = priors[keep_idxs]
        #     score_factor = score_factor[keep_idxs]

        #     mlvl_bbox_preds.append(bbox_pred)
        #     mlvl_valid_priors.append(priors)
        #     mlvl_scores.append(scores)
        #     mlvl_labels.append(labels)
        #     mlvl_score_factors.append(score_factor)

        # assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        cls_score = cls_score_list[0].clone()
        bbox_pred = bbox_pred_list[0].clone()
        score_factor = score_factor_list[0].clone()
        priors = mlvl_priors[0].clone()
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
        score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, 80)
        scores = cls_score.sigmoid()

        score_thr = 0.025
        scores, labels, keep_idxs = filter_scores_and_topk(
            scores, score_thr, nms_pre
        )
        bbox_pred = bbox_pred[keep_idxs]
        priors = priors[keep_idxs]
        score_factor = score_factor[keep_idxs]

        mlvl_bbox_preds.append(bbox_pred.clone())
        mlvl_valid_priors.append(priors.clone())
        mlvl_scores.append(scores.clone())
        mlvl_labels.append(labels.clone())
        mlvl_score_factors.append(score_factor.clone())

        cls_score = cls_score_list[1].clone()
        bbox_pred = bbox_pred_list[1].clone()
        score_factor = score_factor_list[1].clone()
        priors = mlvl_priors[1].clone()
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
        score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, 80)
        scores = cls_score.sigmoid()

        score_thr = 0.025
        scores, labels, keep_idxs = filter_scores_and_topk(
            scores, score_thr, nms_pre
        )
        bbox_pred = bbox_pred[keep_idxs]
        priors = priors[keep_idxs]
        score_factor = score_factor[keep_idxs]

        mlvl_bbox_preds.append(bbox_pred.clone())
        mlvl_valid_priors.append(priors.clone())
        mlvl_scores.append(scores.clone())
        mlvl_labels.append(labels.clone())
        mlvl_score_factors.append(score_factor.clone())

        cls_score = cls_score_list[2]
        bbox_pred = bbox_pred_list[2]
        score_factor = score_factor_list[2]
        priors = mlvl_priors[2]
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
        score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, 80)
        scores = cls_score.sigmoid()

        score_thr = 0.025
        scores, labels, keep_idxs = filter_scores_and_topk(
            scores, score_thr, nms_pre
        )
        bbox_pred = bbox_pred[keep_idxs]
        priors = priors[keep_idxs]
        score_factor = score_factor[keep_idxs]

        mlvl_bbox_preds.append(bbox_pred.clone())
        mlvl_valid_priors.append(priors.clone())
        mlvl_scores.append(scores.clone())
        mlvl_labels.append(labels.clone())
        mlvl_score_factors.append(score_factor.clone())

        cls_score = cls_score_list[3].clone()
        bbox_pred = bbox_pred_list[3].clone()
        score_factor = score_factor_list[3].clone()
        priors = mlvl_priors[3].clone()
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
        score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, 80)
        scores = cls_score.sigmoid()

        score_thr = 0.025
        scores, labels, keep_idxs = filter_scores_and_topk(
            scores, score_thr, nms_pre
        )
        bbox_pred = bbox_pred[keep_idxs]
        priors = priors[keep_idxs]
        score_factor = score_factor[keep_idxs]

        mlvl_bbox_preds.append(bbox_pred.clone())
        mlvl_valid_priors.append(priors.clone())
        mlvl_scores.append(scores.clone())
        mlvl_labels.append(labels.clone())
        mlvl_score_factors.append(score_factor.clone())

        cls_score = cls_score_list[4].clone()
        bbox_pred = bbox_pred_list[4].clone()
        score_factor = score_factor_list[4].clone()
        priors = mlvl_priors[4].clone()
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
        score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, 80)
        scores = cls_score.sigmoid()

        score_thr = 0.025
        scores, labels, keep_idxs = filter_scores_and_topk(
            scores, score_thr, nms_pre
        )
        bbox_pred = bbox_pred[keep_idxs]
        priors = priors[keep_idxs]
        score_factor = score_factor[keep_idxs]

        mlvl_bbox_preds.append(bbox_pred.clone())
        mlvl_valid_priors.append(priors.clone())
        mlvl_scores.append(scores.clone())
        mlvl_labels.append(labels.clone())
        mlvl_score_factors.append(score_factor.clone())

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = torch.cat(mlvl_valid_priors)
        bboxes = distance2bbox(priors, bbox_pred, max_shape=img_shape)
        scores = torch.cat(mlvl_scores)
        labels = torch.cat(mlvl_labels)
        score_factors = torch.cat(mlvl_score_factors)
        scores *= score_factors

        if bboxes.numel() == 0:
            det_bboxes = torch.cat([bboxes, scores[:, None]], -1)
            return det_bboxes, labels

        det_bboxes, keep_idxs = batched_nms(bboxes, scores, labels)
        max_per_img = 100
        det_bboxes = det_bboxes[:max_per_img]
        det_labels = labels[keep_idxs][:max_per_img]

        return det_bboxes, det_labels


if __name__ == '__main__':
    mask = FCOSBBox().cuda().eval()
    mask = torch.jit.script(mask)
    print(mask.graph)
    torch.jit.save(mask, 'fcos_bbox.pt')

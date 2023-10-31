from typing import List, Tuple

from torch import Tensor

import torch


def mask_matrix_nms(masks: Tensor,
                    labels: Tensor,
                    scores: Tensor,
                    mask_area: Tensor):
    nms_pre = 500
    max_num = 100
    sigma = 2.0
    filter_thr = 0.05

    # assert len(labels) == len(masks) == len(scores)
    # assert len(masks) == len(mask_area)

    # sort and keep top nms_pre
    scores, sort_inds = torch.sort(scores, descending=True)
    keep_inds = sort_inds
    sort_inds = sort_inds[:nms_pre]
    keep_inds = keep_inds[:nms_pre]
    scores = scores[:nms_pre]
    masks = masks[sort_inds]
    mask_area = mask_area[sort_inds]
    labels = labels[sort_inds]

    num_masks = labels.size(0)
    flatten_masks = masks.reshape(
        num_masks, masks.size(-2) * masks.size(-1)).float()
    # inter.
    inter_matrix = torch.mm(flatten_masks, flatten_masks.transpose(1, 0))
    expanded_mask_area = mask_area.expand(num_masks, num_masks)
    # Upper triangle iou matrix.
    iou_matrix = (inter_matrix /
                  (expanded_mask_area + expanded_mask_area.transpose(1, 0) -
                   inter_matrix)).triu(diagonal=1)
    # label_specific matrix.
    expanded_labels = labels.expand(num_masks, num_masks)
    # Upper triangle label matrix.
    label_matrix = (expanded_labels == expanded_labels.transpose(
        1, 0)).triu(diagonal=1)

    # IoU compensation
    if iou_matrix.size(0) != 0:
        compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    else:
        compensate_iou = iou_matrix.new_zeros(0)
    compensate_iou = compensate_iou.expand(num_masks,
                                           num_masks).transpose(1, 0)

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # Calculate the decay_coefficient
    decay_matrix = torch.exp(-1 * sigma * (decay_iou**2))
    compensate_matrix = torch.exp(-1 * sigma * (compensate_iou**2))
    if decay_matrix.size(0) != 0:
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    else:
        decay_coefficient = decay_matrix.new_zeros(0)
    # update the score.
    scores = scores * decay_coefficient

    keep = scores >= filter_thr
    keep_inds = keep_inds[keep]
    masks = masks[keep]
    scores = scores[keep]
    labels = labels[keep]

    # sort and keep top max_num
    scores, sort_inds = torch.sort(scores, descending=True)
    keep_inds = keep_inds[sort_inds]
    sort_inds = sort_inds[:max_num]
    keep_inds = keep_inds[:max_num]
    scores = scores[:max_num]
    masks = masks[sort_inds]
    labels = labels[sort_inds]

    return scores, labels, masks, keep_inds


class SOLOV2Mask(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_levels = 5
        self.mask_stride = 4
        self.cls_out_channels = 80
        self.kernel_out_channels = 128
        self.num_grids = [40, 36, 24, 16, 12]
        self.strides = [8, 8, 16, 32, 32]
        self.dynamic_conv_size = 1

    def forward(self,
                mlvl_kernel_preds: List[Tensor],
                mlvl_cls_scores: List[Tensor],
                mask_feats: Tensor):
        # assert len(mlvl_kernel_preds) == len(mlvl_cls_scores)
        # mlvl_kernel_preds: [torch.Size([1, 128, 40, 40]), torch.Size([1, 128, 36, 36]), torch.Size([1, 128, 24, 24]), torch.Size([1, 128, 16, 16]), torch.Size([1, 128, 12, 12])]
        # mlvl_cls_preds: [torch.Size([1, 80, 40, 40]), torch.Size([1, 80, 36, 36]), torch.Size([1, 80, 24, 24]), torch.Size([1, 80, 16, 16]), torch.Size([1, 80, 12, 12])]
        # mask_feats: torch.Size([1, 128, 80, 80])

        num_levels = len(mlvl_cls_scores)
        lvl_cls_scores = []
        for cls_scores in mlvl_cls_scores:
            cls_scores = cls_scores.sigmoid()
            local_max = torch.nn.functional.max_pool2d(
                cls_scores, 2, stride=1, padding=1)
            keep_mask = local_max[:, :, :-1, :-1] == cls_scores
            cls_scores = cls_scores * keep_mask
            lvl_cls_scores.append(cls_scores.permute(0, 2, 3, 1))
        mlvl_cls_scores = lvl_cls_scores

        result_list: List[Tuple[Tensor, Tensor, Tensor]] = []
        for img_id in range(mlvl_kernel_preds[0].size(0)):
            img_cls_pred = [
                mlvl_cls_scores[lvl][img_id].view(-1, self.cls_out_channels)
                for lvl in range(num_levels)
            ]
            img_mask_feats = mask_feats[img_id]
            img_kernel_pred = [
                mlvl_kernel_preds[lvl][img_id].permute(1, 2, 0).view(
                    -1, self.kernel_out_channels) for lvl in range(num_levels)
            ]
            img_cls_pred = torch.cat(img_cls_pred, dim=0)
            img_kernel_pred = torch.cat(img_kernel_pred, dim=0)
            result = self._get_results_single(
                img_kernel_pred,
                img_cls_pred,
                img_mask_feats)
            result_list.append(result)

        return result_list

    def _get_results_single(self,
                            kernel_preds: Tensor,
                            cls_scores: Tensor,
                            mask_feats: Tensor,):
        # assert len(kernel_preds) == len(cls_scores)
        featmap_size = mask_feats.size(-2), mask_feats.size(-1)

        # overall info
        img_shape = (320, 320)
        h, w = img_shape
        upsampled_size = (featmap_size[0] * self.mask_stride,
                          featmap_size[1] * self.mask_stride)

        # process.
        score_mask = (cls_scores > 0.1)
        cls_scores = cls_scores[score_mask]
        # if len(cls_scores) == 0:
        # return empty_results(results, cls_scores)

        # cate_labels & kernel_preds
        inds = score_mask.nonzero()
        cls_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        # trans vector.
        lvl_interval = torch.tensor(
            self.num_grids, dtype=cls_labels.dtype, device=cls_labels.device).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(lvl_interval[-1])

        strides[:lvl_interval[0]] *= self.strides[0]
        for lvl in range(self.num_levels - 1):
            strides[lvl_interval[lvl]:lvl_interval[lvl + 1]
                    ] *= self.strides[lvl + 1]
        strides = strides[inds[:, 0]]

        # mask encoding
        if kernel_preds.size(0) != 0:
            kernel_preds = kernel_preds.view(
                kernel_preds.size(0), kernel_preds.size(
                    1), self.dynamic_conv_size,
                self.dynamic_conv_size)
            mask_preds = torch.nn.functional.conv2d(
                mask_feats.unsqueeze(0), kernel_preds, stride=1).sigmoid()[0]
            # mask.
            masks = (mask_preds > 0.5)
            sum_masks = masks.long().sum((1, 2)).float()
            keep = sum_masks > strides
            # if keep.sum() == 0:
            # return empty_results(results, cls_scores)
            masks = masks[keep]
            mask_preds = mask_preds[keep]
            sum_masks = sum_masks[keep]
            cls_scores = cls_scores[keep]
            cls_labels = cls_labels[keep]
            # maskness.
            mask_scores = (mask_preds * masks).sum((1, 2)) / sum_masks
            cls_scores *= mask_scores
        else:
            mask_preds = mask_feats.new_zeros(
                0, mask_feats.size(-2), mask_feats.size(-1), dtype=torch.bool)
            masks = torch.zeros_like(mask_preds, dtype=torch.bool)
            cls_labels = cls_labels.new_zeros(0)
            cls_scores = cls_scores.new_zeros(0)
            sum_masks = mask_feats.new_zeros(0)

        scores, labels, _, keep_inds = mask_matrix_nms(
            masks,
            cls_labels,
            cls_scores,
            mask_area=sum_masks)
        mask_preds = mask_preds[keep_inds]
        if mask_preds.size(0) != 0:
            mask_preds = torch.nn.functional.interpolate(
                mask_preds.unsqueeze(0),
                size=upsampled_size,
                mode='bilinear',
                align_corners=False)[:, :, :h, :w]
            mask_preds = torch.nn.functional.interpolate(
                mask_preds,
                size=img_shape,
                mode='bilinear',
                align_corners=False).squeeze(0)
        else:
            mask_preds = torch.zeros(0, h, w, dtype=mask_preds.dtype)
        masks = mask_preds > 0.5

        return masks, labels, scores


if __name__ == '__main__':
    mask = SOLOV2Mask().cuda().eval()
    mask = torch.jit.script(mask)
    print(mask.graph)
    torch.jit.save(mask, 'solov2_mask.pt')

from typing import Tuple

import torch
from torch import nn
from torch import Tensor

from mmcv.cnn import ConvModule, is_norm

from mmengine.model import bias_init_with_prob, constant_init, normal_init

from mmdet.models.layers.transformer import inverse_sigmoid

from mmdet.models.backbones import CSPNeXt
from mmdet.models.necks import CSPNeXtPAFPN
from mmdet.models.dense_heads import RTMDetHead
from mmdet.models.utils import sigmoid_geometric_mean


class RTMDetSepBNHead(RTMDetHead):
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 share_conv: bool = True,
                 norm_cfg = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg = dict(type='SiLU'),
                 pred_kernel_size: int = 1,
                 exp_on_reg=False,
                 **kwargs) -> None:
        self.share_conv = share_conv
        self.exp_on_reg = exp_on_reg
        super().__init__(
            num_classes,
            in_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            pred_kernel_size=pred_kernel_size,
            **kwargs)

    def _init_layers(self) -> None:
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        if self.with_objectness:
            self.rtm_obj = nn.ModuleList()
        for n in range(len(self.prior_generator.strides)):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

            self.rtm_cls.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.cls_out_channels,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))
            self.rtm_reg.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * 4,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))
            if self.with_objectness:
                self.rtm_obj.append(
                    nn.Conv2d(
                        self.feat_channels,
                        1,
                        self.pred_kernel_size,
                        padding=self.pred_kernel_size // 2))

        if self.share_conv:
            for n in range(len(self.prior_generator.strides)):
                for i in range(self.stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)
        bias_cls = bias_init_with_prob(0.01)
        for rtm_cls, rtm_reg in zip(self.rtm_cls, self.rtm_reg):
            normal_init(rtm_cls, std=0.01, bias=bias_cls)
            normal_init(rtm_reg, std=0.01)
        if self.with_objectness:
            for rtm_obj in self.rtm_obj:
                normal_init(rtm_obj, std=0.01, bias=bias_cls)

    def forward(self, feats: Tuple[Tensor, ...]) -> tuple:
        cls_scores = []
        bbox_preds = []
        for idx, (x, stride) in enumerate(
                zip(feats, self.prior_generator.strides)):
            cls_feat = x
            reg_feat = x

            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls[idx](cls_feat)

            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)

            if self.with_objectness:
                objectness = self.rtm_obj[idx](reg_feat)
                cls_score = inverse_sigmoid(
                    sigmoid_geometric_mean(cls_score, objectness))
            if self.exp_on_reg:
                reg_dist = self.rtm_reg[idx](reg_feat).exp() * torch.tensor((stride[0]), device="cuda")
            else:
                reg_dist = self.rtm_reg[idx](reg_feat) * torch.tensor((stride[0]), device="cuda")
            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
        return cls_scores, bbox_preds



class RTMDetNet(torch.nn.Module):
    ckpt_file = 'models/cspnext-tiny_imagenet_600e.pth'
    def __init__(self):
        super().__init__()
        self.backbone = CSPNeXt(
            arch='P5',
            expand_ratio=0.5,
            deepen_factor=0.167,
            widen_factor=0.375,
            channel_attention=True,
            norm_cfg=dict(type='SyncBN'),
            act_cfg=dict(type='SiLU'),
            init_cfg=dict(
                type='Pretrained', prefix='backbone.')
        )

        self.neck = CSPNeXtPAFPN(
            in_channels=[96, 192, 384], 
            out_channels=96, 
            num_csp_blocks=1,
            expand_ratio=0.5,
            norm_cfg=dict(type='SyncBN'),
            act_cfg=dict(type='SiLU')
        )

        self.head = RTMDetSepBNHead(
            num_classes=80,
            in_channels=96, 
            stacked_convs=2,
            feat_channels=96, 
            exp_on_reg=False,
            anchor_generator=dict(
                type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
            bbox_coder=dict(type='DistancePointBBoxCoder'),
            loss_cls=dict(
                type='QualityFocalLoss',
                use_sigmoid=True,
                beta=2.0,
                loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
            with_objectness=False,
            share_conv=True,
            pred_kernel_size=1,
            norm_cfg=dict(type='SyncBN'),
            act_cfg=dict(type='SiLU')
        )
    
    def forward(self, img):
        feat = self.backbone(img)
        feats = self.neck(feat)
        pred_map = self.head(feats)
        return pred_map


if __name__ == '__main__':
    net = RTMDetNet()
    net._load_from_state_dict(torch.load(RTMDetNet.ckpt_file)["state_dict"],
                              "backbone",
                              {},
                              True,
                              {},
                              [],
                              [])
    net = net.cuda().eval()
    net = torch.jit.trace(net, torch.empty(
        1, 3, 320, 320).float().cuda())
    net = torch.jit.freeze(net)
    torch.jit.save(net, 'rtmdet_net.pt')
    print(net.graph)






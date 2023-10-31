import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmdet.models.layers import InvertedResidual
from mmdet.models.utils import make_divisible

conv_cfg = None
norm_cfg = {'type': 'BN', 'eps': 0.001, 'momentum': 0.03}
act_cfg = {'type': 'ReLU6'}


class SSD(torch.nn.Module):
    ckpt_file = 'models/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pt'

    def __init__(self) -> None:
        super().__init__()
        self.backbone = MobileNetV2()
        self.neck = SSDNeck()
        self.bbox_head = SSDHead()

    def forward(self, img):
        feat = self.backbone(img)
        feats = self.neck(feat)
        return self.bbox_head(feats)


class MobileNetV2(torch.nn.Module):
    # Parameters to build layers. 4 parameters are needed to construct a
    # layer, from left to right: expand_ratio, channel, num_blocks, stride.
    arch_settings = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2],
                     [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2],
                     [6, 320, 1, 1]]

    def __init__(self):
        super().__init__()

        self.widen_factor = 1.
        self.out_indices = (4, 7)
        self.with_cp = False
        self.in_channels = make_divisible(32 * self.widen_factor, 8)

        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.layers = []

        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks, stride = layer_cfg
            out_channels = make_divisible(channel * self.widen_factor, 8)
            inverted_res_layer = self.make_layer(
                out_channels=out_channels,
                num_blocks=num_blocks,
                stride=stride,
                expand_ratio=expand_ratio)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)

        self.out_channel = 1280

        layer = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.add_module('conv2', layer)
        self.layers.append('conv2')

    def make_layer(self, out_channels, num_blocks, stride, expand_ratio):
        layers = []
        for i in range(num_blocks):
            if i >= 1:
                stride = 1
            layers.append(
                InvertedResidual(
                    self.in_channels,
                    out_channels,
                    mid_channels=int(round(self.in_channels * expand_ratio)),
                    stride=stride,
                    with_expand_conv=expand_ratio != 1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    with_cp=self.with_cp))
            self.in_channels = out_channels

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


class SSDNeck(torch.nn.Module):
    def __init__(self):
        super(SSDNeck, self).__init__()

        in_channels = (96, 1280)
        out_channels = (96, 1280, 512, 256, 256, 128)
        level_strides = (2, 2, 2, 2)
        level_paddings = (1, 1, 1, 1)
        last_kernel_size = 3

        self.extra_layers = torch.nn.ModuleList()
        extra_layer_channels = out_channels[len(in_channels):]
        second_conv = DepthwiseSeparableConvModule

        for i, (out_channel, stride, padding) in enumerate(
                zip(extra_layer_channels, level_strides, level_paddings)):
            kernel_size = last_kernel_size \
                if i == len(extra_layer_channels) - 1 else 3
            per_lvl_convs = torch.nn.Sequential(
                ConvModule(
                    out_channels[len(in_channels) - 1 + i],
                    out_channel // 2,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
                second_conv(
                    out_channel // 2,
                    out_channel,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.extra_layers.append(per_lvl_convs)

    def forward(self, inputs):
        outs = [feat for feat in inputs]
        feat = outs[-1]
        for layer in self.extra_layers:
            feat = layer(feat)
            outs.append(feat)
        return tuple(outs)


class SSDHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 80
        self.in_channels = (96, 1280, 512, 256, 256, 128)
        self.stacked_convs = 0
        self.feat_channels = 256
        self.use_depthwise = True
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.cls_out_channels = 81  # add background class
        # Usually the numbers of anchors for each level are the same
        # except SSD detectors. So it is an int in the most dense
        # heads but a list of int in SSDHead
        self.num_base_priors = [6, 6, 6, 6, 6, 6]

        self._init_layers()

    def _init_layers(self):
        self.cls_convs = torch.nn.ModuleList()
        self.reg_convs = torch.nn.ModuleList()
        conv = DepthwiseSeparableConvModule

        for channel, num_base_priors in zip(self.in_channels,
                                            self.num_base_priors):
            cls_layers = []
            reg_layers = []
            in_channel = channel
            # build stacked conv tower, not used in default ssd
            for i in range(self.stacked_convs):
                cls_layers.append(
                    conv(
                        in_channel,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                reg_layers.append(
                    conv(
                        in_channel,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                in_channel = self.feat_channels
            # SSD-Lite head
            if self.use_depthwise:
                cls_layers.append(
                    ConvModule(
                        in_channel,
                        in_channel,
                        3,
                        padding=1,
                        groups=in_channel,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                reg_layers.append(
                    ConvModule(
                        in_channel,
                        in_channel,
                        3,
                        padding=1,
                        groups=in_channel,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            cls_layers.append(
                torch.nn.Conv2d(
                    in_channel,
                    num_base_priors * self.cls_out_channels,
                    kernel_size=1 if self.use_depthwise else 3,
                    padding=0 if self.use_depthwise else 1))
            reg_layers.append(
                torch.nn.Conv2d(
                    in_channel,
                    num_base_priors * 4,
                    kernel_size=1 if self.use_depthwise else 3,
                    padding=0 if self.use_depthwise else 1))
            self.cls_convs.append(torch.nn.Sequential(*cls_layers))
            self.reg_convs.append(torch.nn.Sequential(*reg_layers))

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs,
                                            self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))
        return cls_scores, bbox_preds


if __name__ == '__main__':
    net = SSD().cuda().eval()
    net.load_state_dict(torch.load(net.ckpt_file)['state_dict'])
    net = torch.jit.trace(net, torch.empty(
        1, 3, 320, 320).cuda(), strict=False)
    net = torch.jit.freeze(net)
    torch.jit.save(net, 'ssd_net.pt')
    print(net.graph)

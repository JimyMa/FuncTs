import torch
from mmcv.cnn import ConvModule
from mmdet.models.layers import InvertedResidual
from mmdet.models.utils import make_divisible

conv_cfg = None
norm_cfg = {'type': 'BN'}
act_cfg = {'type': 'LeakyReLU', 'negative_slope': 0.1}


class YOLOV3(torch.nn.Module):
    ckpt_file = 'models/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pt'

    def __init__(self) -> None:
        super().__init__()
        self.backbone = MobileNetV2()
        self.neck = YOLOV3Neck()
        self.bbox_head = YOLOV3Head()

    def forward(self, img):
        feat = self.backbone(img)
        feats = self.neck(feat)
        pred_maps = self.bbox_head(feats)
        return pred_maps


class MobileNetV2(torch.nn.Module):
    # Parameters to build layers. 4 parameters are needed to construct a
    # layer, from left to right: expand_ratio, channel, num_blocks, stride.
    arch_settings = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2],
                     [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2],
                     [6, 320, 1, 1]]

    def __init__(self):
        super().__init__()

        self.widen_factor = 1.
        self.out_indices = (2, 4, 6)
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


class YOLOV3Neck(torch.nn.Module):
    def __init__(self):
        super(YOLOV3Neck, self).__init__()
        # assert (num_scales == len(in_channels) == len(out_channels))
        self.num_scales = 3
        self.in_channels = [320, 96, 32]
        self.out_channels = [96, 96, 96]

        # To support arbitrary scales, the code looks awful, but it works.
        # Better solution is welcomed.
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.detect1 = DetectionBlock(
            self.in_channels[0], self.out_channels[0])
        for i in range(1, self.num_scales):
            in_c, out_c = self.in_channels[i], self.out_channels[i]
            inter_c = self.out_channels[i - 1]
            self.add_module(f'conv{i}', ConvModule(inter_c, out_c, 1, **cfg))
            # in_c + out_c : High-lvl feats will be cat with low-lvl feats
            self.add_module(f'detect{i+1}',
                            DetectionBlock(in_c + out_c, out_c))

    def forward(self, feats):
        assert len(feats) == self.num_scales

        # processed from bottom (high-lvl) to top (low-lvl)
        outs = []
        out = self.detect1(feats[-1])
        outs.append(out)

        for i, x in enumerate(reversed(feats[:-1])):
            conv = getattr(self, f'conv{i+1}')
            tmp = conv(out)

            # Cat with low-lvl feats
            tmp = torch.nn.functional.interpolate(tmp, scale_factor=2)
            tmp = torch.cat((tmp, x), 1)

            detect = getattr(self, f'detect{i+2}')
            out = detect(tmp)
            outs.append(out)

        return tuple(outs)


class DetectionBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(DetectionBlock, self).__init__()
        double_out_channels = out_channels * 2

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1 = ConvModule(in_channels, out_channels, 1, **cfg)
        self.conv2 = ConvModule(
            out_channels, double_out_channels, 3, padding=1, **cfg)
        self.conv3 = ConvModule(double_out_channels, out_channels, 1, **cfg)
        self.conv4 = ConvModule(
            out_channels, double_out_channels, 3, padding=1, **cfg)
        self.conv5 = ConvModule(double_out_channels, out_channels, 1, **cfg)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.conv2(tmp)
        tmp = self.conv3(tmp)
        tmp = self.conv4(tmp)
        out = self.conv5(tmp)
        return out


class YOLOV3Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 80
        self.in_channels = [96, 96, 96]
        self.out_channels = [96, 96, 96]
        self.featmap_strides = [32, 16, 8]
        self.num_base_priors = 3
        self._init_layers()

    @property
    def num_levels(self):
        return len(self.featmap_strides)

    @property
    def num_attrib(self):
        return 5 + self.num_classes

    def _init_layers(self):
        self.convs_bridge = torch.nn.ModuleList()
        self.convs_pred = torch.nn.ModuleList()
        for i in range(self.num_levels):
            conv_bridge = ConvModule(
                self.in_channels[i],
                self.out_channels[i],
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            conv_pred = torch.nn.Conv2d(self.out_channels[i],
                                        self.num_base_priors * self.num_attrib, 1)
            self.convs_bridge.append(conv_bridge)
            self.convs_pred.append(conv_pred)

    def forward(self, feats):
        pred_maps = []
        for i in range(self.num_levels):
            x = feats[i]
            x = self.convs_bridge[i](x)
            pred_map = self.convs_pred[i](x)
            pred_maps.append(pred_map)
        return pred_maps


if __name__ == '__main__':
    net = YOLOV3().cuda().eval()
    net.load_state_dict(torch.load(net.ckpt_file)['state_dict'])
    net = torch.jit.trace(net, torch.empty(
        1, 3, 320, 320).cuda(), strict=False)
    net = torch.jit.freeze(net)
    torch.jit.save(net, 'yolov3_net.pt')
    print(net.graph)

import torch
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer


class SOLOV2(torch.nn.Module):
    ckpt_file = 'models/solov2_light_r18_fpn_3x_coco_20220511_083717-75fa355b.pt'

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ResNet()
        self.neck = FPN()
        self.mask_head = SOLOV2Head()

    def forward(self, img):
        feat = self.backbone(img)
        feats = self.neck(feat)
        out = self.mask_head(feats)
        return out


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 ):
        super().__init__()

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        out = _inner_forward(x)
        out = self.relu(out)

        return out


class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        in_channels = 3
        self.depth = 18
        self.stem_channels = 64
        self.base_channels = 64
        self.num_stages = 4
        self.strides = (1, 2, 2, 2)
        self.dilations = (1, 1, 1, 1)
        self.out_indices = (0, 1, 2, 3)
        self.avg_down = False
        self.frozen_stages = 1
        self.conv_cfg = None
        self.norm_cfg = {'type': 'BN', 'requires_grad': True}
        self.with_cp = False
        self.norm_eval = True
        self.dcn = None
        self.stage_with_dcn = (False, False, False, False)
        self.plugins = None
        self.block = BasicBlock
        self.stage_blocks = (2, 2, 2, 2)
        self.inplanes = 64

        self._make_stem_layer(in_channels, self.stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            planes = self.base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                with_cp=self.with_cp,
            )
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * self.base_channels * 2**(
            len(self.stage_blocks) - 1)

    def make_res_layer(self,
                       block,
                       inplanes,
                       planes,
                       num_blocks,
                       stride=1,
                       **kwargs):

        downsample = None
        if stride != 1 or inplanes != planes * self.block.expansion:
            downsample = []
            conv_stride = stride
            downsample.extend([
                build_conv_layer(
                    self.conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1]
            ])
            downsample = torch.nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                **kwargs))
        inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    **kwargs))
        return torch.nn.Sequential(*layers)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            stem_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, stem_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


class FPN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = [64, 128, 256, 512]
        self.out_channels = 256
        self.num_ins = 4
        self.num_outs = 5
        self.upsample_cfg = {'mode': 'nearest'}
        self.backbone_end_level = self.num_ins
        self.start_level = 0
        conv_cfg = None
        norm_cfg = None
        act_cfg = None

        self.lateral_convs = torch.nn.ModuleList()
        self.fpn_convs = torch.nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                self.in_channels[i],
                self.out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.out_channels,
                self.out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs):
        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + torch.nn.functional.interpolate(
                laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            for i in range(self.num_outs - used_backbone_levels):
                outs.append(torch.nn.functional.max_pool2d(
                    outs[-1], 1, stride=2))

        return tuple(outs)


class SOLOV2Head(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.num_levels = 5
        self.cls_out_channels = 80
        self.in_channels = 256
        self.feat_channels = 256
        self.stacked_convs = 2
        self.num_grids = [40, 36, 24, 16, 12]
        self.norm_cfg = {'type': 'GN', 'num_groups': 32, 'requires_grad': True}
        mask_out_channels = 128
        self.dynamic_conv_size = 1
        self.kernel_out_channels = \
            mask_out_channels * self.dynamic_conv_size * self.dynamic_conv_size

        mask_feature_head = {'feat_channels': 128, 'start_level': 0, 'end_level': 3, 'out_channels': 128,
                             'mask_stride': 4, 'norm_cfg': {'type': 'GN', 'num_groups': 32, 'requires_grad': True}, 'in_channels': 256}
        self.mask_feature_head = MaskFeatModule(**mask_feature_head)
        self.mask_stride = self.mask_feature_head.mask_stride

        self._init_layers()

    def _init_layers(self):
        self.cls_convs = torch.nn.ModuleList()
        self.kernel_convs = torch.nn.ModuleList()
        conv_cfg = None
        for i in range(self.stacked_convs):
            chn = self.in_channels + 2 if i == 0 else self.feat_channels
            self.kernel_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

        self.conv_cls = torch.nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)

        self.conv_kernel = torch.nn.Conv2d(
            self.feat_channels, self.kernel_out_channels, 3, padding=1)

    def forward(self, feats):
        assert len(feats) == self.num_levels
        mask_feats = self.mask_feature_head(feats)
        feats = self.resize_feats(feats)
        mlvl_kernel_preds = []
        mlvl_cls_preds = []
        for i in range(self.num_levels):
            ins_kernel_feat = feats[i]
            # ins branch
            # concat coord
            coord_feat = generate_coordinate(ins_kernel_feat.size(),
                                             ins_kernel_feat.device)
            ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)

            # kernel branch
            kernel_feat = ins_kernel_feat
            kernel_feat = torch.nn.functional.interpolate(
                kernel_feat,
                size=self.num_grids[i],
                mode='bilinear',
                align_corners=False)

            cate_feat = kernel_feat[:, :-2, :, :]

            kernel_feat = kernel_feat.contiguous()
            for i, kernel_conv in enumerate(self.kernel_convs):
                kernel_feat = kernel_conv(kernel_feat)
            kernel_pred = self.conv_kernel(kernel_feat)

            # cate branch
            cate_feat = cate_feat.contiguous()
            for i, cls_conv in enumerate(self.cls_convs):
                cate_feat = cls_conv(cate_feat)
            cate_pred = self.conv_cls(cate_feat)

            mlvl_kernel_preds.append(kernel_pred)
            mlvl_cls_preds.append(cate_pred)

        return mlvl_kernel_preds, mlvl_cls_preds, mask_feats

    def resize_feats(self, feats):
        out = []
        for i in range(len(feats)):
            if i == 0:
                out.append(
                    torch.nn.functional.interpolate(
                        feats[0],
                        size=feats[i + 1].shape[-2:],
                        mode='bilinear',
                        align_corners=False))
            elif i == len(feats) - 1:
                out.append(
                    torch.nn.functional.interpolate(
                        feats[i],
                        size=feats[i - 1].shape[-2:],
                        mode='bilinear',
                        align_corners=False))
            else:
                out.append(feats[i])
        return out


def generate_coordinate(featmap_sizes, device='cuda'):
    x_range = torch.linspace(-1, 1, featmap_sizes[-1], device=device)
    y_range = torch.linspace(-1, 1, featmap_sizes[-2], device=device)
    y, x = torch.meshgrid(y_range, x_range)
    y = y.expand([featmap_sizes[0], 1, -1, -1])
    x = x.expand([featmap_sizes[0], 1, -1, -1])
    coord_feat = torch.cat([x, y], 1)

    return coord_feat


class MaskFeatModule(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 start_level,
                 end_level,
                 out_channels,
                 mask_stride=4,
                 conv_cfg=None,
                 norm_cfg=None
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.start_level = start_level
        self.end_level = end_level
        self.mask_stride = mask_stride
        assert start_level >= 0 and end_level >= start_level
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()
        self.fp16_enabled = False

    def _init_layers(self):
        self.convs_all_levels = torch.nn.ModuleList()
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = torch.nn.Sequential()
            if i == 0:
                convs_per_level.add_module(
                    f'conv{i}',
                    ConvModule(
                        self.in_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        inplace=False))
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    if i == self.end_level:
                        chn = self.in_channels + 2
                    else:
                        chn = self.in_channels
                    convs_per_level.add_module(
                        f'conv{j}',
                        ConvModule(
                            chn,
                            self.feat_channels,
                            3,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            inplace=False))
                    convs_per_level.add_module(
                        f'upsample{j}',
                        torch.nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=False))
                    continue

                convs_per_level.add_module(
                    f'conv{j}',
                    ConvModule(
                        self.feat_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        inplace=False))
                convs_per_level.add_module(
                    f'upsample{j}',
                    torch.nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False))

            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = ConvModule(
            self.feat_channels,
            self.out_channels,
            1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)

    def forward(self, feats):
        inputs = feats[self.start_level:self.end_level + 1]
        assert len(inputs) == (self.end_level - self.start_level + 1)
        feature_add_all_level = self.convs_all_levels[0](inputs[0])
        for i in range(1, len(inputs)):
            input_p = inputs[i]
            if i == len(inputs) - 1:
                coord_feat = generate_coordinate(input_p.size(),
                                                 input_p.device)
                input_p = torch.cat([input_p, coord_feat], 1)

            # fix runtime error of "+=" inplace operation in PyTorch 1.10
            feature_add_all_level = feature_add_all_level + \
                self.convs_all_levels[i](input_p)

        feature_pred = self.conv_pred(feature_add_all_level)
        return feature_pred


if __name__ == '__main__':
    net = SOLOV2().cuda().eval()
    net.load_state_dict(torch.load(net.ckpt_file)['state_dict'])
    net = torch.jit.trace(net, torch.empty(
        1, 3, 320, 320).cuda(), strict=False)
    net = torch.jit.freeze(net)
    torch.jit.save(net, 'solov2_net.pt')
    print(net.graph)

import torch
import torch.nn.functional as F


class Config:
    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)


cfg = Config({
    'mask_proto_net': [(256, 3, {'padding': 1}), (256, 3, {'padding': 1}), (256, 3, {'padding': 1}), (None, -2, {}), (256, 3, {'padding': 1}), (32, 1, {})],
    'pred_aspect_ratios': [[[1, 0.5, 2]], [[1, 0.5, 2]], [[1, 0.5, 2]], [[1, 0.5, 2]], [[1, 0.5, 2]]],
    'pred_scales': [[24], [48], [96], [192], [384]],
    'extra_head_net': [(256, 3, {'padding': 1})],
    'head_layer_params': {'kernel_size': 3, 'padding': 1},
})


class Yolact(torch.nn.Module):
    ckpt_file = 'models/yolact_edge_mobilenetv2_54_800000.pt'

    def __init__(self):
        super().__init__()

        self.backbone = construct_backbone()
        self.training = False

        # Compute mask_dim here and add it back to the config. Make sure Yolact's constructor is called early!
        self.num_grids = 0
        self.proto_src = 0
        in_channels = 256
        # The include_last_relu=false here is because we might want to change it to another function
        self.proto_net, _ = make_net(
            in_channels, cfg.mask_proto_net, include_last_relu=False)

        self.selected_layers = [3, 4, 6]
        src_channels = self.backbone.channels

        # Some hacky rewiring to accomodate the FPN
        self.fpn = FPN([src_channels[i] for i in self.selected_layers])
        self.selected_layers = list(
            range(len(self.selected_layers) + 2))
        src_channels = [256] * len(self.selected_layers)

        self.prediction_layers = torch.nn.ModuleList()

        for idx, layer_idx in enumerate(self.selected_layers):
            # If we're sharing prediction module weights, have every module's parent be the first one
            parent = None
            if idx > 0:
                parent = self.prediction_layers[0]

            pred = PredictionModule(src_channels[layer_idx], src_channels[layer_idx],
                                    aspect_ratios=cfg.pred_aspect_ratios[idx],
                                    scales=cfg.pred_scales[idx],
                                    parent=parent,
                                    index=idx)
            self.prediction_layers.append(pred)

        self.semantic_seg_conv = torch.nn.Conv2d(
            src_channels[0], 81-1, kernel_size=1)

    def forward(self, x):
        outs = self.backbone(x)
        outs = [outs[i] for i in [3, 4, 6]]
        outs = self.fpn(outs)

        proto_x = outs[0]
        proto_out = self.proto_net(proto_x)
        proto_out = F.relu(proto_out, inplace=True)

        mlvl_bboxes, mlvl_confs, mlvl_masks = [], [], []
        for idx, pred_layer in zip(self.selected_layers, self.prediction_layers):
            pred_x = outs[idx]
            bbox, conf, mask = pred_layer(pred_x)
            mlvl_bboxes.append(bbox)
            mlvl_confs.append(conf)
            mlvl_masks.append(mask)

        return mlvl_bboxes, mlvl_confs, mlvl_masks, proto_out


def construct_backbone():
    backbone = MobileNetV2Backbone(1.0, [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [
                                   6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]], 8)

    # Add downsampling layers until we reach the number we need
    num_layers = max([3, 4, 6]) + 1

    while len(backbone.layers) < num_layers:
        backbone.add_layer()

    return backbone


class InvertedResidual(torch.nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim,
                       stride=stride, groups=hidden_dim),
            # pw-linear
            torch.nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            torch.nn.BatchNorm2d(oup),
        ])

        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBNReLU(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, activation=torch.nn.ReLU6(inplace=True)):
        padding = (kernel_size - 1) // 2
        super().__init__(
            torch.nn.Conv2d(in_planes, out_planes, kernel_size,
                            stride, padding, groups=groups, bias=False),
            torch.nn.BatchNorm2d(out_planes),
            activation
        )


class MobileNetV2Backbone(torch.nn.Module):
    def __init__(self,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=InvertedResidual):
        super(MobileNetV2Backbone, self).__init__()

        input_channel = 32
        last_channel = 1280
        self.channels = []

        self.layers = torch.nn.ModuleList()

        if inverted_residual_setting is None:
            raise ValueError("Must provide inverted_residual_setting where each element is a list "
                             "that represents the MobileNetV2 t,c,n,s values for that layer.")

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(
            input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest)
        self.layers.append(ConvBNReLU(3, input_channel, stride=2))
        self.channels.append(input_channel)

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            input_channel = self._make_layer(
                input_channel, width_mult, round_nearest, t, c, n, s, block)
            self.channels.append(input_channel)

        # building last several layers
        self.layers.append(ConvBNReLU(
            input_channel, self.last_channel, kernel_size=1))
        self.channels.append(self.last_channel)

        # These modules will be initialized by init_backbone,
        # so don't overwrite their initialization later.
        self.backbone_modules = [
            m for m in self.modules() if isinstance(m, torch.nn.Conv2d)]

    def _make_layer(self, input_channel, width_mult, round_nearest, t, c, n, s, block):
        """A layer is a combination of inverted residual blocks"""
        layers = []
        output_channel = _make_divisible(c * width_mult, round_nearest)

        for i in range(n):
            stride = s if i == 0 else 1
            layers.append(
                block(input_channel, output_channel, stride, expand_ratio=t))
            input_channel = output_channel

        self.layers.append(torch.nn.Sequential(*layers))
        return input_channel

    def forward(self, x):
        """ Returns a list of convouts for each layer. """
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        outs = []

        for idx, layer in enumerate(self.layers):
            x = layer(x)
            outs.append(x)

        return tuple(outs)

    def add_layer(self, conv_channels=1280, t=1, c=1280, n=1, s=2):
        self._make_layer(conv_channels, 1.0, 8, t, c, n, s, InvertedResidual)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Concat(torch.nn.Module):
    def __init__(self, nets, extra_params):
        super().__init__()

        self.nets = torch.ModuleList(nets)
        self.extra_params = extra_params

    def forward(self, x):
        # Concat each along the channel dimension
        return torch.cat([net(x) for net in self.nets], dim=1, **self.extra_params)


def make_net(in_channels, conf, include_last_relu=True):
    def make_layer(layer_cfg):
        nonlocal in_channels

        if isinstance(layer_cfg[0], str):
            layer_name = layer_cfg[0]

            if layer_name == 'cat':
                nets = [make_net(in_channels, x) for x in layer_cfg[1]]
                layer = Concat([net[0] for net in nets], layer_cfg[2])
                num_channels = sum([net[1] for net in nets])
        else:
            num_channels = layer_cfg[0]
            kernel_size = layer_cfg[1]

            if kernel_size > 0:
                layer = torch.nn.Conv2d(in_channels, num_channels,
                                        kernel_size, **layer_cfg[2])
            else:
                if num_channels is None:
                    layer = InterpolateModule(
                        scale_factor=-kernel_size, mode='bilinear', align_corners=False, **layer_cfg[2])
                else:
                    layer = torch.nn.ConvTranspose2d(
                        in_channels, num_channels, -kernel_size, **layer_cfg[2])

        in_channels = num_channels if num_channels is not None else in_channels

        return [layer, torch.nn.ReLU(inplace=True)]

    # Use sum to concat together all the component layer lists
    net = sum([make_layer(x) for x in conf], [])
    if not include_last_relu:
        net = net[:-1]

    return torch.nn.Sequential(*(net)), in_channels


class InterpolateModule(torch.nn.Module):
    def __init__(self, *args, **kwdargs):
        super().__init__()

        self.args = args
        self.kwdargs = kwdargs

    def forward(self, x):
        return F.interpolate(x, *self.args, **self.kwdargs)


class PredictionModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels=1024, aspect_ratios=[[1]], scales=[1], parent=None, index=0):
        super().__init__()

        self.params = [in_channels, out_channels,
                       aspect_ratios, scales, parent, index]

        self.num_classes = 81
        self.mask_dim = 32
        self.num_priors = sum(len(x) for x in aspect_ratios)
        self.parent = [parent]  # Don't include this in the state dict
        self.index = index

        if parent is None:
            self.upfeature, out_channels = make_net(
                in_channels, cfg.extra_head_net)

            self.bbox_layer = torch.nn.Conv2d(
                out_channels, self.num_priors * 4,                **cfg.head_layer_params)
            self.conf_layer = torch.nn.Conv2d(
                out_channels, self.num_priors * self.num_classes, **cfg.head_layer_params)
            self.mask_layer = torch.nn.Conv2d(
                out_channels, self.num_priors * self.mask_dim,    **cfg.head_layer_params)

            # What is this ugly lambda doing in the middle of all this clean prediction module code?
            def make_extra(num_layers):
                if num_layers == 0:
                    return lambda x: x
                else:
                    # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
                    return torch.nn.Sequential(*sum([[
                        torch.nn.Conv2d(out_channels, out_channels,
                                        kernel_size=3, padding=1),
                        torch.nn.ReLU(inplace=True)
                    ] for _ in range(num_layers)], []))

            self.bbox_extra, self.conf_extra, self.mask_extra = [
                make_extra(x) for x in (0, 0, 0)]

        self.aspect_ratios = aspect_ratios
        self.scales = scales

        self.priors = None
        self.last_conv_size = None

    def forward(self, x):
        # In case we want to use another module's layers
        src = self if self.parent[0] is None else self.parent[0]

        x = src.upfeature(x)

        bbox_x = src.bbox_extra(x)
        conf_x = src.conf_extra(x)
        mask_x = src.mask_extra(x)

        bbox = src.bbox_layer(bbox_x)
        conf = src.conf_layer(conf_x)
        mask = torch.tanh(src.mask_layer(mask_x))

        return bbox, conf, mask


class Bottleneck(torch.nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation,)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FPN(torch.nn.Module):
    __constants__ = ['interpolation_mode',
                     'num_downsample', 'use_conv_downsample']

    def __init__(self, in_channels):
        super().__init__()

        self.lat_layers = torch.nn.ModuleList([
            torch.nn.Conv2d(x, 256, kernel_size=1)
            for x in reversed(in_channels)
        ])

        # This is here for backwards compatability
        padding = 1
        self.pred_layers = torch.nn.ModuleList([
            torch.nn.Conv2d(256, 256,
                            kernel_size=3, padding=padding)
            for _ in in_channels
        ])

        if True:
            self.downsample_layers = torch.nn.ModuleList([
                torch.nn.Conv2d(256, 256,
                                kernel_size=3, padding=1, stride=2)
                for _ in range(2)
            ])

        self.interpolation_mode = 'bilinear'
        self.num_downsample = 2
        self.use_conv_downsample = True

    def forward(self, convouts):
        out = []
        x = torch.zeros(1, device=convouts[0].device)
        for i in range(len(convouts)):
            out.append(x)

        # For backward compatability, the conv layers are stored in reverse but the input and output is
        # given in the correct order. Thus, use j=-i-1 for the input and output and i for the conv layers.
        j = len(convouts)
        for lat_layer in self.lat_layers:
            j -= 1

            if j < len(convouts) - 1:
                _, _, h, w = convouts[j].size()
                x = F.interpolate(
                    x, size=(h, w), mode=self.interpolation_mode, align_corners=False)
            x = x + lat_layer(convouts[j])
            out[j] = x

        # This janky second loop is here because TorchScript.
        j = len(convouts)
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = F.relu(pred_layer(out[j]))

        # In the original paper, this takes care of P6
        if self.use_conv_downsample:
            for downsample_layer in self.downsample_layers:
                out.append(downsample_layer(out[-1]))
        else:
            for idx in range(self.num_downsample):
                # Note: this is an untested alternative to out.append(out[-1][:, :, ::2, ::2]). Thanks TorchScript.
                out.append(torch.nn.functional.max_pool2d(
                    out[-1], 1, stride=2))

        return out


if __name__ == '__main__':
    net = Yolact().cuda().eval()
    net.load_state_dict(torch.load(net.ckpt_file))
    net = torch.jit.trace(net, torch.empty(
        1, 3, 320, 320).cuda(), strict=False)
    net = torch.jit.freeze(net)
    torch.jit.save(net, 'yolact_net.pt')
    print(net.graph)

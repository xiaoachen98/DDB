# Obtained from https://github.com/lhoyer/DAFormer
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .uda_decode_head import UDADecodeHead


class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """

    def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg, act_cfg):
        super(ASPPModule, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for dilation in dilations:
            self.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
            )

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))

        return aspp_outs


@HEADS.register_module()
class DLV2AdapterHead(UDADecodeHead):
    def __init__(self, dilations=(6, 12, 18, 24), **kwargs):
        super(DLV2AdapterHead, self).__init__(**kwargs)
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels,
            self.channels // len(dilations),
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.bottleneck = ConvModule(
            self.channels,
            self.channels,
            kernel_size=3,
            dilation=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def forward(self, inputs, return_feature=False):
        x = self._transform_inputs(inputs)
        aspp_outs = self.aspp_modules(x)
        out = aspp_outs[0]
        for i in range(len(aspp_outs) - 1):
            out = torch.cat((out, aspp_outs[i + 1]), dim=1)
        out = self.bottleneck(out)
        out = self.dropout(out)
        if return_feature:
            return self.conv_seg(out), out
        # generate the second-last feature for calculating prototypes
        out = self.conv_seg(out)
        return out

# Obtained from https://github.com/lhoyer/DAFormer

import torch.nn as nn

from mmcv.cnn import ConvModule
from .uda_decode_head import UDADecodeHead
import torch
from ..builder import HEADS


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
class DLV2Head(UDADecodeHead):
    def __init__(self, dilations=(6, 12, 18, 24), **kwargs):
        assert "channels" not in kwargs
        assert "dropout_ratio" not in kwargs
        assert "norm_cfg" not in kwargs
        kwargs["channels"] = 1
        kwargs["dropout_ratio"] = 0
        kwargs["norm_cfg"] = None
        super(DLV2Head, self).__init__(**kwargs)
        del self.conv_seg
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels,
            self.num_classes,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None,
        )

    def forward(self, inputs, return_feature=False):
        x = self._transform_inputs(inputs)
        aspp_outs = self.aspp_modules(x)
        out = aspp_outs[0]
        for i in range(len(aspp_outs) - 1):
            out += aspp_outs[i + 1]
        if return_feature:
            return out, torch.cat(aspp_outs, dim=1)
        return out

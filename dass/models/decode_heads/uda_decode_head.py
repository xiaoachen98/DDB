import torch.nn as nn
from mmcv.runner import force_fp32
from mmseg.models import accuracy
from mmseg.ops import resize
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


class UDADecodeHead(BaseDecodeHead):
    def forward_train(
            self,
            inputs,
            img_metas,
            gt_semantic_seg,
            train_cfg,
            seg_weight=None,
            return_logits=False,
    ):
        """Forward function for training.
        Args:

            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.
            seg_weight (List): The weight of classes for calculating the losses

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.losses(
            seg_logits, gt_semantic_seg, seg_weight, return_logits=return_logits
        )
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, return_feature=False):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        if return_feature:
            return self.forward(inputs, return_feature)
        return self.forward(inputs)

    @force_fp32(apply_to=("seg_logit",))
    def losses(self, seg_logit, seg_label, seg_weight=None, return_logits=False):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        if return_logits:
            loss["logits"] = seg_logit
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index,
                )
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index,
                )

        loss["acc_seg"] = accuracy(seg_logit, seg_label)
        return loss

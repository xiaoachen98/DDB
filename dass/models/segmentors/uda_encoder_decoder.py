import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.models.segmentors import EncoderDecoder
from mmseg.ops import resize
from ..builder import SEGMENTORS


@SEGMENTORS.register_module()
class UDAEncoderDecoder(EncoderDecoder):
    def _decode_head_forward_train(
            self, x, img_metas, gt_semantic_seg, seg_weight=None, return_logits=False
    ):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(
            x,
            img_metas,
            gt_semantic_seg,
            self.train_cfg,
            seg_weight,
            return_logits=return_logits,
        )

        losses.update(add_prefix(loss_decode, "decode"))
        return losses

    def _decode_head_forward_test(self, x, img_metas, return_feature=False):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg,
                                                   return_feature)
        return seg_logits

    def encode_decode(self, img, img_metas, return_feature=False):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        if return_feature:
            out, feature = self._decode_head_forward_test(x, img_metas, True)
            out = resize(
                input=out,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            feature = resize(
                input=feature,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            return out, feature
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def forward_train(
            self, img, img_metas, gt_semantic_seg, seg_weight=None, return_logits=False
    ):
        """Forward function for training. Add seg_weight and return_feat

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)

        losses = dict()

        loss_decode = self._decode_head_forward_train(
            x, img_metas, gt_semantic_seg, seg_weight, return_logits=return_logits
        )
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    def inference(self, img, img_meta, rescale, return_seg_logit=False):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.
            return_seg_logit (bool): Whether return the seg logit

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ["slide", "whole"]
        ori_shape = img_meta[0]["ori_shape"]
        assert all(_["ori_shape"] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == "slide":
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]["flip"]
        if return_seg_logit:
            if flip:
                flip_direction = img_meta[0]["flip_direction"]
                assert flip_direction in ["horizontal", "vertical"]
                if flip_direction == "horizontal":
                    output = output.flip(dims=(3,))
                    seg_logit = seg_logit.flip(dims=(3,))
                elif flip_direction == "vertical":
                    output = output.flip(dims=(2,))
                    seg_logit = seg_logit.flip(dims=(2,))
            return output, seg_logit
        if flip:
            flip_direction = img_meta[0]["flip_direction"]
            assert flip_direction in ["horizontal", "vertical"]
            if flip_direction == "horizontal":
                output = output.flip(dims=(3,))
            elif flip_direction == "vertical":
                output = output.flip(dims=(2,))

        return output

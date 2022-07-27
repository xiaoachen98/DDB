from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.utils import print_log
from torch.nn.parallel.distributed import _find_tensors

from mmseg.core import add_prefix
from mmseg.models import build_segmentor
from .uda_decorator import UDADecorator, get_module
from ..utils.daformer_utils import downscale_label_ratio
from ..builder import UDA
from ..utils.module import kl_loss


@UDA.register_module()
class CKD(UDADecorator):
    def __init__(self, **cfg):
        super(CKD, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg["max_iters"]
        self.pseudo_threshold = cfg["pseudo_threshold"]

        self.stu_model = getattr(self, "model")
        self.stu_model.sub_name = "stu_model"
        delattr(self, "model")

        if cfg["teacher_model_cfg"] is None:
            self.teacher_model_cfg = deepcopy(cfg["model"])
        else:
            self.teacher_model_cfg = Config.fromfile(cfg["teacher_model_cfg"]).model
        self.cu_model = build_segmentor(deepcopy(self.teacher_model_cfg))
        self.cu_model.sub_name = "cu_model"
        self.ca_model = build_segmentor(deepcopy(self.teacher_model_cfg))
        self.ca_model.sub_name = "ca_model"

        self.cu_model_load_from = cfg["cu_model_load_from"]
        self.ca_model_load_from = cfg["ca_model_load_from"]
        self.soft_distill = cfg["soft_distill"]
        self.soft_distill_w = cfg["soft_distill_w"]
        self.temp = cfg["temp"]

        self.proto_rectify = cfg["proto_rectify"]
        self.rectify_on_prob = cfg["rectify_on_prob"]
        self.cu_proto_path = cfg["cu_proto_path"]
        self.ca_proto_path = cfg["ca_proto_path"]

        self.cu_vectors = torch.zeros([self.num_classes, 256])
        self.ca_vectors = torch.zeros([self.num_classes, 256])
        self.cu_vectors_num = torch.zeros([self.num_classes])
        self.ca_vectors_num = torch.zeros([self.num_classes])

        self.use_pl_weight = cfg["use_pl_weight"]

        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

    def calculate_mean_vector(self, feature, logit, thresh=None):
        outputs_softmax = F.softmax(logit, dim=1)
        if thresh is None:
            thresh = -1
        conf = outputs_softmax.max(dim=1, keepdim=True)[0]
        mask = conf.ge(thresh)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        outputs_argmax = self.process_label(outputs_argmax.float())
        outputs_pred = outputs_argmax

        scale_factor = F.adaptive_avg_pool2d(outputs_pred * mask, 1)
        vectors = []
        ids = []
        for n in range(feature.size()[0]):
            for t in range(self.num_classes):
                if scale_factor[n][t].item() == 0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feature[n] * outputs_pred[n][t] * mask[n]
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                vectors.append(s)
                ids.append(t)
        return vectors, ids

    def process_label(self, label):
        batch, _, w, h = label.size()
        pred1 = torch.zeros(batch, self.num_classes + 1, w, h).cuda()
        idx = torch.where(
            label < self.num_classes, label, torch.Tensor([self.num_classes]).cuda()
        )
        pred1 = pred1.scatter_(1, idx.long(), 1)
        return pred1

    def get_prototype_weight(self, feature: torch.Tensor, name="stu"):
        feat_proto_dis = self.feat_proto_distance(feature, name)
        feat_nearest_proto_dis, _ = feat_proto_dis.min(dim=1, keepdim=True)
        feat_proto_dis = feat_proto_dis - feat_nearest_proto_dis
        weight = F.softmax(-feat_proto_dis, dim=1)
        return weight

    def feat_proto_distance(self, feature: torch.Tensor, name="stu"):
        n, _, h, w = feature.shape
        feat_proto_distance = -torch.ones((n, self.num_classes, h, w)).cuda()
        for i in range(self.num_classes):
            feat_proto_distance[:, i, :, :] = torch.norm(
                getattr(self, f"{name}_vectors")[i].reshape(-1, 1, 1).expand(-1, h, w)
                - feature,
                2,
                dim=1,
            )
        return feat_proto_distance

    def _init_sub_model_weights(self):
        cu_ckpt = torch.load(self.cu_model_load_from, map_location="cpu")
        ca_ckpt = torch.load(self.ca_model_load_from, map_location="cpu")
        self._load_checkpoint("cu", cu_ckpt)
        self._load_checkpoint("ca", ca_ckpt)
        print_log(
            f"Load checkpoint for ca_model from {self.ca_model_load_from}", "mmseg"
        )
        print_log(
            f"Load checkpoint for cu_model from {self.cu_model_load_from}", "mmseg"
        )
        for cu_param, ca_param in zip(
                self.get_model("cu").parameters(), self.get_model("ca").parameters()
        ):
            cu_param.detach_(), ca_param.detach_()
        if self.proto_rectify:
            self.cu_vectors = torch.load(self.cu_proto_path, map_location="cpu").cuda()
            self.ca_vectors = torch.load(self.ca_proto_path, map_location="cpu").cuda()

    def _load_checkpoint(self, name, checkpoint):
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        if name != "stu":
            for k, v in state_dict.items():
                if "imnet" not in k:
                    if "ema" not in k:
                        new_state_dict[k.replace("model.", "")] = v
        else:
            for k, v in state_dict.items():
                if "stu" in k:
                    new_state_dict[k.replace("stu_model.", "")] = v
        self.get_model(name).load_state_dict(new_state_dict, strict=True)

    def train_step(self, data_batch, optimizer, ddp_reducer=None, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        self.ddp_reducer = ddp_reducer  # store ddp reducer
        optimizer["stu_model"].zero_grad()
        log_vars = self(**data_batch)
        optimizer["stu_model"].step()

        log_vars.pop("loss", None)  # remove the unnecessary 'loss'
        outputs = dict(log_vars=log_vars, num_samples=len(data_batch["img_metas"]))
        self.ddp_reducer = None  # drop ddp reducer
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            self.get_model('imnet').eval()
            feat_imnet = self.get_model('imnet').extract_feat(img)
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def forward_train(
            self, img, img_metas, gt_semantic_seg, target_img, target_img_metas, bridging
    ):
        """Forward function for training.

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
        log_vars = {}
        dev = img.device
        stu_tgt_img = bridging["img"]

        # Init the teacher model
        if self.local_iter == 0:
            self._init_sub_model_weights()

        # Set the teacher model to eval mode
        self.get_model("cu").eval()
        self.get_model("ca").eval()

        # Train the student model on the source domain
        stu_src_losses = self.get_model("stu").forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True
        )
        stu_src_feat = stu_src_losses.pop('features')
        stu_src_losses = add_prefix(stu_src_losses, "stu_src")
        stu_src_loss, stu_src_log_vars = self._parse_losses(stu_src_losses)
        log_vars.update(stu_src_log_vars)
        if getattr(self, "ddp_reducer", None):
            self.ddp_reducer.prepare_for_backward(_find_tensors(stu_src_loss))
        stu_src_loss.backward(retain_graph=self.enable_fdist)

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      stu_src_feat)
            if getattr(self, "ddp_reducer", None):
                self.ddp_reducer.prepare_for_backward(_find_tensors(feat_loss))
            feat_loss.backward()
            log_vars.update(add_prefix(feat_log, 'src'))

        # Train the student model on the target domain
        if not self.soft_distill:
            with torch.no_grad():
                if self.proto_rectify:
                    cu_tgt_logits, cu_tgt_feature = self.encode_decode(
                        target_img, target_img_metas, "cu", return_feature=True
                    )
                    ca_tgt_logits, ca_tgt_feature = self.encode_decode(
                        target_img, target_img_metas, "ca", return_feature=True
                    )
                    cu_proto_weights = self.get_prototype_weight(
                        cu_tgt_feature, name="cu"
                    )
                    ca_proto_weights = self.get_prototype_weight(
                        ca_tgt_feature, name="ca"
                    )
                    if self.rectify_on_prob:
                        cu_tgt_softmax = torch.softmax(cu_tgt_logits, dim=1)
                        ca_tgt_softmax = torch.softmax(ca_tgt_logits, dim=1)
                        _, cu_tgt_pl = torch.max(cu_tgt_softmax, dim=1)
                        _, ca_tgt_pl = torch.max(ca_tgt_softmax, dim=1)

                        cu_tgt_softmax = cu_proto_weights * cu_tgt_softmax
                        ca_tgt_softmax = ca_proto_weights * ca_tgt_softmax
                        _, cu_rectify_tgt_pl = torch.max(cu_tgt_softmax, dim=1)
                        _, ca_rectify_tgt_pl = torch.max(ca_tgt_softmax, dim=1)
                    else:
                        _, cu_tgt_pl = torch.max(cu_tgt_logits, dim=1)
                        _, ca_tgt_pl = torch.max(ca_tgt_logits, dim=1)
                        cu_tgt_logits = cu_proto_weights * cu_tgt_logits
                        ca_tgt_logits = ca_proto_weights * ca_tgt_logits
                        cu_tgt_softmax = torch.softmax(cu_tgt_logits, dim=1)
                        ca_tgt_softmax = torch.softmax(ca_tgt_logits, dim=1)
                        _, cu_rectify_tgt_pl = torch.max(cu_tgt_logits, dim=1)
                        _, ca_rectify_tgt_pl = torch.max(ca_tgt_logits, dim=1)
                else:
                    cu_tgt_logits = self.encode_decode(
                        target_img, target_img_metas, "cu"
                    )
                    ca_tgt_logits = self.encode_decode(
                        target_img, target_img_metas, "ca"
                    )
                    cu_tgt_softmax = torch.softmax(cu_tgt_logits, dim=1)
                    ca_tgt_softmax = torch.softmax(ca_tgt_logits, dim=1)
                    _, cu_tgt_pl = torch.max(cu_tgt_softmax, dim=1)
                    _, ca_tgt_pl = torch.max(ca_tgt_softmax, dim=1)
                ensemble_tgt_softmax = (
                        cu_tgt_softmax.detach() * 0.5 + ca_tgt_softmax.detach() * 0.5
                )
                tgt_pl_prob, tgt_pseudo_label = torch.max(ensemble_tgt_softmax, dim=1)
                if self.use_pl_weight:
                    # Generate the pseudo label and weight map for ce loss
                    pl_large_p = tgt_pl_prob.ge(self.pseudo_threshold).long() == 1
                    pl_size = np.size(np.array(tgt_pseudo_label.cpu()))
                    pseudo_weight = torch.sum(pl_large_p).item() / pl_size
                else:
                    pseudo_weight = torch.FloatTensor([1.0]).cuda()
                pseudo_weight = pseudo_weight * torch.ones(tgt_pl_prob.shape, device=dev)

            stu_tgt_losses = self.get_model("stu").forward_train(
                stu_tgt_img,
                target_img_metas,
                tgt_pseudo_label.unsqueeze(1),
                pseudo_weight,
                return_logits=self.soft_distill,
            )
            stu_tgt_losses = add_prefix(stu_tgt_losses, "stu_tgt")
            stu_tgt_loss, stu_tgt_log_vars = self._parse_losses(stu_tgt_losses)
        else:
            stu_tgt_logits = self.encode_decode(stu_tgt_img, target_img_metas, "stu")
            stu_tgt_softmax = torch.log_softmax(stu_tgt_logits / self.temp, dim=1)
            with torch.no_grad():
                cu_tgt_logits, cu_tgt_feature = self.encode_decode(
                    target_img, target_img_metas, "cu", return_feature=True
                )
                ca_tgt_logits, ca_tgt_feature = self.encode_decode(
                    target_img, target_img_metas, "ca", return_feature=True
                )
                if self.proto_rectify:
                    cu_proto_weights = self.get_prototype_weight(
                        cu_tgt_feature, name="cu"
                    )
                    ca_proto_weights = self.get_prototype_weight(
                        ca_tgt_feature, name="ca"
                    )
                    cu_tgt_logits = cu_proto_weights * cu_tgt_logits
                    ca_tgt_logits = ca_proto_weights * ca_tgt_logits

            ensemble_tgt_softmax = torch.softmax(
                (cu_tgt_logits + ca_tgt_logits) / 2 / self.temp, dim=1
            )
            stu_tgt_losses = (
                    kl_loss(stu_tgt_softmax, ensemble_tgt_softmax.detach())
                    * self.soft_distill_w
                    * self.temp ** 2
            )

            stu_tgt_loss, stu_tgt_log_vars = self._parse_losses(
                {"stu_tgt.loss_distill": stu_tgt_losses}
            )

        log_vars.update(stu_tgt_log_vars)
        if getattr(self, "ddp_reducer", None):
            self.ddp_reducer.prepare_for_backward(_find_tensors(stu_tgt_loss))
        stu_tgt_loss.backward()
        self.local_iter += 1

        return log_vars

    def get_model(self, name="stu"):
        return get_module(self.__getattr__(name + "_model"))

    def extract_feat(self, img, name="stu"):
        """Extract features from images."""
        return self.get_model(name).extract_feat(img)

    def encode_decode(self, img, img_metas, name="stu", **kwargs):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        return self.get_model(name).encode_decode(img, img_metas, **kwargs)

    def inference(self, img, img_meta, rescale, name="stu", return_seg_logit=False):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.
            name (str): the name of sub model selected for the inference
            return_seg_logit (bool): Whether return the seg logit for
                sub model ensemble

        Returns:
            Tensor: The output segmentation map.
        """
        return self.get_model(name).inference(img, img_meta, rescale, return_seg_logit)

    def simple_test(self, img, img_meta, rescale=True, name="stu"):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale, name)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True, name="stu"):
        """Test with augmentations.
        Only rescale=True is supported.
        """
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs, img_metas, rescale, name)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

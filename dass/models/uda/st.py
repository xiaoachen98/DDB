from collections import OrderedDict
from copy import deepcopy

import matplotlib as mpl
import numpy as np
import torch
import math
from mmcv.utils import print_log
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd
from torch.nn.parallel.distributed import _find_tensors

from mmseg.core import add_prefix
from mmseg.models import build_segmentor
from .uda_decorator import UDADecorator, get_module
from ..builder import UDA
from ..utils.daformer_utils import downscale_label_ratio

mpl.use("Agg")  # To solve the main thread is not in main loop error


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(), model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class ST(UDADecorator):
    def __init__(self, **cfg):
        super(ST, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg["max_iters"]
        self.alpha = cfg["alpha"]
        self.pseudo_threshold = cfg["pseudo_threshold"]
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']

        ema_cfg = deepcopy(cfg["model"])
        self.ema_model = build_segmentor(ema_cfg)

        self.distilled_model_path = cfg["distilled_model_path"]

        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

    def get_model(self, name="model"):
        if "model" not in name:
            name = "_".join([name, "model"])
        return get_module(getattr(self, name))

    def _init_distill_weights(self):
        distill_ckpt = torch.load(self.distilled_model_path, map_location="cpu")
        self._load_checkpoint("model", distill_ckpt)
        print_log(f"Load checkpoint from {self.distilled_model_path}", "mmseg")

    def _load_checkpoint(self, name, checkpoint):
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "stu" in k:
                new_state_dict[k.replace("stu_model.", "")] = v
        self.get_model(name).load_state_dict(new_state_dict, strict=True)

    def _init_ema_weights(self):
        for param in self.get_model("ema").parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_model("ema").parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(
                self.get_model("ema").parameters(), self.get_model().parameters()
        ):
            if not param.data.shape:  # scalar tensor
                ema_param.data = (
                        alpha_teacher * ema_param.data + (1 - alpha_teacher) * param.data
                )
            else:
                ema_param.data[:] = (
                        alpha_teacher * ema_param[:].data[:]
                        + (1 - alpha_teacher) * param[:].data[:]
                )

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
        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

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
        # Init/update ema model
        if self.local_iter == 0:
            if self.distilled_model_path is not None:
                self._init_distill_weights()
            self._init_ema_weights()
            assert _params_equal(self.get_model("ema"), self.get_model())
        else:
            self._update_ema(self.local_iter)

        # Train on source images
        clean_losses = self.get_model().forward_train(img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        if getattr(self, "ddp_reducer", None):
            self.ddp_reducer.prepare_for_backward(_find_tensors(clean_loss))
        clean_loss.backward(retain_graph=self.enable_fdist)
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            if getattr(self, "ddp_reducer", None):
                self.ddp_reducer.prepare_for_backward(_find_tensors(feat_loss))
            feat_loss.backward()
            log_vars.update(add_prefix(feat_log, 'src'))
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')

        # Generate pseudo-label
        for m in self.get_model("ema").modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        ema_logits = self.get_model("ema").encode_decode(
            target_img, target_img_metas
        )
        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1

        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(pseudo_prob.shape, device=dev)
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones(pseudo_weight.shape, device=dev)

        # Apply mixing
        mixed_img, mix_mask = bridging["img"], bridging["mask"]
        mixed_lbl = mix_mask * gt_semantic_seg + (1 - mix_mask) * pseudo_label.unsqueeze(1)
        mixed_weight = (mix_mask.squeeze(1) * gt_pixel_weight + (1 - mix_mask.squeeze(1)) * pseudo_weight)

        # Train on mixed images
        mix_losses = self.get_model().forward_train(
            mixed_img, img_metas, mixed_lbl, mixed_weight
        )
        mix_losses = add_prefix(mix_losses, "mix")
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mix_log_vars)
        if getattr(self, "ddp_reducer", None):
            self.ddp_reducer.prepare_for_backward(_find_tensors(mix_loss))
        mix_loss.backward()
        self.local_iter += 1

        return log_vars

import json
import os.path as osp
import random

import kornia
from mmcv import print_log
import numpy as np
import torch
import torch.nn as nn
from mmcv.parallel import DataContainer as DC

from mmseg.datasets import CityscapesDataset
from .utils import get_rcs_class_probs
from .builder import DATASETS
from ..models.utils.dacs_transforms import denorm_, renorm_


@DATASETS.register_module()
class STDataset(object):
    def __init__(self, source, target, cfg):
        self.cfg = cfg
        self.source = source
        self.target = target
        if hasattr(target, "ignore_index"):
            self.ignore_index = target.ignore_index
        else:
            self.ignore_index = source.ignore_index
        self.CLASSES = target.CLASSES
        self.PALETTE = target.PALETTE
        assert target.CLASSES == source.CLASSES
        assert target.PALETTE == source.PALETTE

        self.post_pmd = cfg["post_pmd"] or False
        self.post_blur = cfg["post_blur"] or False
        self.img_mean = cfg.img_norm_cfg["mean"]
        self.img_std = cfg.img_norm_cfg["std"]
        self.mask = cfg["mask"]
        assert self.mask in [
            "class",
            "cut",
            "zero",
        ], f"{self.mask} mask mode not support yet"

        rcs_cfg = cfg.get('rare_class_sampling')
        self.rcs_enabled = rcs_cfg is not None
        if self.rcs_enabled:
            self.rcs_class_temp = rcs_cfg['class_temp']
            self.rcs_min_crop_ratio = rcs_cfg['min_crop_ratio']
            self.rcs_min_pixels = rcs_cfg['min_pixels']

            self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(
                cfg['source']['data_root'], self.rcs_class_temp)
            print_log(f'RCS Classes: {self.rcs_classes}', 'mmseg')
            print_log(f'RCS ClassProb: {self.rcs_classprob}', 'mmseg')

            with open(
                    osp.join(cfg['source']['data_root'],
                             'samples_with_class.json'), 'r') as of:
                samples_with_class_and_n = json.load(of)
            samples_with_class_and_n = {
                int(k): v
                for k, v in samples_with_class_and_n.items()
                if int(k) in self.rcs_classes
            }
            self.samples_with_class = {}
            for c in self.rcs_classes:
                self.samples_with_class[c] = []
                for file, pixels in samples_with_class_and_n[c]:
                    if pixels > self.rcs_min_pixels:
                        self.samples_with_class[c].append(file.split('/')[-1])
                assert len(self.samples_with_class[c]) > 0
            self.file_to_idx = {}
            for i, dic in enumerate(self.source.img_infos):
                file = dic['ann']['seg_map']
                if isinstance(self.source, CityscapesDataset):
                    file = file.split('/')[-1]
                self.file_to_idx[file] = i

    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        f1 = np.random.choice(self.samples_with_class[c])
        i1 = self.file_to_idx[f1]
        s1 = self.source[i1]
        if self.rcs_min_crop_ratio > 0:
            for j in range(10):
                n_class = torch.sum(s1['gt_semantic_seg'].data == c)
                # mmcv.print_log(f'{j}: {n_class}', 'mmseg')
                if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
                    break
                # Sample a new random crop from source image i1.
                # Please note, that self.source.__getitem__(idx) applies the
                # preprocessing pipeline to the loaded image, which includes
                # RandomCrop, and results in a new crop of the image.
                s1 = self.source[i1]
        return s1

    def __getitem__(self, idx):
        target_idx = np.random.choice(range(len(self.target)))
        tgt = self.target[target_idx]
        tgt_img = tgt["img"].data

        if self.rcs_enabled:
            src = self.get_rare_class_sample()
        else:
            src = self.source[idx % len(self.source)]
        src_img, src_gt = src["img"].data, src["gt_semantic_seg"].data
        bridging = self.get_bridging(src_img, src_gt, tgt_img, mode=self.mask)

        return {
            **src,
            "target_img_metas": tgt["img_metas"],
            "target_img": tgt["img"],
            "bridging": bridging,
        }

    def __len__(self):
        return max(len(self.source), len(self.target))

    def get_bridging(self, src_img, src_gt, tgt_img, mode="class"):
        if mode == "zero":
            _, h, w = src_img.shape
            mask = torch.zeros([1, h, w], dtype=torch.long)
        else:
            mask = self.get_mask(src_gt, src_img.shape, mode)
        brg_img = mask * src_img + (1 - mask) * tgt_img
        if self.post_pmd or self.post_blur:
            brg_img = self.post_process(brg_img)
        brg = dict(img=DC(brg_img, stack=True), mask=DC(mask, stack=True))
        return brg

    def get_mask(self, src_gt, img_shape, mode="class"):
        if mode == "class":
            mask = self.get_class_mask(src_gt)
        else:
            mask = self.get_cut_mask(img_shape)
        return mask

    @staticmethod
    def get_class_mask(s_gt):
        classes = torch.unique(s_gt)
        num_classes = classes.shape[0]
        class_choice = np.random.choice(
            num_classes, int((num_classes + num_classes % 2) / 2), replace=False
        )
        classes = classes[torch.Tensor(class_choice).long()]
        label, classes = torch.broadcast_tensors(
            s_gt, classes.unsqueeze(1).unsqueeze(2)
        )
        mask = label.eq(classes).sum(0, keepdims=True)
        return mask

    @staticmethod
    def get_cut_mask(img_shape, cut_mask_props=0.4):
        _, h, w = img_shape
        y_props = np.exp(
            np.random.uniform(low=0.0, high=1, size=(1,)) * np.log(cut_mask_props)
        )
        x_props = (
                cut_mask_props / y_props
        )  # the cut_mask_props aims to control the mask area
        sizes = np.round(
            np.stack([y_props, x_props], axis=1) * np.array((h, w))[None, :]
        )

        positions = np.round(
            (np.array((h, w)) - sizes)
            * np.random.uniform(low=0.0, high=1.0, size=sizes.shape)
        )
        rectangle = np.append(positions, positions + sizes, axis=1)

        mask = torch.zeros((1, h, w)).long()
        y0, x0, y1, x1 = rectangle[0]
        mask[0, int(y0): int(y1), int(x0): int(x1)] = 1
        return mask

    @staticmethod
    def color_jitter(data, mean, std, s=0.2, p=0.2):
        # s is the strength of colorjitter
        if random.uniform(0, 1) > p:
            if isinstance(s, dict):
                seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
            else:
                seq = nn.Sequential(
                    kornia.augmentation.ColorJitter(
                        brightness=s, contrast=s, saturation=s, hue=s
                    )
                )
            denorm_(data, mean, std)
            data = seq(data).squeeze(0)
            renorm_(data, mean, std)
        return data

    @staticmethod
    def gaussian_blur(data=None, p=0.5):
        if random.uniform(0, 1) > p:
            data.unsqueeze_(0)
            sigma = np.random.uniform(0.15, 1.15)
            kernel_size_y = int(
                np.floor(
                    np.ceil(0.1 * data.shape[1])
                    - 0.5
                    + np.ceil(0.1 * data.shape[1]) % 2
                )
            )
            kernel_size_x = int(
                np.floor(
                    np.ceil(0.1 * data.shape[2])
                    - 0.5
                    + np.ceil(0.1 * data.shape[2]) % 2
                )
            )
            kernel_size = (kernel_size_y, kernel_size_x)
            seq = nn.Sequential(
                kornia.filters.GaussianBlur2d(
                    kernel_size=kernel_size, sigma=(sigma, sigma)
                )
            )
            data = seq(data).squeeze(0)
        return data

    def post_process(self, img):
        mean = torch.as_tensor(self.img_mean).view(3, 1, 1)
        std = torch.as_tensor(self.img_std).view(3, 1, 1)
        if self.post_pmd:
            img = self.color_jitter(img, mean, std)
        if self.post_blur:
            img = self.gaussian_blur(img)
        return img

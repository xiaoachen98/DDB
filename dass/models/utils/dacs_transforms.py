# Obtained from: https://github.com/vikolss/DACS
import random

import kornia
import numpy as np
import torch
import torch.nn as nn


def strong_transform(param, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = one_mix(mask=param['mix'], data=data, target=target)
    data, target = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data,
        target=target)
    data, target = gaussian_blur(blur=param['blur'], data=data, target=target)
    return data, target


def get_mean_std(img_metas, dev):
    mean = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['mean'], device=dev)
        for i in range(len(img_metas))
    ]
    mean = torch.stack(mean).view(-1, 3, 1, 1)
    std = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['std'], device=dev)
        for i in range(len(img_metas))
    ]
    std = torch.stack(std).view(-1, 3, 1, 1)
    return mean, std


def denorm(img, mean, std):
    return img.mul(std).add(mean) / 255.0


def denorm_(img, mean, std):
    img.mul_(std).add_(mean).div_(255.0)


def renorm_(img, mean, std):
    img.mul_(255.0).sub_(mean).div_(std)


def color_jitter(color_jitter, mean, std, data=None, target=None, s=.25, p=.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                denorm_(data, mean, std)
                data = seq(data)
                renorm_(data, mean, std)
    return data, target


def gaussian_blur(blur, data=None, target=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2]) - 0.5 +
                        np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3]) - 0.5 +
                        np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target


def get_class_masks(labels, mask_wo_ignore=False, rcs_classes=None, rcs_classesprob=None, temp=0.01):
    class_masks = []
    for label in labels:
        classes = torch.unique(labels)
        nclasses = classes.shape[0]
        if rcs_classes is None:
            class_choice = np.random.choice(
                nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
            classes = classes[torch.Tensor(class_choice).long()]
            if mask_wo_ignore:
                classes = classes[classes != 255]
        else:
            num_classes2choice = int((nclasses + nclasses % 2) / 2)
            classes = classes[classes != 255]
            new_classes = []
            new_classesprob = []

            for i, c in enumerate(rcs_classes):
                if c in list(classes.cpu().numpy()):
                    new_classes.append(c)
                    new_classesprob.append(rcs_classesprob[i])
            new_classesprob = torch.tensor(list(new_classesprob))
            new_classesprob = torch.softmax(new_classesprob / temp, dim=-1)
            class_choice = np.random.choice(new_classes, num_classes2choice, p=new_classesprob.numpy(), replace=False)
            classes = torch.Tensor(class_choice).long().cuda()

        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
    return class_masks


def generate_class_mask(label, classes):
    label, classes = torch.broadcast_tensors(label,
                                             classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask


def get_cut_masks(img_shape, random_aspect_ratio=True, within_bounds=True, mask_props=0.4):
    n, _, h, w = img_shape
    if random_aspect_ratio:
        y_props = np.exp(np.random.uniform(low=0.0, high=1, size=(n, 1)) * np.log(mask_props))
        x_props = mask_props / y_props
    else:
        y_props = x_props = np.sqrt(mask_props)

    sizes = np.round(np.stack([y_props, x_props], axis=2) * np.array((h, w))[None, None, :])
    if within_bounds:
        positions = np.round(
            (np.array((h, w)) - sizes) * np.random.uniform(low=0.0, high=1.0, size=sizes.shape))
        rectangles = np.append(positions, positions + sizes, axis=2)
    else:
        centres = np.round(np.array((h, w)) * np.random.uniform(low=0.0, high=1.0, size=sizes.shape))
        rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=2)

    masks = []
    mask = torch.zeros((n, 1, h, w)).long().cuda()
    for i, sample_rectangles in enumerate(rectangles):
        y0, x0, y1, x1 = sample_rectangles[0]
        mask[i, 0, int(y0):int(y1), int(x0):int(x1)] = 1
        masks.append(mask[i].unsqueeze(0))
    return masks


def get_masks(img_shape, labels, mask_type='class', cut_mask_props=0.4):
    if mask_type == 'class':
        return get_class_masks(labels)
    elif mask_type == 'cut':
        return get_cut_masks(img_shape, mask_props=cut_mask_props)
    else:
        raise ValueError


def one_mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] +
                (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] +
                  (1 - stackedMask0) * target[1]).unsqueeze(0)
    return data, target

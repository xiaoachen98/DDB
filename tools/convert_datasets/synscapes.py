import argparse
import os.path as osp

import cityscapesscripts.helpers.labels as CSLabels
import mmcv
import numpy as np
from PIL import Image

palette = np.zeros((len(CSLabels.id2label), 3), dtype=np.uint8)
for label_id, label in CSLabels.id2label.items():
    palette[label_id] = label.color
id_to_trainid = {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18
}


def convert_to_train_id(file):
    # re-assign labels to match the format of Cityscapes
    pil_label = Image.open(file)
    label = np.asarray(pil_label)

    label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
    sample_class_stats = {}
    for k, v in id_to_trainid.items():
        k_mask = label == k
        label_copy[k_mask] = v
    new_file = file.replace('.png', '_gt_labelTrainIds.png')
    color_file = file.replace('.png', '_gt_color.png')
    assert file != new_file != color_file
    # save labelId and color format gt
    Image.fromarray(label_copy, mode='L').save(new_file)
    color_label = Image.fromarray(label, mode='P')
    color_label.putpalette(palette)
    color_label.save(color_file)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Synscapes annotations to TrainIds and Color')
    parser.add_argument('syns_path', help='synscapes data path')
    parser.add_argument('--gt-dir', default='img/class', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=8, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    syns_path = args.syns_path
    out_dir = args.out_dir if args.out_dir else syns_path
    mmcv.mkdir_or_exist(out_dir)

    gt_dir = osp.join(syns_path, args.gt_dir)

    poly_files = []
    for poly in mmcv.scandir(
            gt_dir, suffix=tuple(f'{i}.png' for i in range(10)),
            recursive=True):
        poly_file = osp.join(gt_dir, poly)
        poly_files.append(poly_file)
    poly_files = sorted(poly_files)

    if args.nproc > 1:
        mmcv.track_parallel_progress(
            convert_to_train_id, poly_files, args.nproc)
    else:
        mmcv.track_progress(convert_to_train_id,
                            poly_files)


if __name__ == '__main__':
    main()

from mmseg.datasets import DATASETS  # # Keep the same registry as mmseg
from mmseg.datasets import build_dataset
import copy

DATASETS = DATASETS


def _uda_concat_dataset(cfg, default_args=None):
    """Build :obj:`ConcatDataset by."""
    from .uda_concat import UDAConcatDataset

    img_dir = cfg["img_dir"]
    ann_dir = cfg.get("ann_dir", None)
    split = cfg.get("split", None)
    # pop 'separate_eval' since it is not a valid key for common datasets.
    separate_eval = cfg.pop("separate_eval", True)
    num_img_dir = len(img_dir) if isinstance(img_dir, (list, tuple)) else 1
    if ann_dir is not None:
        num_ann_dir = len(ann_dir) if isinstance(ann_dir, (list, tuple)) else 1
    else:
        num_ann_dir = 0
    if split is not None:
        num_split = len(split) if isinstance(split, (list, tuple)) else 1
    else:
        num_split = 0
    if num_img_dir > 1:
        assert num_img_dir == num_ann_dir or num_ann_dir == 0
        assert num_img_dir == num_split or num_split == 0
    else:
        assert num_split == num_ann_dir or num_ann_dir <= 1
    num_dset = max(num_split, num_img_dir)

    datasets = []
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        if isinstance(img_dir, (list, tuple)):
            data_cfg["img_dir"] = img_dir[i]
        if isinstance(ann_dir, (list, tuple)):
            data_cfg["ann_dir"] = ann_dir[i]
        if isinstance(split, (list, tuple)):
            data_cfg["split"] = split[i]
        datasets.append(build_uda_dataset(data_cfg, default_args))

    return UDAConcatDataset(datasets, separate_eval)


def build_uda_dataset(cfg, default_args=None):
    """Build datasets."""
    from .uda_concat import UDAConcatDataset
    from .st_dataset import STDataset
    from .uda_dataset import UDADataset

    if isinstance(cfg, (list, tuple)):
        dataset = UDAConcatDataset([build_uda_dataset(c, default_args) for c in cfg])
    elif cfg["type"] == "STDataset":
        dataset = STDataset(
            source=build_uda_dataset(cfg["source"], default_args),
            target=build_uda_dataset(cfg["target"], default_args),
            cfg=cfg,
        )
    elif cfg["type"] == "UDADataset":
        dataset = UDADataset(
            source=build_uda_dataset(cfg["source"], default_args),
            target=build_uda_dataset(cfg["target"], default_args),
            cfg=cfg,
        )
    elif isinstance(cfg.get("img_dir"), (list, tuple)) or isinstance(
            cfg.get("split", None), (list, tuple)
    ):
        dataset = _uda_concat_dataset(cfg, default_args)
    else:
        dataset = build_dataset(cfg, default_args)

    return dataset

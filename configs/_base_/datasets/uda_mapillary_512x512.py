mapillary_type = "MapillaryDataset"
mapillary_root = "data/mapillary/"
mapillary_img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
mapillary_crop_size = (512, 512)
mapillary_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=(1024, 512), keep_ratio=True, min_size=512),
    dict(type="RandomCrop", crop_size=mapillary_crop_size),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Normalize", **mapillary_img_norm_cfg),
    dict(type="Pad", size=mapillary_crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
mapillary_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1024, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **mapillary_img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
train_mapillary = dict(
    type=mapillary_type,
    data_root=mapillary_root,
    img_dir="training/images",
    ann_dir="cityscapes_trainIdLabel/train/label",
    pipeline=mapillary_train_pipeline,
)
val_mapillary = dict(
    type=mapillary_type,
    data_root=mapillary_root,
    img_dir="half/val_img",
    ann_dir="half/val_label",
    pipeline=mapillary_test_pipeline,
)

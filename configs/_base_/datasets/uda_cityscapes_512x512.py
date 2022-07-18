cityscapes_type = "CityscapesDataset"
cityscapes_root = "data/cityscapes/"
cityscapes_img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
cityscapes_crop_size = (512, 512)
cityscapes_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=(1024, 512)),
    dict(type="RandomCrop", crop_size=cityscapes_crop_size),  # cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Normalize", **cityscapes_img_norm_cfg),
    dict(type="Pad", size=cityscapes_crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
cityscapes_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1024, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **cityscapes_img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
train_cityscapes = dict(
    type=cityscapes_type,
    data_root=cityscapes_root,
    img_dir="leftImg8bit/train",
    ann_dir="gtFine/train",
    pipeline=cityscapes_train_pipeline,
)
val_cityscapes = dict(
    type=cityscapes_type,
    data_root=cityscapes_root,
    img_dir="leftImg8bit/val",
    ann_dir="gtFine/val",
    pipeline=cityscapes_test_pipeline,
)

dataset_type = "MapillaryDataset"
dataset_root = "data/mapillary/"
dataset_train_img_dir = "training/images"
dataset_test_img_dir = "half/val_img"
dataset_train_ann_dir = "cityscapes_trainIdLabel/train/label"
dataset_test_ann_dir = "half/val_label"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (512, 512)
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1024, 512),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    test=dict(
        type=dataset_type,
        data_root=dataset_root,
        img_dir=dataset_test_img_dir,
        ann_dir=dataset_test_ann_dir,
        pipeline=test_pipeline,
    ),
)

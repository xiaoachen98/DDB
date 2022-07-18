gtav_img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
gtav_crop_size = (512, 512)
gtav_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=(1280, 720)),
    dict(type="RandomCrop", crop_size=gtav_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    # dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **gtav_img_norm_cfg),
    dict(type="Pad", size=gtav_crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
train_gtav = dict(
    type="GTADataset",
    data_root="data/gta/",
    img_dir="images",
    ann_dir="labels",
    pipeline=gtav_train_pipeline,
)

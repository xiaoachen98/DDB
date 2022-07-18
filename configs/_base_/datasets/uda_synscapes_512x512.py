synscapes_img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
synscapes_crop_size = (512, 512)
synscapes_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=(1280, 720)),
    dict(type="RandomCrop", crop_size=synscapes_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    # dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **synscapes_img_norm_cfg),
    dict(type="Pad", size=synscapes_crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
train_synscapes = dict(
    type="SynscapesDataset",
    data_root="data/synscapes/",
    img_dir="img/rgb",
    ann_dir="img/class",
    pipeline=synscapes_train_pipeline,
)

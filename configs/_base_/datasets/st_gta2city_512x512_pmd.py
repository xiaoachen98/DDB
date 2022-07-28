_base_ = [
    "./uda_gta_512x512_pmd.py",
    "./uda_cityscapes_512x512.py",
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type="STDataset",
        source={{_base_.train_gtav}},
        target={{_base_.train_cityscapes}},
        post_pmd=True,
        post_blur=True,
        mask="zero",
        img_norm_cfg=img_norm_cfg,
    ),
    val={{_base_.val_cityscapes}},
    test={{_base_.val_cityscapes}},
)

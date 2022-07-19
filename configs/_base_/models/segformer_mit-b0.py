# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='UDAEncoderDecoder',
    pretrained=None,
    backbone=dict(type='mit_b0',
                  pretrained='pretrained/mit_b0.pth',
                  style='pytorch'),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

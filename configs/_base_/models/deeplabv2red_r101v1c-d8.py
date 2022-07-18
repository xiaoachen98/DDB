_base_ = ['deeplabv2red_r50-d8.py']
# models settings
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnet101_v1c')))

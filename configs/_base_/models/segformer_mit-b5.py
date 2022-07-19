# model settings
_base_ = ['segformer_mit-b0.py']
model = dict(
    backbone=dict(
        type='mit_b5',
        pretrained='pretrained/mit_b5.pth'))

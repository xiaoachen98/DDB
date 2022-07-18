# optimizer
optimizer = dict(stu_model=dict(type='AdamW',
                               lr=6e-5,
                               betas=(0.9, 0.999),
                               weight_decay=0.01,
                               paramwise_cfg=dict(custom_keys=dict(head=dict(lr_mult=10.0)))))
optimizer_config = None

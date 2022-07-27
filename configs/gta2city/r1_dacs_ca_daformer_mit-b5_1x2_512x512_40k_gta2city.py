_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/models/daformer_sepaspp_mit-b5.py",
    "../_base_/datasets/st_gta2city_512x512.py",
    # Basic UDA Self-Training
    "../_base_/uda/dacs.py",
    # AdamW Optimizer
    "../_base_/schedules/adamw.py",
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    "../_base_/schedules/poly10warm.py",
    # Schedule
    "../_base_/schedules/schedule_40k.py",
]
# Random Seed
seed = 0
# Fine class path domain bridging with class-mix
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        _delete_=True,
        type='UDADataset',
        source={{_base_.train_gtav}},
        target={{_base_.train_cityscapes}}
    ))
# Optimizer Hyper-parameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 1
# Meta Information for Result Analysis
exp = "daformer"

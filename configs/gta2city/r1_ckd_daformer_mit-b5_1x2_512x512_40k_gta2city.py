_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/models/daformer_sepaspp_mit-b5.py",
    "../_base_/datasets/ckd_gta2city_512x512.py",
    # Basic UDA Self-Training
    "../_base_/uda/ckd.py",
    # AdamW Optimizer
    "../_base_/schedules/adamw_ckd.py",
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    "../_base_/schedules/poly10warm.py",
    # Schedule
    "../_base_/schedules/schedule_40k.py",
]
uda = dict(
    cu_model_load_from="checkpoints/gta2city_round1/gta2city_daformer-cu_latest.pth",
    ca_model_load_from="checkpoints/gta2city_round1/gta2city_daformer-ca_latest.pth",
    cu_proto_path="prototypes/gta2city_round1/gta2city_dacs-fd-rcs-misc-cu_daformer.pth",
    ca_proto_path="prototypes/gta2city_round1/gta2city_dacs-fd-rcs-misc-ca_daformer.pth",
    proto_rectify=True,
)
# Optimizer Hyper-parameters
optimizer = dict(
    stu_model=dict(
        lr=6e-5,
        paramwise_cfg=dict(
            custom_keys=dict(
                head=dict(lr_mult=10.0),
                pos_block=dict(decay_mult=0.0),
                norm=dict(decay_mult=0.0)))))
# Random Seed
seed = 0
n_gpus = 1
# Meta Information for Result Analysis
exp = "ckd"
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4
)

_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/models/deeplabv2red_r101v1c-d8_adapter.py",
    "../_base_/datasets/st_gta2city_512x512.py",
    # Basic UDA Self-Training
    "../_base_/uda/st.py",
    # AdamW Optimizer
    "../_base_/schedules/adamw.py",
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    "../_base_/schedules/poly10warm.py",
    # Schedule
    "../_base_/schedules/schedule_40k.py",
]
uda = dict(distilled_model_path="checkpoints/gta2city_round1/gta2city_ckd-pro_dlv2.pth")
# Random Seed
seed = 0
# Fine class path domain bridging with class-mix
data = dict(samples_per_gpu=4, workers_per_gpu=4, train=dict(mask="class"))
# Optimizer Hyper-parameters
optimizer_config = None
optimizer = dict(
    lr=6e-06, paramwise_cfg=dict(custom_keys=dict(head=dict(lr_mult=10.0)))
)
n_gpus = 1
# Meta Information for Result Analysis
exp = "st"

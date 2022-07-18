_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/models/deeplabv2red_r101v1c-d8_adapter.py",
    "../_base_/datasets/ckd_gta+syns2city_512x512.py",
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
    cu_model_load_from="",
    ca_model_load_from="",
    cu_proto_path="",
    ca_proto_path="",
    proto_rectify=True,
)
# Random Seed
seed = 0
n_gpus = 1
# Meta Information for Result Analysis
exp = "ckd"
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)

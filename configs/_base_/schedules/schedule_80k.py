_base_ = ['schedule_40k.py']
runner = dict(max_iters=80000)
# Logging Configuration
checkpoint_config = dict(interval=8000)
evaluation = dict(interval=8000)

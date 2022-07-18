_base_ = ['schedule_40k.py']
runner = dict(max_iters=60000)
# Logging Configuration
checkpoint_config = dict(interval=6000)
evaluation = dict(interval=6000)

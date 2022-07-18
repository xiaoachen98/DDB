_base_ = ['schedule_40k.py']
runner = dict(max_iters=160000)
# Logging Configuration
checkpoint_config = dict(interval=16000)
evaluation = dict(interval=16000)

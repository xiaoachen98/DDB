_base_ = [
    "r1_dacs-fd-rcs-misc_ca_daformer_mit-b5_1x2_512x512_40k_gta2city.py"
]
data = dict(samples_per_gpu=4)
# Meta Information for Result Analysis
exp = "daformer"

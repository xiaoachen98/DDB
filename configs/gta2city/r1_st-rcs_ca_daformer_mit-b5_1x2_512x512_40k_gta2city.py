_base_ = [
    "r1_st_ca_daformer_mit-b5_1x2_512x512_40k_gta2city.py"
]
# Add Rare Class Sampling
data = dict(
    train=dict(
        # Rare Class Sampling
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)))

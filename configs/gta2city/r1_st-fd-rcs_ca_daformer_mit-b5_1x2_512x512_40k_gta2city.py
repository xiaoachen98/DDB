_base_ = [
    "r1_st_ca_daformer_mit-b5_1x2_512x512_40k_gta2city.py"
]
# Modifications to Basic UDA
uda = dict(
    # Thing-Class Feature Distance
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    imnet_feature_dist_scale_min_ratio=0.75)
# Add Rare Class Sampling
data = dict(
    train=dict(
        # Rare Class Sampling
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)))

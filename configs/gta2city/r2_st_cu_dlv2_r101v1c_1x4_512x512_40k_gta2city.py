_base_ = "./r2_st_ca_dlv2_r101v1c_1x4_512x512_40k_gta2city.py"
# Coarse region path domain bridging with cut-mix
data = dict(train=dict(mask="cut"))
uda = dict(distilled_model_path="checkpoints/gta2city_round1/gta2city_ckd-pro_dlv2.pth")

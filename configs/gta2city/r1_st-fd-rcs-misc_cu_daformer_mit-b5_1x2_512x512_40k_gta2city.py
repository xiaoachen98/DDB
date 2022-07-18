_base_ = "./r1_st-fd-rcs-misc_ca_daformer_mit-b5_1x2_512x512_40k_gta2city.py"
# Coarse region path domain bridging with cut-mix
data = dict(train=dict(mask="cut"))

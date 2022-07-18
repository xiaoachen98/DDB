_base_ = "./r1_st_ca_dlv2_r101v1c_1x4_512x512_40k_gta2city.py"
# Coarse region path domain bridging with cut-mix
data = dict(train=dict(mask="cut"))

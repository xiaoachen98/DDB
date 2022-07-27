_base_ = [
    "r1_dacs-fd-rcs-misc_ca_daformer_mit-b5_1x2_512x512_40k_gta2city.py"
]
# Modifications to Basic UDA
uda = dict(mix='cut')
# Meta Information for Result Analysis
exp = "daformer"

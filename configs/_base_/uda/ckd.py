# Baseline UDA
uda = dict(
    type='CKD',
    pseudo_threshold=0.968,
    teacher_model_cfg=None,
    cu_model_load_from='',
    ca_model_load_from='',
    soft_distill=False,
    soft_distill_w=0.1,
    proto_rectify=False,
    rectify_on_prob=True,
    use_pl_weight=False,
    temp=1,
    cu_proto_path='',
    ca_proto_path='',
    debug_img_interval=1000)

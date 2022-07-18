import warnings

from mmseg.models import SEGMENTORS # Keep the same registry as mmseg

UDA = SEGMENTORS
HEADS = SEGMENTORS


def build_uda_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build model."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            "train_cfg and test_cfg is deprecated, " "please specify them in model",
            UserWarning,
        )
    assert (
        cfg.model.get("train_cfg") is None or train_cfg is None
    ), "train_cfg specified in both outer field and model field "
    assert (
        cfg.model.get("test_cfg") is None or test_cfg is None
    ), "test_cfg specified in both outer field and model field "
    if "uda" in cfg:
        cfg.uda.model = cfg.model
        cfg.uda.max_iters = cfg.runner.max_iters
        return UDA.build(
            cfg.uda, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg)
        )
    else:
        warnings.warn(
            "No UDA keywords are found, and the SEGMENTOR is built normally",
            UserWarning,
        )
        return SEGMENTORS.build(
            cfg.model, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg)
        )

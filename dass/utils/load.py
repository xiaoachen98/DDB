from mmcv.runner import load_state_dict, _load_checkpoint
from mmcv.utils.logging import print_log
from collections import OrderedDict
import re


def custom_load_checkpoint(
    model,
    filename,
    map_location=None,
    strict=False,
    logger=None,
    revise_keys=[(r"^module\.", "")],
):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Default: strip
            the prefix 'module.' by [(r'^module\\.', '')].

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location, logger)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")
    # get state_dict from checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    metadata = getattr(state_dict, "_metadata", OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict({re.sub(p, r, k): v for k, v in state_dict.items()})
    # Keep metadata in state_dict
    state_dict._metadata = metadata
    prefixs = ("model", "stu_model")
    for prefix in prefixs:
        if not prefix.endswith("."):
            prefix += "."
        prefix_len = len(prefix)
        new_state_dict = {
            k[prefix_len:]: v for k, v in state_dict.items() if k.startswith(prefix)
        }
        if len(new_state_dict) > 0:
            state_dict = new_state_dict
            print_log(f"get state dict with prefix {prefix}", logger=logger)
            break
    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint

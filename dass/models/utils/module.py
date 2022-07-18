import torch
import torch.nn.functional as F


def kl_loss(s: torch.Tensor, t: torch.Tensor):
    h, w = t.shape[-2:]
    return F.kl_div(s, t, reduction='batchmean') / (h * w)


def mse_loss(s: torch.Tensor, t: torch.Tensor, **kwargs):
    return F.mse_loss(s, t, **kwargs)


def js_loss(s: torch.Tensor, t: torch.Tensor):
    log_mean_out = ((s + t) / 2.).log()
    return (kl_loss(log_mean_out, s) + kl_loss(log_mean_out, t)) / 2.

import torch
import logging

import torch.nn.functional as F
from sbgm.losses import EDMLoss

logger = logging.getLogger(__name__)

@torch.no_grad() # Disable gradient computation for monitoring
def edm_cosine_metric(loss_obj, model, x0, *, cond_img=None, lsm_cond=None, topo_cond=None, y=None, lr_ups=None, sdf_cond=None):
    """
    Compute the cosine similarity metric for EDM models as per Karras et al. (2022).
    Similarity metric between predicted x0_hat and x0 for EDM.
    Uses the same sigma sampling as in the EDMLoss object.
    """
    if not isinstance(loss_obj, EDMLoss):
        logger.warning("edm_cosine_metric is only defined for EDMLoss. Returning None.")
        return None  # Metric only defined for EDMLoss
    
    B = x0.shape[0]
    device = x0.device
    dtype = x0.dtype

    # Sample sigma from log-normal distribution
    sigma = loss_obj.sample_sigma(B, device, dtype=dtype)
    n = torch.randn_like(x0)
    x_t = x0 + sigma.view(B, 1, 1, 1) * n
    
    # Model is EDMPrecondUNet, predict x0_hat
    x0_hat = model(x_t, sigma, cond_img=cond_img, lsm_cond=lsm_cond, topo_cond=topo_cond, y=y, lr_ups=lr_ups)

    # Flatten per-sample and compute cosine
    cos = F.cosine_similarity(x0_hat.flatten(1), x0.flatten(1), dim=1, eps=1e-8).mean()
    return float(cos)

def _masked_corrcoef_per_sample(
        a: torch.Tensor,
        b: torch.Tensor,
        mask: torch.Tensor | None = None,
        eps: float = 1e-8
) -> torch.Tensor:
    """
        Mean Pearson correlation across the batch, computed per-sample over masked pixels.
        If mask is None, uses all pixels. Ignores samples with near-zero variance.
    """
    B = a.shape[0]
    vals = []
    for i in range(B):
        mi = mask[i] if mask is not None else None
        if mi is not None:
            # Expect land~1. If float, threshold at 0.5, then broadcast to [C,H,W]
            if mi.dtype != torch.bool:
                mi = (mi > 0.5)
            mi = mi.expand_as(a[i])
            ai = a[i][mi]
            bi = b[i][mi]
        else:
            ai = a[i].reshape(-1)
            bi = b[i].reshape(-1)

        if ai.numel() < 2:
            continue

        ai = ai - ai.mean()
        bi = bi - bi.mean()
        denom = ai.std(unbiased=False) * bi.std(unbiased=False) + eps
        corr_i = (ai * bi).mean() / denom
        if torch.isfinite(corr_i):
            vals.append(corr_i)

    if len(vals) == 0:
        return torch.tensor(float('nan'), device=a.device)
    return torch.stack(vals).mean()

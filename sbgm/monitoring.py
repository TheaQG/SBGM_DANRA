import torch
import torch.nn.functional as F
from sbgm.losses import EDMLoss

@torch.no_grad() # Disable gradient computation for monitoring
def edm_cosine_metric(loss_obj, model, x0, *, cond_img=None, lsm_cond=None, topo_cond=None, y=None, lr_ups=None, sdf_cond=None):
    """
    Compute the cosine similarity metric for EDM models as per Karras et al. (2022).
    Similarity metric between predicted x0_hat and x0 for EDM.
    Uses the same sigma sampling as in the EDMLoss object.
    """
    if not isinstance(loss_obj, EDMLoss):
        print("WARNING: edm_cosine_metric is only defined for EDMLoss. Returning None.")
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
    cos = F.cosine_similarity(x0_hat.flatten(1), x0.flatten(1), dim=1).mean()
    return float(cos)
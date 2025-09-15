"""
    Monitoring functions for EDM.
    Currently includes:
        - edm_cosine_metric: Cosine similarity metric for EDM models as per Karras et al. (2022).
        - _masked_corrcoef_per_sample: Mean Pearson correlation across the batch, computed per-sample over masked pixels.
        - report_precip_extremes: Reports extremes in a back-transformed precipitation tensor.

    TODO:
        - FSS (Fractions Skill Score) implementation at 5, 10, 20 km scales
        - PSD slope metric
        - Q95/Q99 metrics
        - Wet-day frequency 
        - Other metrics from "Evaluating Generative Models via Precision and Recall" (Sajjadi et al. 2018)
"""

import torch
import logging
import os

import torch.nn.functional as F
from datetime import datetime

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



def report_precip_extremes(x_bt: torch.Tensor, name: str, cap_mm_day: float = 500.0, logger=print):
    """
        Reports extremes in a back-transformed precipitation tensor.
        Values below 0 are counted as negative, values above cap_mm_day are counted as extreme.
    """
    flat = x_bt.flatten(1)
    p999 = torch.quantile(flat, 0.999, dim=1)
    mx = torch.max(flat, dim=1).values
    n_ex = 0
    vals_ex = []
    n_b0 = 0
    vals_b0 = []
    for i, (p, m) in enumerate(zip(p999.tolist(), mx.tolist())):
        if m > max(5.0 * p, cap_mm_day):
            logger(f"{name} sample {i} has extreme precipitation: max={m:.1f} mm/day > max(5xp99.9={p:.1f} mm/day)")
            n_ex += 1
            vals_ex.append(m)
        if m < 0:
            logger(f"{name} sample {i} has negative precipitation: max={m:.1f} mm/day < 0")
            n_b0 += 1
            vals_b0.append(m)
    if n_b0 > 0 and n_ex > 0:
        return {'has_extreme': True, 'n_extreme': n_ex, 'extreme_values': vals_ex,
                'has_below_zero': True, 'n_below_zero': n_b0, 'below_zero_values': vals_b0}
    if n_ex > 0:
        return {'has_extreme': True, 'n_extreme': n_ex, 'extreme_values': vals_ex}
    if n_b0 > 0:
        return {'has_below_zero': True, 'has_below_zero': True, 'n_below_zero': n_b0, 'below_zero_values': vals_b0}

    return {'has_extreme': False}


def setup_logger(log_dir, name="train_log", log_to_stdout=True):
    # Set up the path for the log directory
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")

    # Set up a logger, with level set to INFO which means it will log INFO, WARNING, ERROR, and CRITICAL messages
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers (we remove all handlers to avoid duplicates)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler to write logs to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    # Set the format for the log messages
    file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    # Apply the formatter to the file handler
    logger.addHandler(file_handler)

    # Optional: also print to terminal
    if log_to_stdout:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(file_formatter)
        logger.addHandler(stream_handler)

    logger.info(f"Logging to {log_path}")
    return logger

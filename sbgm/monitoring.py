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
import math
import json 

import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime

from sbgm.losses import EDMLoss

logger = logging.getLogger(__name__)

def _to_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.tensor(x)

@torch.no_grad()
def compute_fss_at_scales(gen_bt: torch.Tensor, hr_bt: torch.Tensor, *, mask:torch.Tensor|None,
                  grid_km_per_px: float, fss_km: list[float], thr_mm: float, eps: float=1e-8) -> dict[str, float]:
    """
        Fractions Skill Score (FSS) for exceedance over threshold thr_mm at different spatial scales (km).
        gen_bt, hr_bt: [B,1,H,W] back-transformed precipitation tensors in mm/day.
        mask: [B,1,H,W] land mask (1=land, 0=sea) or None
    """
    gen_bt = _to_tensor(gen_bt).float()
    hr_bt = _to_tensor(hr_bt).float()
    if mask is not None:
        mask = mask.bool().expand_as(gen_bt)

    X = (gen_bt > thr_mm).float() # Binary exceedance for generated
    Y = (hr_bt > thr_mm).float() # Binary exceedance for HR

    out = {}
    for km in fss_km:
        rad_px = int(max(1, round(float(km) / float(grid_km_per_px)))) # Radius in pixels (int)
        k = 2 * rad_px + 1 # Odd kernel size (int)
        # Box filter via average pooling
        Xs = F.avg_pool2d(X, kernel_size=k, stride=1, padding=rad_px)
        Ys = F.avg_pool2d(Y, kernel_size=k, stride=1, padding=rad_px)
        if mask is not None:
            m = mask.float()
            num = ((Xs - Ys) ** 2 * m).sum() / (m.sum() + eps) # Mean squared error over land
            den = (((Xs ** 2 + Ys ** 2) * m).sum() / (m.sum() + eps)) + eps
        else:
            num = ((Xs - Ys) ** 2).mean() # Mean squared error over all pixels
            den = (Xs ** 2 + Ys ** 2).mean() + eps
        fss = 1.0 - num / den # Fractions Skill Score
        out[f'{int(km)}km'] = float(fss)#float(fss.clamp(0.0, 1.0)) # Clamp to [0,1]
    
    return out

# === PSD Slope metric
@torch.no_grad()
def _radial_psd_slope_single(img: torch.Tensor, *, mask:torch.Tensor|None=None, ignore_low_k_bins: int=1) -> float:
    """
        Compute the slope of the radially-averaged 2D power spectrum for a single 2D field.
        Returns the log-log slope (beta) from a linear fit of log10(P(k)) vs log10(k).
        The lowest *ignore_low_k_bins* raidal frequency bins are dropped to avoid DC/very-low-k dominance.
    """
    # Ensure 2-D (H,W)
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    elif img.ndim == 4 and img.shape[0] == 1 and img.shape[1] == 1:
        img = img[0,0]
    elif img.ndim != 2:
        img = img.squeeze()
        if img.ndim != 2:
            raise ValueError(f"Input image must be 2D, got shape {img.shape}")
    
    # Optional mask: zero out masked pixels (keeping shape)
    if mask is not None:
        m = mask
        if m.ndim == 3 and m.shape[0] == 1:
            m = m[0]
        elif m.ndim == 4 and m.shape[0] == 1 and m.shape[1] == 1:
            m = m[0,0]
        if m.dtype != torch.bool:
            m = (m > 0.5)
        img = img * m.float()

    # 2D FFT and power spectrum
    F = torch.fft.fft2(img.float())
    P = (F.real ** 2 + F.imag ** 2)  # Power spectrum
    P = torch.fft.fftshift(P)  # Shift zero freq to center

    H, W = img.shape[-2:] # Height, Width
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0  # Center coordinates
    y = torch.arange(H, device=img.device)
    x = torch.arange(W, device=img.device)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    R = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2)  # Radial distances

    # Radial bins: 1 px per bin
    r = R.flatten()
    p = P.flatten()
    rmax = int(torch.max(R).item())
    if rmax < (ignore_low_k_bins + 2):
        return float('nan')  # Not enough radial bins to compute slope
    
    # Bin by integer radius
    nbins = rmax + 1
    sums = torch.zeros(nbins, device=img.device)
    counts = torch.zeros(nbins, device=img.device)
    idx = r.long().clamp(0, rmax)
    sums.scatter_add_(0, idx, p) # Sum power in each radial bin
    ones = torch.ones_like(p, device=img.device)
    counts.scatter_add_(0, idx, ones) # Count pixels in each radial bin

    with torch.no_grad():
        valid = counts > 0
        radii = torch.arange(nbins, device=img.device)[valid]
        prof = (sums[valid] / counts[valid]).clamp(min=1e-12)  # Average power per bin, avoid log(0)

    # Drop DC and a few low-k bins
    if radii.numel() <= (ignore_low_k_bins + 1):
        return float('nan')  # Not enough bins to fit
    radii = radii[ignore_low_k_bins:]
    prof = prof[ignore_low_k_bins:]

    # Linear fit in log-log space
    k = radii.detach().cpu().numpy()
    Pk = prof.detach().cpu().numpy()
    if np.any(k <= 0) or np.any(np.isnan(Pk)):
        return float('nan')
    
    xlog = np.log10(k)
    ylog = np.log10(Pk)
    # Guard against degenerate cases (e.g. constant image)
    if not np.all(np.isfinite(xlog)) or not np.all(np.isfinite(ylog)):
        return float('nan')
    if xlog.size < 3:
        return float('nan')
    beta, _ = np.polyfit(xlog, ylog, 1)  # Slope is beta
    
    return float(beta)


@torch.no_grad()
def compute_psd_slope(
    gen_bt: torch.Tensor,
    hr_bt: torch.Tensor|None= None,
    *,
    mask:torch.Tensor|None=None,
    ignore_low_k_bins: int=1
) -> dict[str, float]:
    """
        Compute radial PSD slope (log-log slope of P(k) vs k) for generated fields and (optionally) HR reference fields.
        Inputs are expected to be back-transformed precipitation (mm/day), shaped [B, 1, H, W] or [B, H, W].
        If *mask* is provided ([B, 1, H, W] or [B, H, W]) only masked pixels are used in the computation.
        Returns a dict with keys
        - 'psd_slope_gen': mean PSD slope for generated fields across the batch
        - 'psd_slope_hr': mean PSD slope for HR fields across the batch (if hr_bt is provided)
        - 'psd_slope_delta': difference in mean PSD slope (gen - hr) if hr_bt is provided
    """
    def _prep(t):
        t = _to_tensor(t).float()
        if t.ndim == 3: # [B,H,W] -> [B,1,H,W]
            t = t[:, None, :, :]
        return t
    
    G = _prep(gen_bt)
    M = None
    if mask is not None:
        M = _prep(mask).bool()
    Href = _prep(hr_bt) if hr_bt is not None else None

    betas_g = []
    betas_h = [] if Href is not None else None

    B = G.shape[0]
    for i in range(B):
        mi = M[i] if M is not None else None
        betas_g.append(_radial_psd_slope_single(G[i], mask=mi, ignore_low_k_bins=ignore_low_k_bins))
        if Href is not None and betas_h is not None:
            betas_h.append(_radial_psd_slope_single(Href[i], mask=mi, ignore_low_k_bins=ignore_low_k_bins))

    def _clean_mean(arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size > 0 else float('nan')
    
    out = {'psd_slope_gen': _clean_mean(betas_g)}
    if betas_h is not None:
        mhr = _clean_mean(betas_h)
        out['psd_slope_hr'] = mhr
        if np.isfinite(out['psd_slope_gen']) and np.isfinite(mhr):
            out['psd_slope_delta'] = float(mhr - out['psd_slope_gen'])

    return out



@torch.no_grad()
def compute_q95_q99_and_wet_day(
    gen_bt: torch.Tensor,
    hr_bt: torch.Tensor|None = None,
    *,
    mask:torch.Tensor|None=None,
    wet_threshold_mm: float=1.0) -> dict[str, float]:
    """
        Compute Q95, Q99 and wet-day frequency (> wet_threshold_mm) for generated fields and (optionally) HR reference fields.
        Inputs can be [B,1,H,W] or [B,H,W] or numpy arrays.
        *mask* is optional and should be broadcastable to [B,1,H,W] or [B,H,W].
        Returns keys:
        - 'gen_q95', 'gen_q99', 'gen_wet_freq'
        - 'hr_q95', 'hr_q99', 'hr_wet_freq' (if hr_bt is provided)
    """
    def _prep(t):
        if t is None:
            return None
        t = _to_tensor(t).float()
        if t.ndim == 3: # [B,H,W] -> [B,1,H,W]
            t = t[:, None, :, :]
        return t
    
    G = _prep(gen_bt)
    H = _prep(hr_bt) 
    M = _prep(mask)
    if M is not None:
        M = M.bool()

    def _metrics(t: torch.Tensor | None, m: torch.Tensor | None):
        if t is None:
            return float('nan'), float('nan'), float('nan')
        if m is not None:
            t = t.masked_fill(~m.expand_as(t), float('nan'))
        # Flatten over spatial dims and channels
        flat = t.reshape(t.shape[0], -1)
        # Remove NaNs
        flat_np = flat.detach().cpu().numpy()
        flat_np = flat_np[~np.isnan(flat_np)]
        if flat_np.size == 0:
            return float('nan'), float('nan'), float('nan')
        q95 = float(np.percentile(flat_np, 95))
        q99 = float(np.percentile(flat_np, 99))
        wet_freq = float(np.mean(flat_np > wet_threshold_mm))
        return q95, q99, wet_freq
    
    q95g, q99g, wfg = _metrics(G, M)
    out = {'gen_q95': q95g, 'gen_q99': q99g, 'gen_wet_freq': wfg}
    if H is not None:
        q95h, q99h, wfh = _metrics(H, M)
        out.update({'hr_q95': q95h, 'hr_q99': q99h, 'hr_wet_freq': wfh})

    return out


@torch.no_grad() # Disable gradient computation for monitoring
def edm_cosine_metric(loss_obj, model, x0, *, cond_img=None, lsm_cond=None, topo_cond=None, y=None, lr_ups=None, sdf_cond=None):
    """
    Compute the cosine similarity metric for EDM models.
    Similarity metric between predicted x0_hat and x0 for EDM.
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



def report_precip_extremes(x_bt: torch.Tensor, name: str, cap_mm_day: float = 500.0):
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
            logger.info(f"{name} sample {i} has extreme precipitation: max={m:.1f} mm/day > max(5xp99.9={p:.1f} mm/day)")
            n_ex += 1
            vals_ex.append(m)
        if m < 0:
            logger.info(f"{name} sample {i} has negative precipitation: max={m:.1f} mm/day < 0")
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






# === Diagnostics helpers for EDM training ===
@torch.no_grad()
def _finite_mask(x: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(x)
@torch.no_grad()
def tensor_stats(x: torch.Tensor, name: str, pctiles=(0.1, 1, 5, 50, 95, 99, 99.9), log_fn=logger.info):
    """
        Quick, safe stats with NaN/Inf awareness. Logs: shape, dtype, device, finite ratio, mean, std, min, max, a few percentiles. 
    """
    if x is None:
        log_fn(f"{name}: None")
        return
    
    # cpu snapshot for percentiles (subsample to keep cheap)
    x_detached = x.detach()
    mask = _finite_mask(x_detached)
    n_total = x_detached.numel()
    n_finite = int(mask.sum().item())

    log_fn(f"[{name}] shape={tuple(x_detached.shape)}, dtype={x_detached.dtype}, device={x_detached.device}, finite_ratio={n_finite} / {n_total} ({100.0 * n_finite / n_total:.2f}%)")

    if n_finite == 0:
        log_fn(f"[{name}] !!! No finite values (NaN/Inf everywhere) !!!")
        return
    
    xf = x_detached[mask]
    # downsample if huge
    if xf.numel() > 1_000_000:
        idx = torch.randint(0, xf.numel(), (200_000,), device=xf.device)
        xf = xf.view(-1)[idx]

    # core stats on device
    x_min = float(xf.min().item())
    x_max = float(xf.max().item())
    x_mean = float(xf.mean().item())
    x_std = float(xf.std(unbiased=False).item())

    # move small vector for percentiles
    xcpu = xf.float().flatten().cpu()
    # percentiles
    pcts = {}
    for p in pctiles:
        q = torch.quantile(xcpu, torch.tensor(float(p) / 100.0))
        pcts[p] = float(q.item())

    pts_str = " ".join([f"P{int(p)}={pcts[p]:.4g}" for p in pctiles])
    log_fn(f"[{name}] min={x_min:.4g} max={x_max:.4g} mean={x_mean:.4g} std={x_std:.4g} | {pts_str}")

@torch.no_grad()
def save_histogram(x:torch.Tensor, save_path: str, bins: int = 200, range_: tuple[float,float] | None = None):
    """
        Save a histogram of tensor x to the specified path.
        x: input tensor
    """
    x = x.detach().flatten()
    x = x[torch.isfinite(x)]  # Keep only finite values
    if x.numel() == 0:
        logger.warning(f"[save_histogram]: No finite values in tensor, skipping histogram save to {save_path}.")
        return
    xcpu = x.float().cpu()
    lo = float(xcpu.min().item()) if range_ is None else range_[0]
    hi = float(xcpu.max().item()) if range_ is None else range_[1]
    
    hist, edges = np.histogram(xcpu.numpy(), bins=bins, range=(lo, hi))
    name = f"hist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = os.path.join(save_path, name)
    with open(path, 'w') as f:
        json.dump({"edges": edges.tolist(), "hist": hist.tolist()}, f)

@torch.no_grad()
def plot_saved_histograms(save_path: str, fig_save_path: str | None = None):
    """
        Plot all saved histograms in the current directory (files named hist_*.json).
    """
    import glob
    files = glob.glob(os.path.join(save_path, "hist_*.json"))
    if len(files) == 0:
        logger.warning(f"No histogram files found in {save_path}.")
        return
    
    plt.figure(figsize=(10,6))
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
        edges = np.array(data['edges'])
        hist = np.array(data['hist'])
        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.plot(centers, hist, label=os.path.basename(file))
    
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Saved Histograms')
    plt.legend()
    plt.grid()

    if fig_save_path is not None:
        plt.savefig(fig_save_path)
        logger.info(f"Saved histogram plot to {fig_save_path}.")
    else:
        plt.show()
    plt.close()
    


import torch
import numpy as np
from typing import Optional, Tuple, Dict, Sequence

# Extremes needs SciPy
from scipy.stats import genextreme as scipy_gev
from scipy.stats import genpareto as scipy_gpd

# =========================
# Helpers (Torch)
# =========================

def _ensure_float(t: torch.Tensor) -> torch.Tensor:
    return t if torch.is_floating_point(t) else t.float()

def _broadcast_mask(mask: Optional[torch.Tensor], target: torch.Tensor) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.dtype != torch.bool:
        mask = mask > 0.5
    while mask.dim() < target.dim():
        mask = mask.unsqueeze(1)
    return mask.expand_as(target)

# =========================
# PIT & Rank histograms
# =========================
@torch.no_grad()
def pit_values_from_ensemble(
    obs: torch.Tensor,     # [B,H,W]
    ens: torch.Tensor,     # [B,M,H,W]
    mask: Optional[torch.Tensor] = None,
    randomized: bool = True,
) -> torch.Tensor:
    """
    Randomized PIT for empirical ensemble CDF (good for mixed discrete-continuous precip).
    For each (b,i,j): PIT = F^-(y-) + U * (F(y) - F^-(y-)), where F is empirical CDF from ensemble.
    Returns: 1-D tensor of PIT values over all valid pixels.
    """
    obs = _ensure_float(obs)
    ens = _ensure_float(ens)
    B, M, H, W = ens.shape

    if mask is not None:
        m = _broadcast_mask(mask, obs.unsqueeze(1))
        if m is not None:
            m = m.squeeze(1)  # [B,H,W]
        else:
            m = torch.ones_like(obs, dtype=torch.bool)
    else:
        m = torch.ones_like(obs, dtype=torch.bool)

    obs_flat = obs[m]                                         # [N]
    ens_flat = ens.permute(0,2,3,1)[m]                        # [N, M]

    # Sort ensemble per location
    ens_sorted, _ = torch.sort(ens_flat, dim=1)               # [N, M]

    # counts strictly less and equal
    less = (ens_sorted < obs_flat.unsqueeze(1)).sum(dim=1)    # [N]
    equal = (ens_sorted == obs_flat.unsqueeze(1)).sum(dim=1)  # [N]

    if randomized:
        # U in [0,1)
        U = torch.rand_like(obs_flat)
        pit = (less.float() + U * equal.float()) / float(M)
    else:
        pit = (less.float() + 0.5 * equal.float()) / float(M)   

    return pit  # [N]

@torch.no_grad()
def rank_histogram(
    obs: torch.Tensor,     # [B,H,W]
    ens: torch.Tensor,     # [B,M,H,W]
    mask: Optional[torch.Tensor] = None,
    randomize_ties: bool = True,
) -> torch.Tensor:
    """
    Rank histogram counts over M+1 bins (ranks 0..M).
    For ties, we either randomize the rank uniformly over the tied interval (recommended),
    or place at mid-rank.
    Returns: counts [M+1] as float tensor (not normalized).
    """
    obs = _ensure_float(obs)
    ens = _ensure_float(ens)
    B, M, H, W = ens.shape

    if mask is not None:
        m = _broadcast_mask(mask, obs.unsqueeze(1))
        if m is not None:
            m = m.squeeze(1)  # [B,H,W]
        else:
            m = torch.ones_like(obs, dtype=torch.bool)
    else:
        m = torch.ones_like(obs, dtype=torch.bool)

    obs_flat = obs[m]                           # [N]
    ens_flat = ens.permute(0,2,3,1)[m]          # [N, M]
    Mfloat = float(M)

    # Sort ensemble & compute how many are < and <= obs
    ens_sorted, _ = torch.sort(ens_flat, dim=1)
    less = (ens_sorted < obs_flat.unsqueeze(1)).sum(dim=1).float()   # [N]
    leq  = (ens_sorted <= obs_flat.unsqueeze(1)).sum(dim=1).float()  # [N]
    ties = leq - less                                                # [N]

    if randomize_ties:
        U = torch.rand_like(less)
        rank = less + U * ties
    else:
        rank = less + 0.5 * ties

    # rank is in [0, M]; bin to integer bins 0..M
    # numerical guard
    rank = torch.clamp(rank, 0.0, Mfloat)
    bins = torch.round(rank).to(torch.int64)  # nearest bin; alt: floor with epsilon handling
    # Better: place into nearest integer with a tiny jitter to avoid boundary pile-ups
    counts = torch.bincount(bins, minlength=M+1).to(torch.float32)
    return counts  # [M+1]

# =========================
# Extremes: Rx1day / Rx5day, GEV & POT with bootstrap CIs
# =========================

def _rolling_sum_np(x: np.ndarray, k: int) -> np.ndarray:
    # x shape [T], returns rolling sum length T-k+1
    if k == 1:
        return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    return c[k:] - c[:-k]

def rxk_series(series: np.ndarray, k: int, block_index: np.ndarray | None = None):
    """
    Compute RxK (maximum K-day running *sum*) within each block.
    - `series`: 1-D array (e.g., daily basin-mean precipitation in mm/day)
    - `k`: window length (days)
    - `block_index`: same length as `series`; equal values define a block
      (e.g., seasonal block id). If None, everything is one block.

    Returns
    -------
    values : np.ndarray
        Array of RxK values (one per non-empty block). If a block has
        fewer than `k` valid samples, it falls back to the max daily value
        in that block. Blocks with no finite values are skipped.
    meta : dict
        Metadata with {"k": k}.
    """
    s = np.asarray(series, dtype=float)
    if s.ndim != 1:
        s = s.reshape(-1)

    if block_index is None:
        block_index = np.zeros_like(s, dtype=int)
    else:
        block_index = np.asarray(block_index)
        if block_index.shape != s.shape:
            raise ValueError("block_index must have the same shape as series")

    values = []
    for b in np.unique(block_index):
        x = s[block_index == b]
        x = x[np.isfinite(x)]  # drop NaNs/inf
        if x.size == 0:
            # nothing to contribute from this block
            continue
        if k <= 1:
            values.append(np.max(x))
            continue
        if x.size < k:
            # Not enough samples for a k-day sum; fall back to daily max
            values.append(np.max(x))
            continue
        # Valid rolling k-day sums
        roll = np.convolve(x, np.ones(int(k), dtype=float), mode='valid')
        # Guard (shouldn’t be empty after size check, but be safe)
        values.append(np.max(roll) if roll.size > 0 else np.max(x))

    return np.asarray(values, dtype=float), {"k": int(k)}

def _return_level_gev(c, loc, scale, rp_years: float, block_per_year: float = 1.0) -> float:
    """
    Return level for GEV with SciPy's parameterization:
    genextreme(c). Here c is shape (xi), loc, scale.
    rp_years refers to return period in years; block_per_year is number of blocks per year.
    """
    # Non-exceedance probability for block maxima over return period:
    # For block maxima, return period (years) -> probability p = 1 - 1/(rp_years * block_per_year)
    p = 1.0 - 1.0/(rp_years * block_per_year)
    return float(scipy_gev.ppf(p, c, loc=loc, scale=scale))

def fit_gev_block_maxima_with_ci(
    block_maxima: np.ndarray,                                 # e.g., Rx1day annual maxima per year
    rps_years: Sequence[float] = (2, 5, 10, 20, 50),
    block_per_year: float = 1.0,                              # 4 for seasonal blocks, 1 for annual
    n_boot: int = 1000,
    random_state: Optional[int] = 42,
) -> Dict[str, np.ndarray | float]:
    """
    MLE fit of GEV to block maxima with nonparametric bootstrap CIs (resample maxima).
    Returns dict with params, return levels, and 95% CIs.
    """
    rng = np.random.default_rng(random_state)
    # SciPy's MLE (allows c any real)
    c, loc, scale = scipy_gev.fit(block_maxima)  # returns (c, loc, scale)
    rls = np.array([_return_level_gev(c, loc, scale, rp, block_per_year) for rp in rps_years])

    # Bootstrap resampling of maxima (same sample size)
    boot_rl = np.empty((n_boot, len(rps_years)))
    for b in range(n_boot):
        sample = rng.choice(block_maxima, size=block_maxima.size, replace=True)
        try:
            cb, lb, sb = scipy_gev.fit(sample)
            boot_rl[b, :] = [_return_level_gev(cb, lb, sb, rp, block_per_year) for rp in rps_years]
        except Exception:
            boot_rl[b, :] = np.nan

    lo = np.nanpercentile(boot_rl, 2.5, axis=0)
    hi = np.nanpercentile(boot_rl, 97.5, axis=0)

    return {
        "gev_shape": c,
        "gev_loc": loc,
        "gev_scale": scale,
        "return_periods_years": np.array(rps_years, dtype=float),
        "return_levels": rls,
        "return_levels_lo": lo,
        "return_levels_hi": hi,
        "n_blocks": block_maxima.size,
    }

def fit_pot_gpd_with_ci(
    y_daily: np.ndarray,                    # daily precip (mm)
    threshold: float,                       # u (e.g., seasonal P95 over wet days)
    days_per_year: float = 365.25,
    rps_years: Sequence[float] = (2, 5, 10, 20, 50),
    n_boot: int = 1000,
    random_state: Optional[int] = 42,
) -> Dict[str, np.ndarray | float]:
    """
    POT with GPD above threshold u, MLE using SciPy, RLs with bootstrap CIs.
    Return level formula:
      RL_T = u + (beta/xi) * ( (lambda_u * T_years)**xi - 1 )   if xi != 0
           = u + beta * log(lambda_u * T_years)                  if xi == 0
    where lambda_u = k / N is rate of exceedances per day.
    """
    y = np.asarray(y_daily, dtype=float)
    N = y.size
    exc = y[y > threshold] - threshold
    k = exc.size
    if k < 10:
        raise ValueError(f"Too few exceedances above u={threshold:.2f} (k={k}).")

    # SciPy GPD MLE (loc fixed at 0)
    c, loc, scale = scipy_gpd.fit(exc, floc=0.0)   # c=xi, scale=beta
    xi, beta = c, scale
    lam_u = k / N                                  # per day
    # Return levels
    rls = []
    for rp in rps_years:
        a = lam_u * rp * days_per_year
        if xi == 0.0:
            rl = threshold + beta * np.log(a)
        else:
            rl = threshold + (beta/xi) * (np.power(a, xi) - 1.0)
        rls.append(rl)
    rls = np.array(rls)

    # Bootstrap daily series with replacement of days (iid assumption);
    # for stronger dependence handling, you could block-bootstrap by weeks.
    rng = np.random.default_rng(random_state)
    boot_rl = np.empty((n_boot, len(rps_years)))
    for b in range(n_boot):
        yb = rng.choice(y, size=N, replace=True)
        excb = yb[yb > threshold] - threshold
        kb = excb.size
        if kb < 5:
            boot_rl[b, :] = np.nan
            continue
        try:
            cb, lb, sb = scipy_gpd.fit(excb, floc=0.0)
            xib, betab = cb, sb
            lam_b = kb / N
            vals = []
            for rp in rps_years:
                a = lam_b * rp * days_per_year
                if xib == 0.0:
                    rl = threshold + betab * np.log(a)
                else:
                    rl = threshold + (betab/xib) * (np.power(a, xib) - 1.0)
                vals.append(rl)
            boot_rl[b, :] = vals
        except Exception:
            boot_rl[b, :] = np.nan

    lo = np.nanpercentile(boot_rl, 2.5, axis=0)
    hi = np.nanpercentile(boot_rl, 97.5, axis=0)

    return {
        "gpd_xi": xi,
        "gpd_beta": beta,
        "threshold": threshold,
        "exceedances": int(k),
        "lambda_per_day": lam_u,
        "return_periods_years": np.array(rps_years, dtype=float),
        "return_levels": rls,
        "return_levels_lo": lo,
        "return_levels_hi": hi,
        "N_days": int(N),
    }

# =========================
# Glue: building series & blocks from Torch tensors
# =========================

def to_numpy_1d_series(x: torch.Tensor, mask: Optional[torch.Tensor] = None, agg: str = "mean") -> np.ndarray:
    """
    Convert daily HR fields to a 1D daily series via spatial aggregation.
    Args:
      x: [T,H,W] tensor (obs precip in mm/day)
      mask: [H,W] or [1,H,W] boolean (e.g., basin mask). If None, aggregate over all pixels.
      agg: 'mean' | 'sum' (use 'sum' for basin-total precip; 'mean' for areal average)
    """
    x = _ensure_float(x).detach().cpu()    
    if mask is not None:
        # Normalize mask to [H,W]
        if mask.dim() == 4 and mask.shape[:2] == (1, 1):
            mask = mask.squeeze(0).squeeze(0)
        elif mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)        
        if mask.dtype != torch.bool:
            mask = mask > 0.5
        mask = mask.to(x.device)
        x = x[:, mask.squeeze()]  # [T, Npix_valid]
    else:
        x = x.view(x.shape[0], -1)
    if agg == "sum":
        series = x.sum(dim=1)
    else:
        series = x.mean(dim=1)
    return series.numpy()

def seasonal_block_index(dates_np: np.ndarray) -> np.ndarray:
    """
    Build block indices per day for seasonal maxima.
    dates_np: array of 'YYYYMMDD' ints or np.datetime64
    Returns: integer block id per day (e.g., 2018*10 + season_code)
    """
    # Accept both integer yyyymmdd and datetime64
    if np.issubdtype(dates_np.dtype, np.datetime64):
        dt = dates_np.astype('datetime64[D]')
        yy = dt.astype('datetime64[Y]').astype(int) + 1970
        mm = (dt.astype('datetime64[M]') - dt.astype('datetime64[Y]')).astype(int) + 1
    else:
        # yyyymmdd -> year, month
        yy = dates_np // 10000
        mm = (dates_np % 10000) // 100

    # DJF(1), MAM(2), JJA(3), SON(4) — with DJF belonging to the *winter year* of Jan/Feb,
    # and Dec assigned to year+1 so that DJF is contiguous.
    season = np.select(
        [np.isin(mm, [12,1,2]), np.isin(mm, [3,4,5]), np.isin(mm, [6,7,8]), np.isin(mm, [9,10,11])],
        [1, 2, 3, 4], default=0
    )
    year_adj = yy + ((mm == 12).astype(int))  # Dec goes to next year
    block_id = year_adj * 10 + season
    return block_id.astype(int)

@torch.no_grad()
def pmm_from_ensemble(ens: torch.Tensor, # [B,M,H,W] ensemble tensor
                      mask: Optional[torch.Tensor] = None, # [B, H, W] or [H, W] boolean mask 
                      exclude_zeros: bool = False, # if True, exclude strictly 0.0 values from the pooled distribution
                      ) -> torch.Tensor:
    """
        Probability-Matched Mean (PMM) for univariate fields.

        For each sample b:
        1) Compute the ensemble-mean field μ(x) over members M.
        2) Take the ranks of μ(x) over VALID pixels.
        3) Pool *all* ensemble values across members at VALID pixels and sort that pooled 1-D list.
        4) Assign to each pixel the pooled value at the corresponding rank (quantile) of μ(x).

        This preserves the pooled marginal distribution of the ensemble while using μ(x) to supply the spatial
        pattern (classic PMM used in QPF).

        Args:
            ens: [B,M,H,W] univariate precipitation ensemble in model/physical space
            mask: Optional boolean mask broadcastable to [B,H,W]. Pixels where mask=False are left as the ensemble mean (i.e. PMM falls back to mean).
            exclude_zeros: If True, exclude strictly 0.0 values from the pooled distribution (useful for precipitation).
        Returns:
            pmm: [B,1,H,W] Probability-Matched Mean fields

    """
    if ens.dim() != 4:
        raise ValueError(f"ens must be [B, M, H, W] for univariate PMM, got {ens.shape}")

    ens = ens if torch.is_floating_point(ens) else ens.float()
    B, M, H, W = ens.shape

    device = ens.device

    # Prepare mask broadcast -> [B, H, W]
    if mask is None:
        mask_bhw = torch.ones((B, H, W), dtype=torch.bool, device=device)
    else:
        m = mask
        if m.dtype != torch.bool:
            m = m > 0.5
        while m.dim() < 3:
            m = m.unsqueeze(0)
        if m.shape[0] == 1 and B > 1:
            m = m.expand(B, -1, -1)
        mask_bhw = m.to(device)

    mean_field = ens.mean(dim=1)                 # [B, H, W]
    pmm_out = torch.empty((B, 1, H, W), dtype=ens.dtype, device=device)


    for b in range(B):
        valid = mask_bhw[b].view(-1)             # [H*W]
        if valid.sum() == 0:
            # No valid pixels → fall back to mean
            pmm_out[b, 0] = mean_field[b]
            continue

        # Spatial pattern ranks from ensemble mean
        mf_vals = mean_field[b].view(-1)[valid]  # [Nv]
        # ranks: 0..Nv-1
        ranks = torch.argsort(torch.argsort(mf_vals))

        # Pooled distribution across all M members at valid pixels
        pooled = ens[b].view(M, -1)[:, valid]    # [M, Nv]
        if exclude_zeros:
            pooled_flat = pooled.reshape(-1)
            pooled_flat = pooled_flat[pooled_flat != 0.0]
        else:
            pooled_flat = pooled.reshape(-1)

        if pooled_flat.numel() == 0:
            # Degenerate: no values after excluding zeros → fall back to mean
            out_vals = mf_vals
        else:
            pooled_sorted, _ = torch.sort(pooled_flat)  # [K]
            Nv = mf_vals.numel()
            # Quantile mapping: place rank r at q=(r+0.5)/Nv (midpoint rule)
            q = (ranks.to(torch.float32) + 0.5) / float(Nv)
            # Convert quantiles to indices in pooled_sorted
            # Use (K-1) so that q=1 maps to last index
            K = pooled_sorted.numel()
            idx = torch.clamp((q * (K - 1)).round().to(torch.long), 0, K - 1)
            out_vals = pooled_sorted[idx]

        # Write back into full image (fallback to mean for invalid)
        flat = mean_field[b].view(-1).clone()
        flat[valid] = out_vals
        pmm_out[b, 0] = flat.view(H, W)

    return pmm_out


def crps_ensemble(
    obs: torch.Tensor,              # [H,W]
    ens: torch.Tensor,              # [M,H,W]
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",        # 'mean' | 'sum' | 'none'
) -> torch.Tensor:
    """
    Continuous Ranked Probability Score (CRPS) for ensemble forecasts.
    0 = perfect; higher = worse.
    Fair CRPS estimator (Hersbach, 2000):
      CRPS = (1/M) * sum_i |X_i - y| - (1/(2 M^2)) * sum_{i,j} |X_i - X_j|

    Vectorized per-pixel. If reduction='none', returns [H,W]; otherwise scalar.
    """
    if obs.dim() != 2 or ens.dim() != 3:
        raise ValueError("obs must be [H,W] and ens must be [M,H,W]")
    M = ens.shape[0]
    obs = obs.to(ens.device, ens.dtype)

    # term1: (1/M) sum |Xi - y|
    term1 = (ens - obs.unsqueeze(0)).abs().mean(dim=0)  # [H,W]

    # term2: (1/(2M^2)) sum_{i,j} |Xi - Xj|  = (1/M^2) * sum_{i<j} (v_j - v_i)
    v = ens.view(M, -1)                  # [M, H*W]
    v_sorted, _ = torch.sort(v, dim=0)   # ascending
    i = torch.arange(M, device=ens.device, dtype=ens.dtype).unsqueeze(1)  # [M,1]
    # sum_{i<j} (v_j - v_i) = sum_k (2k - M + 1) * v_(k)
    pair_sum = ( (2*i - (M - 1)) * v_sorted ).sum(dim=0)  # [H*W]
    term2 = (pair_sum / (M * M)).view_as(term1)           # [H,W]

    crps = term1 - term2  # [H,W]

    if mask is not None:
        mask = mask.to(ens.device)
        if mask.dtype != torch.bool:
            mask = mask > 0.5
        vals = crps[mask]
        if vals.numel() == 0:
            return torch.tensor(0.0, device=ens.device)
        if reduction == "mean": return vals.mean()
        if reduction == "sum":  return vals.sum()
        return vals
    else:
        if reduction == "mean": return crps.mean()
        if reduction == "sum":  return crps.sum()
        return crps


def reliability_exceedance_lr_binned(
    obs: torch.Tensor,                  # [H,W] (mm/day)
    ens: torch.Tensor,                  # [M,H,W] (mm/day)
    threshold: float,
    lr_covariate: Optional[torch.Tensor] = None,  # [H,W]; if None, bin by forecast prob
    n_bins: int = 10,
    mask: Optional[torch.Tensor] = None,
    return_brier: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Reliability diagram data for threshold exceedance.
      p_hat = fraction of members >= threshold
      o     = 1(obs >= threshold)
    If lr_covariate is provided, bins are quantiles of that field; else equal-width bins over p_hat in [0,1].
    Returns: bin_center, prob_pred, freq_obs, count, and (optionally) Brier decomposition terms.
    """
    device = ens.device
    p_hat = (ens >= float(threshold)).float().mean(dim=0)  # [H,W]
    o = (obs.to(device) >= float(threshold)).float()       # [H,W]

    if mask is not None:
        mask = mask.to(device)
        if mask.dtype != torch.bool:
            mask = mask > 0.5
        p_hat = p_hat[mask]
        o = o[mask]
        cov = lr_covariate[mask] if (lr_covariate is not None) else None
    else:
        cov = lr_covariate

    if cov is not None:
        cov = cov.to(device).float().view(-1)
        q = torch.linspace(0, 1, n_bins + 1, device=device)
        edges = torch.quantile(cov, q)
        edges[0]  = cov.min() - 1e-6
        edges[-1] = cov.max() + 1e-6
        which = torch.bucketize(cov, edges) - 1  # [N]
        bin_center = 0.5 * (edges[:-1] + edges[1:])
        p_src = p_hat.view(-1)
        o_src = o.view(-1)
    else:
        p_src = p_hat.view(-1)
        o_src = o.view(-1)
        edges = torch.linspace(0, 1, n_bins + 1, device=device)
        which = torch.bucketize(p_src, edges) - 1
        bin_center = 0.5 * (edges[:-1] + edges[1:])

    prob_pred, freq_obs, count = [], [], []
    for b in range(n_bins):
        sel = (which == b)
        n = int(sel.sum().item())
        count.append(n)
        if n == 0:
            prob_pred.append(0.0); freq_obs.append(0.0); continue
        prob_pred.append(p_src[sel].mean().item())
        freq_obs.append(o_src[sel].mean().item())

    out = {
        "bin_center": bin_center.detach().cpu(),
        "prob_pred": torch.tensor(prob_pred, dtype=torch.float32),
        "freq_obs": torch.tensor(freq_obs, dtype=torch.float32),
        "count": torch.tensor(count, dtype=torch.int64),
    }

    if return_brier:
        o_bar = float(o_src.mean().item()) if o_src.numel() else 0.0
        N = max(int(o_src.numel()), 1)
        rel = 0.0
        res = 0.0
        for k in range(n_bins):
            Nk = int(out["count"][k].item())
            if Nk == 0: continue
            pk = float(out["prob_pred"][k].item())
            ok = float(out["freq_obs"][k].item())
            w = Nk / N
            rel += w * (pk - ok) ** 2
            res += w * (ok - o_bar) ** 2
        unc = o_bar * (1.0 - o_bar)
        out.update({
            "brier": torch.tensor(rel - res + unc, dtype=torch.float32),
            "reliability": torch.tensor(rel, dtype=torch.float32),
            "resolution": torch.tensor(res, dtype=torch.float32),
            "uncertainty": torch.tensor(unc, dtype=torch.float32),
        })

    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.cpu()
    return out


def spread_skill(
    obs: torch.Tensor,             # [H,W]
    ens: torch.Tensor,             # [M,H,W]
    point: str = "pmm",            # 'pmm' | 'mean' | 'median'
    mask: Optional[torch.Tensor] = None,
    n_bins: int = 10,
) -> Dict[str, torch.Tensor]:
    """
    Spread–skill diagnostic:
      spread = ensemble std per pixel
      skill  = |point_estimate - obs| per pixel
    We bin by spread (quantile bins) to assess calibration of spread.
    """
    from sbgm.evaluate_sbgm.metrics_univariate import pmm_from_ensemble

    device = ens.device
    obs = obs.to(device).to(ens.dtype)
    if mask is not None:
        mask = (mask.to(device) > 0.5)

    spread = ens.std(dim=0)  # [H,W]
    if point == "mean":
        pt = ens.mean(dim=0)
    elif point == "median":
        pt = ens.median(dim=0).values
    else:
        pt = pmm_from_ensemble(ens.unsqueeze(0)).squeeze(0).squeeze(0)  # [H,W]
    ae = (pt - obs).abs()

    if mask is not None:
        spread = spread[mask]
        ae = ae[mask]

    q = torch.linspace(0, 1, n_bins + 1, device=device)
    edges = torch.quantile(spread, q)
    edges[0]  = spread.min() - 1e-9
    edges[-1] = spread.max() + 1e-9
    which = torch.bucketize(spread, edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])

    sp_mean, sk_mean, count = [], [], []
    for b in range(n_bins):
        sel = (which == b)
        n = int(sel.sum().item())
        count.append(n)
        if n == 0:
            sp_mean.append(0.0); sk_mean.append(0.0); continue
        sp_mean.append(spread[sel].mean().item())
        sk_mean.append(ae[sel].mean().item())

    return {
        "bin_center": centers.detach().cpu(),
        "spread": torch.tensor(sp_mean, dtype=torch.float32),
        "skill": torch.tensor(sk_mean, dtype=torch.float32),
        "count": torch.tensor(count, dtype=torch.int64),
    }



def compute_isotropic_psd(
    batch: torch.Tensor,        # [B,1,H,W]
    dx_km: float,
    mask: Optional[torch.Tensor] = None,  # [B,1,H,W] or broadcastable
) -> Dict[str, torch.Tensor]:
    """
    Isotropic (radially averaged) 2D PSD for a batch.
    Returns: {'k': [n_bins], 'psd': [n_bins]} averaged over batch.
    """
    if batch.dim() != 4 or batch.shape[1] != 1:
        raise ValueError("batch must be [B,1,H,W]")
    B, _, H, W = batch.shape
    device = batch.device
    x = batch.to(torch.float32)

    if mask is not None:
        m = mask
        while m.dim() < 4: m = m.unsqueeze(0)
        if m.shape[0] == 1 and B > 1: m = m.expand(B, -1, -1, -1)
        x = x.masked_fill(~m.bool(), 0.0)

    X = torch.fft.rfft2(x, norm='ortho')           # [B,1,H,W//2+1]
    P = (X.real**2 + X.imag**2).mean(dim=0).squeeze(0)  # [H, W//2+1]

    ky = torch.fft.fftfreq(H, d=dx_km, device=device)   # 1/km
    kx = torch.fft.rfftfreq(W, d=dx_km, device=device)
    Ky, Kx = torch.meshgrid(ky, kx, indexing='ij')
    Kr = torch.sqrt(Kx**2 + Ky**2)                      # [H, W//2+1]

    kr = Kr.flatten()
    p  = P.flatten()
    kmax = kr.max().item()
    n_bins = int(min(H, W) // 2)
    edges = torch.linspace(0, kmax, n_bins + 1, device=device)
    which = torch.bucketize(kr, edges) - 1

    psd = torch.zeros(n_bins, device=device)
    for i in range(n_bins):
        sel = (which == i)
        if sel.any(): psd[i] = p[sel].mean()

    k_centers = 0.5 * (edges[:-1] + edges[1:])
    return {"k": k_centers.detach().cpu(), "psd": psd.detach().cpu()}
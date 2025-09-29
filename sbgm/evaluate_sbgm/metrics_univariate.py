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

def rxk_series(
    y_daily: np.ndarray, 
    k: int = 1, 
    block_index: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute block maxima of k-day accumulations.
    Args:
      y_daily: [T] daily precip (mm)
      k: 1 for Rx1day, 5 for Rx5day (rolling sum)
      block_index: [T] integer labels for blocks (e.g., year id or season-year id).
                   If None, treat entire series as one block (not typical for GEV).
    Returns:
      maxima_per_block: array of maxima (one per unique block)
      blocks: array of unique block ids aligned to maxima
    """
    T = y_daily.shape[0]
    if block_index is None:
        block_index = np.zeros(T, dtype=int)
    blocks = np.unique(block_index)
    out = []
    out_blocks = []
    for b in blocks:
        mask = (block_index == b)
        xb = y_daily[mask]
        if xb.size == 0:
            continue
        roll = _rolling_sum_np(xb, k)
        out.append(np.max(roll))
        out_blocks.append(b)
    return np.array(out, dtype=float), np.array(out_blocks)

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
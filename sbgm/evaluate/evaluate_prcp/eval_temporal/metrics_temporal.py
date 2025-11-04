from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

# ---------- helpers ----------

def _to_hw(t: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    t = t.float()
    if t.dim() == 4 and t.shape[:2] == (1,1):
        t = t.squeeze(0).squeeze(0)
    elif t.dim() == 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.dim() != 2:
        t = t.reshape(t.shape[-2], t.shape[-1])
    return t

def _apply_mask(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x
    m = mask
    if m.dtype != torch.bool: m = m > 0.5
    if m.dim() == 4 and m.shape[:2] == (1,1): m = m.squeeze(0).squeeze(0)
    elif m.dim() == 3 and m.shape[0] == 1:    m = m.squeeze(0)
    x = x.clone()
    x[~m] = float("nan")
    return x

def _nanmean_hw(x: torch.Tensor) -> float:
    x_np = x.detach().cpu().numpy()
    return float(np.nanmean(x_np))

def _nanstd_hw(x: torch.Tensor) -> float:
    x_np = x.detach().cpu().numpy()
    return float(np.nanstd(x_np))

# ---------- build daily domain-mean series ----------

def build_domain_mean_series(
    dates: Sequence[str],
    resolver,
    *,
    use_mask: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Returns dict with keys present among {"HR","PMM","LR"} mapping to
    daily domain-mean series (np.ndarray, shape [T]) aligned to 'dates'.
    Missing days are NaN.
    """
    out: Dict[str, List[float]] = {"HR": [], "PMM": [], "LR": []}
    have = {"HR": False, "PMM": False, "LR": False}

    for d in dates:
        m = resolver.load_mask(d) if use_mask else None

        hr = resolver.load_obs(d)
        gen = resolver.load_pmm(d)
        lr  = resolver.load_lr(d)

        if hr is not None:
            hr = _apply_mask(_to_hw(hr), m)
            out["HR"].append(_nanmean_hw(hr)); have["HR"] = True
        else:
            out["HR"].append(np.nan)

        if gen is not None:
            gen = _apply_mask(_to_hw(gen), m)
            out["PMM"].append(_nanmean_hw(gen)); have["PMM"] = True
        else:
            out["PMM"].append(np.nan)

        if lr is not None:
            lr = _apply_mask(_to_hw(lr), m)
            out["LR"].append(_nanmean_hw(lr)); have["LR"] = True
        else:
            out["LR"].append(np.nan)

    out_np: Dict[str, np.ndarray] = {}
    for k in ["HR","PMM","LR"]:
        if have[k]:
            out_np[k] = np.asarray(out[k], dtype=np.float32)
    return out_np

# ---------- autocorrelation ----------

def autocorr(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Lag-1..max_lag autocorrelation with NaN-handling (pairwise complete)."""
    s = np.asarray(series, dtype=np.float64)
    # de-mean over finite entries
    mask = np.isfinite(s)
    if mask.sum() < 2:
        return np.full(max_lag, np.nan)
    mu = np.nanmean(s)
    s = s - mu
    var = np.nanvar(s)
    if not np.isfinite(var) or var == 0.0:
        return np.full(max_lag, np.nan)
    ac = np.full(max_lag, np.nan)
    for k in range(1, max_lag+1):
        x = s[:-k]; y = s[k:]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2:
            ac[k-1] = np.nan
        else:
            ac[k-1] = float(np.dot(x[m], y[m]) / (m.sum() * var))
    return ac

# ---------- wet/dry spell (Markov) ----------

def binarize_wet(series: np.ndarray, wet_thr_mm: float) -> np.ndarray:
    b = np.asarray(series, dtype=np.float64)
    return (b >= wet_thr_mm).astype(np.int8)  # 1=wet, 0=dry

def transition_matrix(b: np.ndarray) -> np.ndarray:
    """Return 2x2 matrix P where P[i,j] = P(state_t=j | state_{t-1}=i)."""
    P = np.zeros((2,2), dtype=np.float64)
    valid = np.isfinite(b)
    x = b[valid].astype(np.int32)
    if x.size < 2:
        return np.full((2,2), np.nan)
    for i in (0,1):
        idx = np.where(x[:-1] == i)[0]
        if idx.size == 0:
            P[i,:] = np.nan
            continue
        nxt = x[idx+1]
        P[i,0] = np.mean(nxt == 0) if idx.size > 0 else np.nan
        P[i,1] = np.mean(nxt == 1) if idx.size > 0 else np.nan
    return P

def spell_lengths(b: np.ndarray, state: int) -> np.ndarray:
    """Run-lengths for 'state' (0=dry, 1=wet)."""
    x = b.astype(np.int8)
    if x.size == 0:
        return np.zeros(0, dtype=np.int32)
    runs: List[int] = []
    cur = 0
    for v in x:
        if v == state:
            cur += 1
        else:
            if cur > 0: runs.append(cur)
            cur = 0
    if cur > 0: runs.append(cur)
    return np.asarray(runs, dtype=np.int32)

def spell_histogram(lengths: np.ndarray, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (bins, pmf) where bins = 1..max_len and pmf sums to 1 over observed lengths."""
    if lengths.size == 0:
        bins = np.arange(1, max_len+1, dtype=np.int32)
        return bins, np.full_like(bins, np.nan, dtype=np.float64)
    bins = np.arange(1, max_len+1, dtype=np.int32)
    counts = np.zeros_like(bins, dtype=np.float64)
    for L in lengths:
        if 1 <= L <= max_len:
            counts[L-1] += 1.0
    if counts.sum() == 0:
        pmf = np.full_like(counts, np.nan)
    else:
        pmf = counts / counts.sum()
    return bins, pmf

def geometric_fit_from_P(P: np.ndarray, state: int) -> float:
    """
    Geometric parameter p for spell of 'state' given transition matrix.
    For wet spells (state=1): p = 1 - P[1,1]; for dry (state=0): p = 1 - P[0,0].
    """
    if not np.all(np.isfinite(P)):
        return np.nan
    stay = P[state, state]
    if not np.isfinite(stay):
        return np.nan
    p = 1.0 - stay
    return float(p) if 0.0 < p <= 1.0 else np.nan

def geometric_pmf(bins: np.ndarray, p: float) -> np.ndarray:
    """PMF of geometric distribution on {1,2,...} with parameter p."""
    if not np.isfinite(p) or p <= 0.0 or p > 1.0:
        return np.full_like(bins, np.nan, dtype=np.float64)
    return (1.0 - (1.0 - p) ** (bins-1)) * p / p  # = p*(1-p)^{k-1}

# ---------- package ----------

def compute_temporal_metrics(
    series_dict: Dict[str, np.ndarray],
    *,
    wet_thr_mm: float = 1.0,
    max_lag: int = 30,
    max_spell: int = 30,
) -> Dict[str, dict]:
    """
    Returns nested dict keyed by {"HR","PMM","LR"} with:
      - "autocorr": [max_lag] array
      - "P": 2x2 transition matrix
      - "wet_bins","wet_pmf","wet_geom_p"
      - "dry_bins","dry_pmf","dry_geom_p"
    """
    out: Dict[str, dict] = {}
    for k, s in series_dict.items():
        d: dict = {}
        d["autocorr"] = autocorr(s, max_lag)

        b = binarize_wet(s, wet_thr_mm)
        P = transition_matrix(b)
        d["P"] = P

        # spells
        wet_L = spell_lengths(b, 1)
        dry_L = spell_lengths(b, 0)
        wb, wpmf = spell_histogram(wet_L, max_spell)
        db, dpmf = spell_histogram(dry_L, max_spell)
        d["wet_bins"], d["wet_pmf"] = wb, wpmf
        d["dry_bins"], d["dry_pmf"] = db, dpmf

        pw = geometric_fit_from_P(P, 1)
        pd = geometric_fit_from_P(P, 0)
        d["wet_geom_p"], d["dry_geom_p"] = pw, pd
        out[k] = d
    return out
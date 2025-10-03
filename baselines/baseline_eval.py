"""
Baseline evaluation runner that reuses the *same* metric implementations as the main EDM evaluation:
  - P95/P99 + wet-day frequency (compute_p95_p99_and_wet_day)
  - FSS at user-defined thresholds/scales (compute_fss_at_scales)
  - PSD + PSD slope (compute_psd_slope)
  - Reliability (reliability_exceedance_lr_binned)

Inputs: reads already-generated baseline artifacts:
  ${paths.evaluation_dir}/baselines/{experiment_name}/generated_samples/{pmm, lr_hr, lsm}/DATE.npz

Outputs (mirrors evaluation.py table layout):
  ${paths.evaluation_dir}/baselines/{experiment_name}/evaluation/{split}/tables/
    - fss_summary.csv
    - psd_slope_summary.json
    - tails_summary.json
    - tails_summary_flat.csv
    - reliability_bins.csv
  ${...}/figures/
    - (optional) PSD plots if plot_psd_curves is available
"""

from __future__ import annotations
import json, csv, logging
from pathlib import Path
from typing import Optional, Iterable, Dict

import numpy as np
import torch

# --- use the user's canonical implementations ---
from sbgm.evaluate_sbgm.plot_utils import (
    plot_psd_curves,   # optional; will be tried in a try/except
)
from sbgm.monitoring import (
    compute_fss_at_scales,        # gen_bt/hr_bt: [B,1,H,W]
    compute_psd_slope,            # returns {'psd_slope_gen', 'psd_slope_hr', 'psd_slope_delta'}
    compute_p95_p99_and_wet_day,  # tails dict
)
from sbgm.evaluate_sbgm.metrics_univariate import (
    reliability_exceedance_lr_binned,  # obs [H,W], ens [M,H,W]
)

logger = logging.getLogger(__name__)

# ---------------- I/O helpers ----------------
def _list_dates(base_dir: Path) -> Iterable[str]:
    dates = sorted([f.stem for f in (base_dir / 'pmm').glob("*.npz")])
    logger.info("[baseline_eval] Found %d date files under %s", len(dates), base_dir / 'pmm')
    return dates

def _load_npz(folder: Path, date: str, key: str):
    p = folder / f"{date}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return d.get(key, None)

def _load_obs(dir_lrhr: Path, date: str) -> Optional[torch.Tensor]:
    x = _load_npz(dir_lrhr, date, "hr")
    if x is None:
        return None
    t = torch.from_numpy(x).squeeze(0)  # [1,1,H,W] -> [1,H,W]
    return t.squeeze(0)                 # -> [H,W]

def _load_pmm(dir_pmm: Path, date: str) -> Optional[torch.Tensor]:
    x = _load_npz(dir_pmm, date, "pmm")
    if x is None:
        return None
    t = torch.from_numpy(x).squeeze(0)  # [1,1,H,W] -> [1,H,W]
    return t.squeeze(0)                 # -> [H,W]

def _load_mask(dir_lsm: Path, date: str) -> Optional[torch.Tensor]:
    p = dir_lsm / f"{date}.npz"
    if not p.exists():
        return None
    try:
        arr = np.load(p, allow_pickle=True).get('lsm', None) or np.load(p, allow_pickle=True).get('lsm_hr', None)
        if arr is None:
            return None
        m = torch.from_numpy(arr).to(torch.bool)
        # normalize to [H,W]
        if m.dim() == 4 and m.shape[:2] == (1, 1):
            m = m.squeeze(0).squeeze(0)
        elif m.dim() == 3 and m.shape[0] == 1:
            m = m.squeeze(0)
        return m
    except Exception as e:
        logger.warning(f"[baseline_eval] Failed loading mask for {date}: {e}")
        return None

# ---------------- main evaluation ----------------
def evaluate_baseline(cfg: dict, baseline_type: str, split: str):
    """
    Compute capability metrics for a baseline using the same methods as the EDM evaluation.
    """
    run_name = cfg.get('baseline', {}).get('experiment_name', f'{baseline_type}_baseline')

    base_root = Path(cfg['paths']['sample_dir']) / 'baselines' / run_name
    gen_root  = base_root / 'generated_samples'
    out_root  = base_root / 'evaluation' / split
    tables_dir = out_root / 'tables'
    figs_dir   = out_root / 'figures'
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # knobs (align with evaluation.py defaults)
    grid_km = float(cfg.get('baseline', {}).get('eval', {}).get('grid_km_per_px',
                                                                cfg.get('evaluation', {}).get('grid_km_per_px', 2.5)))
    fss_scales_km = tuple(cfg.get('baseline', {}).get('eval', {}).get('fss_scales_km',
                                cfg.get('evaluation', {}).get('fss_scales_km', (5,10,20))))
    thresholds_mm = tuple(cfg.get('baseline', {}).get('eval', {}).get('thresholds_mm',
                                cfg.get('evaluation', {}).get('thresholds_mm', (1.0,5.0,10.0))))
    wet_thr_mm = float(cfg.get('baseline', {}).get('eval', {}).get('wet_threshold_mm',
                                cfg.get('evaluation', {}).get('wet_threshold_mm', 1.0)))
    rel_bins = int(cfg.get('baseline', {}).get('eval', {}).get('reliability_bins',
                                cfg.get('evaluation', {}).get('reliability_bins', 10)))
    psd_ignore_low_k = int(cfg.get('baseline', {}).get('eval', {}).get('psd_ignore_low_k_bins',
                                cfg.get('evaluation', {}).get('psd_ignore_low_k_bins', 1)))

    # dirs inside gen_root
    dir_pmm  = gen_root / 'pmm'
    dir_lrhr = gen_root / 'lr_hr'
    dir_lsm  = gen_root / 'lsm'

    dates = list(_list_dates(gen_root))
    if not dates:
        logger.warning("[baseline_eval] No dates found. Nothing to evaluate.")
        return {"ok": False, "msg": "no_dates"}

    # ---- stack PMM & HR for capability metrics on [B,1,H,W] ----
    PMM_bt, HR_bt, MASK_bt, dates_used = [], [], [], []
    for d in dates:
        pmm = _load_pmm(dir_pmm, d)      # [H,W]
        hr  = _load_obs(dir_lrhr, d)     # [H,W] (may be None on blind test)
        m   = _load_mask(dir_lsm, d)     # [H,W] or None

        if pmm is None or hr is None:
            logger.debug(f"[baseline_eval] Missing pmm/hr for {d}; skipping in capability stacks.")
            continue
        PMM_bt.append(pmm.unsqueeze(0).unsqueeze(0))
        HR_bt.append(hr.unsqueeze(0).unsqueeze(0))
        if m is not None:
            MASK_bt.append(m.unsqueeze(0).unsqueeze(0))
        dates_used.append(d)

    if len(PMM_bt) == 0:
        logger.warning("[baseline_eval] No PMM/HR pairs to evaluate.")
        return {"ok": False, "msg": "no_pairs"}

    pmm_bt = torch.cat(PMM_bt, 0).nan_to_num(nan=0.0, posinf=None, neginf=0.0).clamp_min(0.0)
    hr_bt  = torch.cat(HR_bt,  0).nan_to_num(nan=0.0, posinf=None, neginf=0.0).clamp_min(0.0)
    mask_bt: Optional[torch.Tensor] = torch.cat(MASK_bt, 0) if len(MASK_bt) == len(PMM_bt) else None

    logger.info("[baseline_eval] Capability stack: pmm_bt=%s hr_bt=%s mask=%s",
                tuple(pmm_bt.shape), tuple(hr_bt.shape), None if mask_bt is None else tuple(mask_bt.shape))

    # ---------- FSS ----------
    fss_rows = []
    for thr in thresholds_mm:
        scores = compute_fss_at_scales(
            gen_bt=pmm_bt, hr_bt=hr_bt, mask=mask_bt,
            grid_km_per_px=grid_km, fss_km=list(fss_scales_km),
            thr_mm=float(thr)
        )
        row = {"thr": float(thr)}
        for km in fss_scales_km:
            key = f"{int(km)}km"
            val = scores.get(key, np.nan) if isinstance(scores, dict) else np.nan
            if isinstance(val, torch.Tensor):
                val = float(val.item()) if val.numel() == 1 else float(val.mean().item())
            row[f"FSS_{int(km)}km"] = float(val)
        fss_rows.append(row)
    # write CSV
    if fss_rows:
        header = list(fss_rows[0].keys())
        with open(tables_dir / "fss_summary.csv", 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=header); w.writeheader()
            for r in fss_rows: w.writerow(r)
        logger.info("[baseline_eval] Wrote FSS → %s", tables_dir / "fss_summary.csv")

    # ---------- PSD slope (+ optional curves) ----------
    psd_summ = compute_psd_slope(
        gen_bt=pmm_bt, hr_bt=hr_bt, mask=mask_bt, ignore_low_k_bins=psd_ignore_low_k
    )
    (tables_dir / "psd_slope_summary.json").write_text(json.dumps(psd_summ, indent=2))
    logger.info("[baseline_eval] Wrote PSD slope → %s", tables_dir / "psd_slope_summary.json")
    try:
        plot_psd_curves(pmm_bt, hr_bt, mask=mask_bt,
                        dx_km=grid_km, seasons=("ALL",), out_dir=str(figs_dir))
    except Exception as e:
        logger.warning(f"[baseline_eval] plot_psd_curves failed: {e}")

    # ---------- P95/P99 & wet-day frequency ----------
    tails = compute_p95_p99_and_wet_day(
        gen_bt=pmm_bt, hr_bt=hr_bt, mask=mask_bt, wet_threshold_mm=wet_thr_mm
    )
    def _py(v):
        import numpy as _np
        if isinstance(v, torch.Tensor): return float(v.detach().cpu().item()) if v.numel()==1 else [float(x) for x in v.detach().cpu().flatten()]
        if isinstance(v, _np.generic):  return float(v.item())
        return v
    tails_py = {k: _py(v) for k, v in (tails or {}).items()}
    (tables_dir / "tails_summary.json").write_text(json.dumps(tails_py, indent=2))
    def _first_float(val):
        if isinstance(val, list):
            return float(val[0]) if val else float('nan')
        try:
            return float(val)
        except Exception:
            return float('nan')

    row = {
        "P95":        _first_float(tails_py.get("gen_p95", np.nan)),
        "P99":        _first_float(tails_py.get("gen_p99", np.nan)),
        "WetDayFreq": _first_float(tails_py.get("gen_wet_freq", np.nan)),
        "HR_P95":     _first_float(tails_py.get("hr_p95", np.nan)),
        "HR_P99":     _first_float(tails_py.get("hr_p99", np.nan)),
        "HR_WetDayFreq": _first_float(tails_py.get("hr_wet_freq", np.nan)),
    }
    with open(tables_dir / "tails_summary_flat.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys())); w.writeheader(); w.writerow(row)
    logger.info("[baseline_eval] Wrote tails → %s ; flat CSV → %s",
                tables_dir / "tails_summary.json", tables_dir / "tails_summary_flat.csv")

    # ---------- Reliability ----------
    rows_rel = []
    for d in dates:
        obs = _load_obs(dir_lrhr, d)   # [H,W]
        pmm = _load_pmm(dir_pmm,  d)   # [H,W]
        if obs is None or pmm is None:
            continue
        m = _load_mask(dir_lsm, d)
        ens = pmm.unsqueeze(0)  # [1,H,W]
        for thr in thresholds_mm:
            rel = reliability_exceedance_lr_binned(
                obs=obs, ens=ens, threshold=float(thr),
                lr_covariate=None, n_bins=int(rel_bins), mask=m, return_brier=True
            )
            bc   = rel.get("bin_center", [])
            pp   = rel.get("prob_pred", [])
            fobs = rel.get("freq_obs", [])
            cnt  = rel.get("count", [])
            L = min(len(bc), len(pp), len(fobs), len(cnt))
            for i in range(L):
                def _f(x):
                    if torch.is_tensor(x): return float(x.detach().cpu().item())
                    try: return float(x)
                    except: return float('nan')
                def _i(x):
                    if torch.is_tensor(x): return int(x.detach().cpu().item())
                    try: return int(x)
                    except: return 0
                rows_rel.append({
                    "date": d,
                    "thr": float(thr),
                    "bin_center": _f(bc[i]),
                    "prob_pred":  _f(pp[i]),
                    "freq_obs":   _f(fobs[i]),
                    "count":      _i(cnt[i]),
                })
    if rows_rel:
        header = list(rows_rel[0].keys())
        with open(tables_dir / "reliability_bins.csv", 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=header); w.writeheader()
            for r in rows_rel: w.writerow(r)
        logger.info("[baseline_eval] Wrote Reliability → %s", tables_dir / "reliability_bins.csv")

    # Provenance
    meta = dict(
        baseline=baseline_type,
        split=split,
        dates=dates,
        n_dates=len(dates),
        config=dict(
            grid_km_per_px=grid_km,
            fss_scales_km=list(fss_scales_km),
            thresholds_mm=list(thresholds_mm),
            wet_threshold_mm=wet_thr_mm,
            reliability_bins=int(rel_bins),
            psd_ignore_low_k_bins=int(psd_ignore_low_k),
        ),
        sources=dict(generated_samples=str(gen_root)),
    )
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("[baseline_eval] wrote meta.json → %s", out_root / "meta.json")
    return {"ok": True, "out_root": str(out_root)}

def run_all(cfg):
    btype  = cfg.get('baseline', {}).get('type', 'all').lower()
    splits = cfg.get('baseline', {}).get('eval', {}).get('splits', ['test'])
    types  = ['bilinear','qm','unet_sr'] if btype in {'all','auto','everything'} else [btype]
    results = {}
    for t in types:
        for sp in splits:
            logger.info(f"[baseline_eval] === {t} | {sp} ===")
            results[(t,sp)] = evaluate_baseline(cfg, t, sp)
    return results
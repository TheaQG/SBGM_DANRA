"""
    Evaluation runner for EDM downscaling (univariate precipitation)

    Reads generation artifacts (prefer physical-space) and computes:
        A) Probabilistic performance (per-day, ensemble):
            - CRPS (Continuous Ranked Probability Score),  mean over pixels/masks
            - Reliability for exceedance (>= 1/5/10 mm/day), LR-binned optional
            - Spread-skill (using PMM as point estimator)
            - PIT (Probability Integral Transform) histograms and rank histograms
        B) Capability (across all days, using daily PMM fields):
            - FSS at 1/5/10 mm and 5/10/20 km scales
            - PSD slope + full PSD curves (with LR_ups reference)
            - P95/P99 + wet-day frequency
        C) Extremes (basin-mean daily series):
            - GEV first for Rx1day/Rx5day with bootstrap CIs
            - POT/GPD over a threshold with bootstrap CIs
    Outputs tables (JSON/CSV) and figures to <eval_out_root>/tables and <eval_out_root>/figures
"""

from __future__ import annotations
import json
import logging
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, Dict, Any

import numpy as np
import torch

# --- project utils you already have ---
from sbgm.utils import get_model_string
# from sbgm.training_utils import load_land_mask_if_any  # if you have something similar; else set mask=None

# --- metrics we already discussed/you implemented ---
from sbgm.evaluate_sbgm.plot_utils import (
    plot_psd_curves,                       # plotting util if present
)
from sbgm.monitoring import (
    compute_fss_at_scales,                 # on batches [B,1,H,W]
    compute_psd_slope,                     # returns dict/slopes
    compute_p95_p99_and_wet_day,           # returns dict
)
from sbgm.evaluate_sbgm.metrics_univariate import (
    crps_ensemble,                         # obs [H,W], ens [M,H,W]
    reliability_exceedance_lr_binned,      # returns per-bin stats
    spread_skill,                          # returns bin stats
    pit_values_from_ensemble,
    rank_histogram,
    to_numpy_1d_series,
    seasonal_block_index,
    rxk_series,
    fit_gev_block_maxima_with_ci,
    fit_pot_gpd_with_ci,
)

logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    gen_dir: str                     # Root directory where generated samples are stored 
    out_dir: str                     # Root directory to save evaluation outputs
    grid_km_per_px: float = 2.5      # Spatial resolution of the data in km/px
    fss_scales_km: tuple = (5, 10, 20)  # Scales (in km) at which to compute FSS
    thresholds_mm: tuple = (1.0, 5.0, 10.0)  # Thresholds (in mm/day) for exceedance reliability and FSS
    wet_threshold_mm: float = 1.0  # Threshold (in mm/day) to define wet days for P95/P99
    reliability_bins: int = 10  # Number of bins for reliability diagrams
    spread_skill_bins: int = 10  # Number of bins for spread-skill analysis
    pit_bins: int = 10  # Number of bins for PIT histograms
    psd_ignore_low_k_bins: int = 1  # Number of lowest k bins to ignore in PSD slope fitting
    random_ref_kind: str = "phase_randomized"  # Kind of random reference for FSS, "iid_marginal" | "spatial_shuffle" | "phase_randomized"
    seasons: tuple = ("ALL", "DJF", "MAM", "JJA", "SON")  # Seasons to consider for seasonal analysis
    seed: int = 504                  # Random seed for reproducibility


class EvaluationRunner:
    def __init__(self, cfg_yaml: dict, eval_cfg: EvaluationConfig, device: torch.device, mask: Optional[torch.Tensor] = None):
        self.cfg_yaml = cfg_yaml
        self.eval_cfg = eval_cfg
        self.device = device
        self.mask = mask  # [H,W] bool or None

        self.gen_root = Path(eval_cfg.gen_dir)
        # Prefer physical-space outputs; fall back to model-space if needed
        self.dir_ens_phys = self.gen_root / "ensembles_phys"
        self.dir_pmm_phys = self.gen_root / "pmm_phys"
        self.dir_lrhr_phys = self.gen_root / "lr_hr_phys"

        self.dir_ens_model = self.gen_root / "ensembles"
        self.dir_pmm_model = self.gen_root / "pmm"

        # Land/sea masks
        self.dir_lsm = self.gen_root / 'lsm'
        self.mask_global: Optional[torch.Tensor] = None  # [H,W] bool
        try:
            p = self.gen_root / 'meta' / 'land_mask.npz'
            if p.exists():
                arr = np.load(p, allow_pickle=True).get('lsm_hr', None)
                if arr is not None:
                    self.mask_global = torch.from_numpy(arr).to(torch.bool)
        except Exception as e:
            logger.warning(f"[eval] Could not load global land mask: {e}")

        self.out_root = Path(eval_cfg.out_dir)
        (self.out_root / "tables").mkdir(parents=True, exist_ok=True)
        (self.out_root / "figures").mkdir(parents=True, exist_ok=True)


    # ---------- I/O helpers --------
    def _list_dates(self) -> Iterable[str]:
        base = self.dir_pmm_phys if self.dir_pmm_phys.exists() else self.dir_pmm_model
        dates = sorted([f.stem for f in base.glob("*.npz")])
        logger.info("[eval] Found %d date files under %s", len(dates), base)
        return dates

    def _load_npz(self, folder: Path, date: str, key: str):
        p = folder / f"{date}.npz"
        if not p.exists(): return None
        d = np.load(p, allow_pickle=True)
        return d.get(key, None)

    def _load_obs(self, date: str) -> Optional[torch.Tensor]:
        # prefer physical HR if available
        x = self._load_npz(self.dir_lrhr_phys, date, "hr")
        if x is None:
            # fallback to model-space HR is not ideal; if needed, you can apply back-transform here.
            x = self._load_npz(self.gen_root / "lr_hr", date, "hr")
        if x is None: return None
        t = torch.from_numpy(x).squeeze(0)  # [1,1,H,W] -> [1,H,W] -> squeeze to [H,W] below
        return t.squeeze(0)

    def _load_ens(self, date: str) -> Optional[torch.Tensor]:
        x = self._load_npz(self.dir_ens_phys, date, "ens")
        if x is None:
            x = self._load_npz(self.dir_ens_model, date, "ens")
        if x is None: return None
        t = torch.from_numpy(x).squeeze(1)  # [M,1,H,W] -> [M,H,W]
        return t

    def _load_pmm(self, date: str) -> Optional[torch.Tensor]:
        x = self._load_npz(self.dir_pmm_phys, date, "pmm")
        if x is None:
            x = self._load_npz(self.dir_pmm_model, date, "pmm")
        if x is None: return None
        t = torch.from_numpy(x).squeeze(0)  # [1,1,H,W] -> [1,H,W] -> squeeze to [H,W] below
        return t.squeeze(0)

    def _load_mask(self, date: str) -> Optional[torch.Tensor]:
        # Prefer a global canonical mask; else try per-date; else return None
        if self.mask_global is not None:
            m = self.mask_global
        else:
            p = self.dir_lsm / f"{date}.npz"
            if not p.exists():
                return None
            try:
                arr = np.load(p, allow_pickle=True).get('lsm_hr', None)
                if arr is None:
                    return None
                m = torch.from_numpy(arr).to(torch.bool)
            except Exception as e:
                logger.warning(f"[eval] Failed loading per-date mask for {date}: {e}")
                return None

        # Normalize mask to [H,W]
        if m.dim() == 4 and m.shape[:2] == (1, 1):   # [1,1,H,W]
            m = m.squeeze(0).squeeze(0)
        elif m.dim() == 3 and m.shape[0] == 1:       # [1,H,W]
            m = m.squeeze(0)
        elif m.dim() == 2:
            pass
        else:
            logger.warning("[eval] Unexpected mask shape %s; coercing last two dims as HxW.", tuple(m.shape))
            m = m.reshape(m.shape[-2], m.shape[-1])
        return m
    

    # ---------- Probabilistic metrics (per day) ----------
    def eval_probabilistic(self):
        """
        Compute per-day probabilistic metrics and save tables/figures
        Evaluates: 
            - CRPS (mean over pixels/masks)
            - Reliability for exceedance by thresholds
            - Spread-skill (using PMM as point)
            - PIT histograms and rank histograms
        Outputs CSV tables and NPZ files for PIT/rank histograms
        """
        tables_dir = self.out_root / "tables"
        figs_dir = self.out_root / "figures"

        rows_crps = []
        rows_rel = []
        rows_ss = []

        pit_values_all = []
        rank_counts = None

        for date in self._list_dates():
            obs = self._load_obs(date)      # [H,W] or None
            ens = self._load_ens(date)      # [M,H,W] or None
            if obs is None or ens is None: 
                logger.warning(f"[eval] Missing obs/ens for {date}, skipping.")
                continue

            # Prefer mask saved during generation (stationary canonical or per-date),
            # fall back to user-provided mask.
            m = self._load_mask(date)
            mask = m if (m is not None) else self.mask

            # CRPS (mean over masked pixels)
            crps_val = crps_ensemble(obs, ens, mask=mask, reduction="mean")
            rows_crps.append({"date": date, "crps": float(crps_val)})

            # Reliability for exceedance by thresholds — explode per-bin vectors into scalar rows
            for thr in self.eval_cfg.thresholds_mm:
                rel = reliability_exceedance_lr_binned(
                    obs=obs, ens=ens, threshold=float(thr),
                    lr_covariate=None,  # or pass basin-mean LR if you like
                    n_bins=int(self.eval_cfg.reliability_bins),
                    mask=mask, return_brier=True,
                )
                bc   = rel.get("bin_center", [])
                pp   = rel.get("prob_pred", [])
                fobs = rel.get("freq_obs", [])
                cnt  = rel.get("count", [])
                L = min(len(bc), len(pp), len(fobs), len(cnt))

                added = 0
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
                        "date": date,
                        "thr": float(thr),
                        "bin_center": _f(bc[i]),
                        "prob_pred":  _f(pp[i]),
                        "freq_obs":   _f(fobs[i]),
                        "count":      _i(cnt[i]),
                    })
                    added += 1
                logger.info("[eval] reliability: date=%s thr=%.1f → %d rows", date, float(thr), added)

            # Spread–skill (using PMM as point) — explode per-bin vectors into scalar rows
            ss = spread_skill(obs=obs, ens=ens, point="pmm", mask=mask,
                            n_bins=int(self.eval_cfg.spread_skill_bins))
            bc  = ss.get("bin_center", [])
            spr = ss.get("spread", [])
            skl = ss.get("skill", [])
            cnt = ss.get("count", [])
            L = min(len(bc), len(spr), len(skl), len(cnt))

            added = 0
            for i in range(L):
                def _f(x):
                    if torch.is_tensor(x): return float(x.detach().cpu().item())
                    try: return float(x)
                    except: return float('nan')
                def _i(x):
                    if torch.is_tensor(x): return int(x.detach().cpu().item())
                    try: return int(x)
                    except: return 0

                rows_ss.append({
                    "date": date,
                    "bin_center": _f(bc[i]),
                    "spread":     _f(spr[i]),
                    "skill":      _f(skl[i]),
                    "count":      _i(cnt[i]),
                })
                added += 1
            logger.info("[eval] spread–skill: date=%s → %d rows", date, added)


            # PIT & rank hist (accumulate) — expected by metrics: obs [B,H,W], ens [B,M,H,W]
            obs_bhw  = obs.unsqueeze(0)                   # [1,H,W]
            ens_bmhw = ens.unsqueeze(0)                   # [1,M,H,W]

            # Make mask [B,H,W] (B=1), squeeze any channel dim if present
            mask_bhw = None
            if mask is not None:
                m = mask
                if m.dim() == 4 and m.shape[1] == 1:      # [B,1,H,W] -> [B,H,W]
                    m = m.squeeze(1)
                elif m.dim() == 3 and m.shape[0] != 1:    # [H,W,?] unexpected; fall back to [1,H,W]
                    m = m[:1]
                elif m.dim() == 2:                        # [H,W] -> [1,H,W]
                    m = m.unsqueeze(0)
                mask_bhw = m

            pits = pit_values_from_ensemble(obs_bhw, ens_bmhw, mask=mask_bhw)  # 1-D tensor
            pit_values_all.append(pits.cpu())
            rh = rank_histogram(obs_bhw, ens_bmhw, mask=mask_bhw)              # [M+1]
            rank_counts = rh if rank_counts is None else (rank_counts + rh)

        # --- write CSVs without pandas ---
        def _write_csv(path, rows):
            if not rows:
                open(path, 'w').close(); return
            norm_rows = []
            for r in rows:
                out = {}
                for k, v in r.items():
                    if isinstance(v, torch.Tensor):
                        out[k] = float(v.item()) if v.numel() == 1 else str(v.detach().cpu().numpy().tolist())
                    else:
                        out[k] = v
                norm_rows.append(out)
            header = list(norm_rows[0].keys())
            with open(path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=header)
                w.writeheader()
                for r in norm_rows:
                    w.writerow(r)

        _write_csv(str(tables_dir / "crps_daily.csv"), rows_crps)
        logger.info(f"[eval] Wrote CRPS table to {tables_dir / 'crps_daily.csv'}")
        _write_csv(str(tables_dir / "reliability_bins.csv"), rows_rel)
        logger.info(f"[eval] Wrote Reliability table to {tables_dir / 'reliability_bins.csv'}")
        _write_csv(str(tables_dir / "spread_skill.csv"), rows_ss)
        logger.info(f"[eval] Wrote Spread Skill table to {tables_dir / 'spread_skill.csv'}")

        # Save PIT histogram
        if len(pit_values_all) > 0:
            pits = torch.cat(pit_values_all, dim=0).numpy()
            np.savez_compressed(figs_dir / "pit_values_all.npz", pits=pits)
            logger.info("[eval] Saved PIT values → %s", figs_dir / "pit_values_all.npz")
        if rank_counts is not None:
            np.savez_compressed(figs_dir / "rank_hist_counts.npz", counts=rank_counts.cpu().numpy())
            logger.info("[eval] Saved rank histogram counts → %s", figs_dir / "rank_hist_counts.npz")



    # ---------- Capability metrics on PMM (across all days) ----------
    def eval_capability(self):
        """
        Compute capability metrics using daily PMM fields and save tables/figures
        Evaluates: 
            - FSS at 1/5/10 mm and 5/10/20 km scales
            - PSD slope + full PSD curves (with LR_ups reference)
            - P95/P99 + wet-day frequency
        Outputs tables (CSV/JSON) and figures
        """
        tables_dir = self.out_root / "tables"
        figs_dir = self.out_root / "figures"

        PMM, HR, dates_used = [], [], []
        for date in self._list_dates():
            pmm = self._load_pmm(date)   # [H,W]
            obs = self._load_obs(date)   # [H,W]
            if pmm is None or obs is None:
                continue
            PMM.append(pmm.unsqueeze(0).unsqueeze(0))  # -> [1,1,H,W]
            HR.append(obs.unsqueeze(0).unsqueeze(0))
            dates_used.append(date)

        if len(PMM) == 0:
            logger.warning("[eval] No PMM/HR pairs found.")
            return

        pmm_bt = torch.cat(PMM, dim=0)  # [B,1,H,W]
        hr_bt  = torch.cat(HR, dim=0)   # [B,1,H,W]

        # Sanitize: replace NaNs/±inf and clamp negatives to 0 for precip
        pmm_bt = torch.nan_to_num(pmm_bt, nan=0.0, posinf=None, neginf=0.0).clamp_min(0.0)
        hr_bt  = torch.nan_to_num(hr_bt,  nan=0.0, posinf=None, neginf=0.0).clamp_min(0.0)
        logger.info("[eval] Capability stack: pmm_bt=%s hr_bt=%s (after nan→num & clamp≥0)", tuple(pmm_bt.shape), tuple(hr_bt.shape))

        # Build per-sample mask batch aligned to dates_used
        mask_bt = None
        if dates_used:
            mask_list = []
            all_have_mask = True
            for d in dates_used:
                m = self._load_mask(d)
                if m is None:
                    m = self.mask
                if m is None:
                    all_have_mask = False
                    break
                # Normalize to [H,W]
                if m.dim() == 4 and m.shape[:2] == (1, 1):
                    m = m.squeeze(0).squeeze(0)
                elif m.dim() == 3 and m.shape[0] == 1:
                    m = m.squeeze(0)
                mask_list.append(m.unsqueeze(0).unsqueeze(0))  # [1,1,H,W]
            if all_have_mask and len(mask_list) == len(dates_used):
                mask_bt = torch.cat(mask_list, dim=0)
                logger.info("[eval] Using per-date mask batch → %s", tuple(mask_bt.shape))
            else:
                mask_bt = None
                logger.info("[eval] Proceeding without mask for capability (not all dates had masks).")
        # FSS for thresholds x scales
        fss_rows = []
        for thr in self.eval_cfg.thresholds_mm:
            # Basic checks
            assert pmm_bt.shape == hr_bt.shape and pmm_bt.dim() == 4 and pmm_bt.shape[1] == 1
            if mask_bt is not None:
                assert mask_bt.shape[:1] == pmm_bt.shape[:1] and mask_bt.shape[2:] == pmm_bt.shape[2:]
            
            scores = compute_fss_at_scales(gen_bt=pmm_bt, hr_bt=hr_bt, mask=mask_bt,
                                        grid_km_per_px=self.eval_cfg.grid_km_per_px,
                                        fss_km=list(self.eval_cfg.fss_scales_km),
                                        thr_mm=float(thr))
            # Normalize to columns like FSS_5km, FSS_10km, FSS_20km (what plot_utils expects)
            row = {"thr": float(thr)}
            for km in self.eval_cfg.fss_scales_km:
                # try several likely key variants that compute_fss_at_scales() might return
                key_candidates = [
                    f"FSS_{int(km)}km", f"FSS_{float(km)}km",
                    f"{int(km)}km", f"{float(km)}km",
                    f"FSS_{km}", str(km), f"k{int(km)}", f"k{float(km)}",
                ]
                val = None
                for kc in key_candidates:
                    if isinstance(scores, dict) and kc in scores:
                        val = scores[kc]
                        break
                if val is None:
                    # fallback: if scores is an ordered dict/list aligned with fss_scales_km
                    try:
                        idx = list(self.eval_cfg.fss_scales_km).index(km)
                        if isinstance(scores, dict):
                            val = list(scores.values())[idx]
                        else:
                            val = scores[idx]
                    except Exception:
                        val = float('nan')
                if isinstance(val, torch.Tensor):
                    val = float(val.item()) if val.numel() == 1 else float(val.mean().item())
                row[f"FSS_{int(km)}km"] = float(val)
            fss_rows.append(row)

        # write fss_summary.csv without pandas
        if fss_rows:
            header = list(fss_rows[0].keys())
            with open(tables_dir / "fss_summary.csv", 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=header)
                w.writeheader()
                for r in fss_rows:
                    r2 = {k: (float(v.item()) if isinstance(v, torch.Tensor) and v.numel()==1 else v) for k,v in r.items()}
                    w.writerow(r2)
        else:
            open(tables_dir / "fss_summary.csv", 'w').close()
        logger.info(f"[eval] Wrote FSS summary to {tables_dir / 'fss_summary.csv'}")


        # PSD slope & curves
        psd_summ = compute_psd_slope(gen_bt=pmm_bt, hr_bt=hr_bt, mask=mask_bt,
                                     ignore_low_k_bins=self.eval_cfg.psd_ignore_low_k_bins)
        (tables_dir / "psd_slope_summary.json").write_text(json.dumps(psd_summ, indent=2))
        logger.info(f"[eval] Wrote PSD slope summary to {tables_dir / 'psd_slope_summary.json'}")
        try:
            # only if you have this plotting helper
            plot_psd_curves(pmm_bt, hr_bt, mask=mask_bt,
                            dx_km=self.eval_cfg.grid_km_per_px,
                            seasons=self.eval_cfg.seasons,
                            out_dir=str(figs_dir))
        except Exception as e:
            logger.warning(f"[eval] plot_psd_curves failed: {e}")

        # P95/P99 & wet-day frequency
        tails = compute_p95_p99_and_wet_day(
            pmm_bt, hr_bt=hr_bt, mask=mask_bt,
            wet_threshold_mm=self.eval_cfg.wet_threshold_mm
        )
        def _py(v):
            import numpy as _np
            if isinstance(v, torch.Tensor):
                return float(v.detach().cpu().item()) if v.numel() == 1 else [float(x) for x in v.detach().cpu().flatten()]
            if isinstance(v, _np.generic):
                return float(v.item())
            return v
        tails_py = {k: _py(v) for k, v in (tails or {}).items()}
        (tables_dir / "tails_summary.json").write_text(json.dumps(tails_py, indent=2))
        logger.info("[eval] Wrote tails summary to %s : %s", tables_dir / "tails_summary.json", tails_py)

        # Additionally write a flat CSV so downstream summary scripts don’t produce NaNs
        # Columns: choose PMM (gen_) as the model’s point estimate and include HR for reference
        def _first_scalar(val):
            # If val is a list, return its first element; else return as is
            if isinstance(val, list):
                return float(val[0]) if val else float('nan')
            return float(val)
        tails_row = {
            "P95":        _first_scalar(tails_py.get("gen_p95", np.nan)),
            "P99":        _first_scalar(tails_py.get("gen_p99", np.nan)),
            "WetDayFreq": _first_scalar(tails_py.get("gen_wet_freq", np.nan)),
            "HR_P95":     _first_scalar(tails_py.get("hr_p95", np.nan)),
            "HR_P99":     _first_scalar(tails_py.get("hr_p99", np.nan)),
            "HR_WetDayFreq": _first_scalar(tails_py.get("hr_wet_freq", np.nan)),
        }
        tails_csv = tables_dir / "tails_summary_flat.csv"
        with open(tails_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tails_row.keys()))
            w.writeheader(); w.writerow(tails_row)
        logger.info("[eval] Wrote flat tails CSV to %s : %s", tails_csv, tails_row)


    # ---------- Extremes (basin-mean series) ----------
    def eval_extremes(self):
        """
        Compute extreme value metrics using basin-mean daily series and save tables/figures
        Evaluates: 
            - GEV first for Rx1day/Rx5day with bootstrap CIs
            - POT/GPD over a threshold with bootstrap CIs
        Outputs tables (JSON)
        """
        tables_dir = self.out_root / "tables"
        figs_dir = self.out_root / "figures"

        # Build daily basin-mean series from HR & PMM
        dates = list(self._list_dates())
        HR = []
        PMM = []
        for date in dates:
            obs = self._load_obs(date)     # [H,W]
            pmm = self._load_pmm(date)     # [H,W]
            if obs is None or pmm is None: continue
            HR.append(obs.unsqueeze(0))
            PMM.append(pmm.unsqueeze(0))
        if len(HR) == 0:
            logger.warning("[eval] No data for extremes.")
            return

        hr_stack  = torch.stack(HR, 0).squeeze(1)   # [T,H,W]
        pmm_stack = torch.stack(PMM, 0).squeeze(1)  # [T,H,W]

        # Prefer a canonical mask; if only per-date masks exist, intersect them
        if self.mask_global is not None:
            basin = self.mask_global
        else:
            # Intersect all per-date masks (logical AND) so extremes compare the same area
            masks = []
            for d in dates:
                md = self._load_mask(d)
                if md is not None:
                    masks.append(md)
            basin = None
            if len(masks) > 0:
                mb = masks[0].clone()
                for md in masks[1:]:
                    mb &= md
                basin = mb

        series_hr  = to_numpy_1d_series(hr_stack,  mask=basin, agg="mean")  # mm/day
        series_pmm = to_numpy_1d_series(pmm_stack, mask=basin, agg="mean")

        # Dates to np.datetime64 (assumes YYYY-MM-DD filenames)
        dates_np = np.array([np.datetime64(d) for d in dates])
        blk = seasonal_block_index(dates_np)

        # GEV on Rx1day & Rx5day (seasonal blocks, 4 per year)
        rx1_hr, _  = rxk_series(series_hr,  1, block_index=blk)
        rx1_pmm,_  = rxk_series(series_pmm, 1, block_index=blk)
        rx5_hr,_   = rxk_series(series_hr,  5, block_index=blk)
        rx5_pmm,_  = rxk_series(series_pmm, 5, block_index=blk)

        min_gev_n = 8  # pragmatic minimum; adjust if desired
        if (len(rx1_hr) < min_gev_n) or (len(rx1_pmm) < min_gev_n) or \
        (len(rx5_hr) < min_gev_n) or (len(rx5_pmm) < min_gev_n):
            logger.warning(
                "[eval] Too few block maxima for GEV fitting "
                "(n_rx1_hr=%d, n_rx1_pmm=%d, n_rx5_hr=%d, n_rx5_pmm=%d). Skipping GEV/POT.",
                len(rx1_hr), len(rx1_pmm), len(rx5_hr), len(rx5_pmm)
            )
            return

        def _to_py(obj):
            import numpy as _np
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
            if isinstance(obj, _np.generic):
                return obj.item()
            if isinstance(obj, dict):
                return {k: _to_py(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_py(v) for v in obj]
            return obj

        def _dump(name, res):
            (tables_dir / name).write_text(json.dumps(_to_py(res), indent=2))

        _dump("gev_rx1_hr.json",  fit_gev_block_maxima_with_ci(rx1_hr,  block_per_year=4.0))
        _dump("gev_rx1_pmm.json", fit_gev_block_maxima_with_ci(rx1_pmm, block_per_year=4.0))
        _dump("gev_rx5_hr.json",  fit_gev_block_maxima_with_ci(rx5_hr,  block_per_year=4.0))
        _dump("gev_rx5_pmm.json", fit_gev_block_maxima_with_ci(rx5_pmm, block_per_year=4.0))
        logger.info("[eval] Wrote GEV JSONs (rx1/rx5 for HR and PMM) to %s", tables_dir)

        # POT/GPD threshold (wet-day P95 on HR series)
        wet_hr = series_hr[np.isfinite(series_hr) & (series_hr >= self.eval_cfg.wet_threshold_mm)]
        if wet_hr.size >= 10:
            u = float(np.percentile(wet_hr, 95.0))
            wrote_any = False

            # HR POT
            try:
                res_hr = fit_pot_gpd_with_ci(series_hr, threshold=u)
                _dump("pot_hr.json", res_hr)
                wrote_any = True
            except ValueError as e:
                logger.warning("[eval] POT/HR skipped: %s", e)

            # PMM POT (may have fewer exceedances than HR at same u)
            try:
                res_pmm = fit_pot_gpd_with_ci(series_pmm, threshold=u)
                _dump("pot_pmm.json", res_pmm)
                wrote_any = True
            except ValueError as e:
                logger.warning("[eval] POT/PMM skipped: %s", e)

            if wrote_any:
                logger.info("[eval] Wrote POT JSONs (threshold=HR P95=%.2f mm/day) to %s", u, tables_dir)
            else:
                logger.warning("[eval] POT skipped for both HR and PMM (insufficient exceedances at u=%.2f).", u)
        else:
            logger.warning("[eval] Too few wet HR days (N=%d) to set a robust POT threshold; skipping POT.", wet_hr.size)


    # ---------- Orchestrator ----------
    def run_all(self, do_prob=True, do_cap=True, do_ext=True):

        if do_prob: self.eval_probabilistic()
        if do_cap:  self.eval_capability()
        if do_ext:  self.eval_extremes()
        
        logger.info("[eval] run_all: do_prob=%s, do_cap=%s, do_ext=%s", do_prob, do_cap, do_ext)
        
        # manifest
        manifest = {
            "gen_dir": str(self.gen_root),
            "out_dir": str(self.out_root),
            "thresholds_mm": list(map(float, self.eval_cfg.thresholds_mm)),
            "fss_scales_km": list(map(float, self.eval_cfg.fss_scales_km)),
            "grid_km_per_px": float(self.eval_cfg.grid_km_per_px),
        }
        (self.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        logger.info("[eval] Wrote manifest to %s", self.out_root / "manifest.json")
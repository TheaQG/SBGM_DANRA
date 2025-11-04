from __future__ import annotations
from pathlib import Path
from typing import Sequence, Dict, List
import numpy as np

from sbgm.evaluate.evaluate_prcp.eval_temporal.metrics_temporal import (
    build_domain_mean_series,
    compute_temporal_metrics,
)

from sbgm.evaluate.evaluate_prcp.eval_temporal.plot_temporal import (
    plot_temporal
)

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def run_temporal(
    resolver,
    eval_cfg,
    out_root: str | Path,
    *,
    group_by: str = "year",         # "year" | "season" | "all"
    seasons: Sequence[str] = ("ALL","DJF","MAM","JJA","SON"),
    make_plots: bool = True,
) -> None:
    out_root = Path(out_root)
    tables_dir = _ensure_dir(out_root / "tables")
    figs_dir = _ensure_dir(out_root / "figures")

    # config
    max_lag   = int(getattr(eval_cfg, "temporal_max_lag", 30))
    max_spell = int(getattr(eval_cfg, "temporal_max_spell", 25))
    wet_thr   = float(getattr(eval_cfg, "temporal_wet_thr_mm", 1.0))
    include_lr = bool(getattr(eval_cfg, "temporal_include_lr", True))

    # dates universe from PMM folder (like spatial)
    all_dates: List[str] = list(resolver.list_dates())
    if not all_dates:
        return

    def _year_of(d: str) -> int:
        s = d.strip()
        if len(s) == 8 and s.isdigit(): return int(s[:4])
        return int(s[:4])

    def _season_of(d: str) -> str:
        s = d.strip()
        if len(s) == 8 and s.isdigit(): m = int(s[4:6])
        else:                             m = int(s[5:7])
        if m in (12,1,2): return "DJF"
        if m in (3,4,5):  return "MAM"
        if m in (6,7,8):  return "JJA"
        return "SON"

    def split_dates(dates: List[str]) -> Dict[str, List[str]]:
        if group_by == "all":
            return {"ALL": dates}
        if group_by == "season":
            out: Dict[str, List[str]] = {s: [] for s in seasons}
            for d in dates:
                out[_season_of(d)].append(d)
            return {s: out.get(s, []) for s in seasons if s in out}
        buckets: Dict[str, List[str]] = {}
        for d in dates:
            y = _year_of(d)
            buckets.setdefault(str(y), []).append(d)
        return dict(sorted(buckets.items()))

    groups = split_dates(all_dates)

    for gname, gdates in groups.items():
        if not gdates: continue

        series = build_domain_mean_series(
            gdates, resolver, use_mask=bool(getattr(eval_cfg, "eval_land_only", True))
        )
        if not include_lr and "LR" in series:
            series.pop("LR", None)

        # Save timeseries table
        np.savez_compressed(
            tables_dir / f"temporal_series_{gname}.npz",
            dates=np.asarray(gdates),
            **{f"{k}_series": v for k,v in series.items()}
        )

        metrics = compute_temporal_metrics(
            series, wet_thr_mm=wet_thr, max_lag=max_lag, max_spell=max_spell
        )
        # Pairwise distribution distances (GEN↔HR, LR↔HR) for wet/dry PMFs
        def _js_distance(p, q):
            p = np.asarray(p, dtype=float)
            q = np.asarray(q, dtype=float)
            m = 0.5*(p+q)
            mask = np.isfinite(p) & np.isfinite(q) & np.isfinite(m) & (m > 0)
            p = p[mask]; q = q[mask]; m = m[mask]
            # add tiny epsilon to avoid log(0)
            eps = 1e-12
            p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0); m = np.clip(m, eps, 1.0)
            kl_pm = np.sum(p * (np.log(p) - np.log(m)))
            kl_qm = np.sum(q * (np.log(q) - np.log(m)))
            return float(np.sqrt(0.5*(kl_pm + kl_qm)))
        def _ks_distance(p, q):
            p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
            mask = np.isfinite(p) & np.isfinite(q)
            p = p[mask]; q = q[mask]
            Pc = np.cumsum(p); Qc = np.cumsum(q)
            return float(np.nanmax(np.abs(Pc - Qc))) if p.size and q.size else np.nan
        pair_metrics = {"wet": {}, "dry": {}}
        if "HR" in metrics and "GEN" in metrics:
            pair_metrics["wet"]["JSD_GEN_HR"] = _js_distance(metrics["GEN"]["wet_pmf"], metrics["HR"]["wet_pmf"])
            pair_metrics["wet"]["KS_GEN_HR"]  = _ks_distance(metrics["GEN"]["wet_pmf"], metrics["HR"]["wet_pmf"])
            pair_metrics["dry"]["JSD_GEN_HR"] = _js_distance(metrics["GEN"]["dry_pmf"], metrics["HR"]["dry_pmf"])
            pair_metrics["dry"]["KS_GEN_HR"]  = _ks_distance(metrics["GEN"]["dry_pmf"], metrics["HR"]["dry_pmf"])
        if "HR" in metrics and "LR" in metrics:
            pair_metrics["wet"]["JSD_LR_HR"] = _js_distance(metrics["LR"]["wet_pmf"], metrics["HR"]["wet_pmf"])
            pair_metrics["wet"]["KS_LR_HR"]  = _ks_distance(metrics["LR"]["wet_pmf"], metrics["HR"]["wet_pmf"])
            pair_metrics["dry"]["JSD_LR_HR"] = _js_distance(metrics["LR"]["dry_pmf"], metrics["HR"]["dry_pmf"])
            pair_metrics["dry"]["KS_LR_HR"]  = _ks_distance(metrics["LR"]["dry_pmf"], metrics["HR"]["dry_pmf"])

        # Save metrics table
        payload = {}
        for k, d in metrics.items():
            payload[f"{k}_autocorr"] = d["autocorr"]
            payload[f"{k}_P"]        = d["P"]
            payload[f"{k}_wet_bins"] = d["wet_bins"]
            payload[f"{k}_wet_pmf"]  = d["wet_pmf"]
            payload[f"{k}_wet_geom_p"] = d["wet_geom_p"]
            payload[f"{k}_dry_bins"] = d["dry_bins"]
            payload[f"{k}_dry_pmf"]  = d["dry_pmf"]
            payload[f"{k}_dry_geom_p"] = d["dry_geom_p"]
        # add pair metrics scalars to table for reproducibility
        for which in ("wet","dry"):
            for key, val in pair_metrics[which].items():
                payload[f"pair_{which}_{key}"] = val
        np.savez_compressed(tables_dir / f"temporal_metrics_{gname}.npz", **payload)

        if make_plots:
            plot_temporal(figs_dir, gname, np.asarray(gdates), series, metrics, pair_metrics=pair_metrics)
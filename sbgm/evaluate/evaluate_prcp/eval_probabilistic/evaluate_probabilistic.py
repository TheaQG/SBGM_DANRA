"""
    Probabilistic metrics for precipitation evaluation.
    Gathers computation and plotting functions.

"""

from __future__ import annotations
from pathlib import Path
from typing import Sequence, Optional, Dict, Any, List

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

from sbgm.evaluate.evaluate_prcp.eval_probabilistic.metrics_probabilistic import (
    crps_ensemble,
    pit_values_from_ensemble,
    rank_histogram,
    reliability_exceedance_binned,
    aggregate_reliability_bins,
    spread_skill_binned,
    energy_score,
    variogram_score,
)
from sbgm.evaluate.evaluate_prcp.eval_probabilistic.plot_probabilistic import (
    plot_probabilistic,
)

# Helper: select CRPS example dates (best/worst, deprioritize zeros)
def _select_crps_example_dates(crps_csv_path: Path, n_examples: int = 6) -> list[tuple[str, float]]:
    lines = crps_csv_path.read_text().strip().splitlines()
    if len(lines) <= 1:
        return []
    rows: list[tuple[str, float]] = []
    for ln in lines[1:]:
        s = ln.split(",")
        if len(s) < 2:
            continue
        date_s = s[0].strip()
        try:
            crps_v = float(s[1])
        except Exception:
            continue
        rows.append((date_s, crps_v))
    if not rows:
        return []
    rows_sorted = sorted(rows, key=lambda x: x[1])
    eps = 0.01
    nonzero_rows = [r for r in rows_sorted if r[1] > eps]
    zero_rows = [r for r in rows_sorted if r[1] <= eps]
    n_half = max(1, n_examples // 2)
    if len(nonzero_rows) >= n_half:
        best = nonzero_rows[:n_half]
    else:
        best = nonzero_rows + zero_rows[:(n_half - len(nonzero_rows))]
    worst = rows_sorted[-n_half:]
    return best + worst

# Helper: build and save member-based CRPS example payload for plotting
def _build_and_save_crps_examples_members(
    resolver,
    tables_dir: Path,
    *,
    n_members_to_show: int = 4,
    member_seed: int = 1234,
) -> None:
    crps_csv = tables_dir / "prob_crps_daily.csv"
    if not crps_csv.exists():
        return
    selected = _select_crps_example_dates(crps_csv, n_examples=6)
    if not selected:
        return

    rng = np.random.RandomState(member_seed)
    payload: Dict[str, Any] = {"dates": np.array([d for d, _ in selected])}

    for d, _ in selected:
        obs = resolver.load_obs(d)            # [H,W]
        ens = resolver.load_ens(d)            # [M,H,W]
        try:
            pmm = resolver.load_pmm(d)        # [H,W] (optional but desired)
        except Exception:
            pmm = None
        try:
            mask = resolver.load_mask(d)
        except Exception:
            mask = None

        if obs is None or ens is None:
            continue

        obs_t = torch.from_numpy(np.asarray(obs)) if not torch.is_tensor(obs) else obs
        ens_t = torch.from_numpy(np.asarray(ens)) if not torch.is_tensor(ens) else ens

        # Ensemble CRPS (scalar) for the column title
        crps_val = crps_ensemble(obs_t, ens_t, mask=mask, reduction="mean")
        payload[f"CRPS_ENS_{d}"] = float(crps_val)

        # Save HR (always)
        payload[f"HR_{d}"] = np.asarray(obs_t.cpu()).astype(np.float32)

        # Save PMM if available (for last row)
        if pmm is not None:
            payload[f"PMM_{d}"] = np.asarray(pmm).astype(np.float32)

        # Choose members to display
        M = ens_t.shape[0]
        take = min(n_members_to_show, M)
        idx = rng.choice(M, size=take, replace=False)

        # Optional mask for MAE
        m = None
        if mask is not None:
            m = np.asarray(mask).astype(bool)

        # Save members and their MAE (CRPS for a single member reduces to MAE)
        hr_np = payload[f"HR_{d}"]
        for j, k in enumerate(idx):
            mem = ens_t[k].cpu().numpy().astype(np.float32)
            payload[f"MEM_{j}_{d}"] = mem
            if m is not None:
                mae = np.nanmean(np.abs(np.where(m, mem, np.nan) - np.where(m, hr_np, np.nan)))
            else:
                mae = float(np.mean(np.abs(mem - hr_np)))
            payload[f"MAE_MEM_{j}_{d}"] = float(mae)

    # Ensure all payload values are array-like before saving to satisfy type checkers
    np.savez_compressed(
        tables_dir / "prob_crps_examples_members.npz",
        **{k: np.asarray(v) for k, v in payload.items()}
    )

def run_probabilistic(
        resolver,
        eval_cfg,
        out_root: str | Path,
        *,
        plot_only: bool = False,
) -> None:
    """
        Main entry for precipitation probabilistic evaluation.

        Parameters:
            resolver
                Object that knows how to access evaluation data. Must provide:
                    - list_dates()
                    - load_obs(date)
                    - load_ens(date)
                    - load_pmm(date) (optional, may return None)
                    - load_mask(date) (optional, may return None)
            eval_cfg
                Config-like object with evaluation settings.
                    - thresholds_mm
                    - reliability_bins
                    - spread_skill_bins
                    - pit_bins
            out_root
                Directory to save outputs to.
            plot_only
                If True, only generate plots from (existing) data.
    """
    out_root = Path(out_root)
    tables_dir = out_root / "tables"
    figs_dir = out_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # if user only wants plots, just read + plot
    if plot_only:
        thresholds = getattr(eval_cfg, "thresholds_mm", (1.0, 5.0, 10.0))
        pit_bins = int(getattr(eval_cfg, "pit_bins", 20))
        plot_probabilistic(out_root, thresholds=thresholds, pit_bins=pit_bins)
        return
    
    # ================================================================================
    # 1) setup / config
    # ================================================================================
    thresholds: Sequence[float] = getattr(eval_cfg, "thresholds_mm", (1.0, 5.0, 10.0))
    n_rel_bins: int = int(getattr(eval_cfg, "reliability_bins", 10))
    n_ss_bins: int = int(getattr(eval_cfg, "spread_skill_bins", 10))
    pit_bins: int = int(getattr(eval_cfg, "pit_bins", 20))

    vs_p: float = float(getattr(eval_cfg, "variogram_p", 0.5))
    vs_max_pairs: int = int(getattr(eval_cfg, "variogram_max_pairs", 4000))

    dates: List[str] = list(resolver.list_dates())

    # data accumulators
    crps_lines: List[str] = ["date,crps"]
    es_lines: List[str] = ["date,energy_score"]
    vs_lines: List[str] = ["date,variogram_score"]
    all_pit: List[np.ndarray] = []
    rank_acc: Optional[torch.Tensor] = None
    rel_acc: Dict[float, List[Dict[str, torch.Tensor]]] = {float(t): [] for t in thresholds}
    ss_lines: List[str] = ["date,spread_mean,skill_mean"]    

    # ================================================================================
    # 2) per-date loop
    # ================================================================================
    _crps_sum = None
    _crps_cnt = None
    for d in dates:
        logger.info(f"[eval_probabilistic] Processing date {d} ...")
        # load

        obs = resolver.load_obs(d)     # [H,W]
        ens = resolver.load_ens(d)     # [M,H,W]
        if obs is None or ens is None:
            # skip incomplete samples
            continue

        mask = resolver.load_mask(d)   # [H,W] or None
        pmm  = None
        # optional PMM (prefer phys)
        try:
            pmm = resolver.load_pmm(d)
        except Exception:
            pmm = None

        # make sure tensors are on a device (CPU is fine here)
        if not torch.is_tensor(obs):
            obs = torch.from_numpy(obs)
        if not torch.is_tensor(ens):
            ens = torch.from_numpy(ens)

        # 2.1 CRPS (domain average, masked)
        crps_val = crps_ensemble(obs, ens, mask=mask, reduction="mean")
        crps_lines.append(f"{d},{float(crps_val):.6f}")

        # Energy score (field-wise, multivariate generalization of CRPS)
        es_val = energy_score(obs, ens, mask=mask)
        es_lines.append(f"{d},{float(es_val):.6f}")
        
        # Variogram score (spatial dependence)
        vs_val = variogram_score(
            obs,
            ens,
            mask=mask,
            p=vs_p,
            max_pairs=vs_max_pairs,
            seed=0,
        )
        vs_lines.append(f"{d},{float(vs_val):.6f}")        

        # 2.1b CRPS map (for spatial mean later)
        crps_map = crps_ensemble(obs, ens, mask=mask, reduction="none")  # [H,W]
        
        if _crps_sum is None:
            _crps_sum = crps_map.clone().float()
            if mask is not None:
                _crps_cnt = mask.to(crps_map.dtype)
            else:
                _crps_cnt = torch.ones_like(crps_map, dtype=torch.float32)
        else:
            _crps_sum = _crps_sum + crps_map
            if mask is not None:
                _crps_cnt = _crps_cnt + mask.to(crps_map.dtype)
            else:
                _crps_cnt = _crps_cnt + torch.ones_like(crps_map, dtype=torch.float32)

        # 2.2 PIT
        # metrics expect B>1, so wrap a batch dimension
        pits = pit_values_from_ensemble(
            obs.unsqueeze(0),            # [1,H,W]
            ens.unsqueeze(0),            # [1,M,H,W]
            mask=mask,
            randomized=True,
        )
        all_pit.append(pits.numpy())

        # 2.3 Rank histogram
        rh = rank_histogram(
            obs.unsqueeze(0),
            ens.unsqueeze(0),
            mask=mask,
            randomize_ties=True,
        )
        if rank_acc is None:
            rank_acc = rh
        else:
            rank_acc = rank_acc + rh

        # 2.4 Reliability per threshold
        for thr in thresholds:
            rel = reliability_exceedance_binned(
                obs=obs,
                ens=ens,
                threshold=float(thr),
                lr_covariate=None,               # hook for LR-stratified plots later
                n_bins=n_rel_bins,
                mask=mask,
                return_brier=True,
            )
            rel_acc[float(thr)].append(rel)

        # 2.5 Spread–skill
        ss = spread_skill_binned(
            obs=obs,
            ens=ens,
            point_field=pmm,              # use PRECOMPUTED PMM if available
            point="mean",                 # fallback if pmm is None
            mask=mask,
            n_bins=n_ss_bins,
        )
        # We want a per-date SINGLE number (for the time series plot),
        # so take a simple count-weighted mean over bins:
        cnt = ss["count"].numpy().astype(np.int64)
        spr = ss["spread"].numpy()
        skl = ss["skill"].numpy()
        w = cnt.clip(min=0)
        if w.sum() > 0:
            spread_mean = float((spr * w).sum() / w.sum())
            skill_mean = float((skl * w).sum() / w.sum())
        else:
            spread_mean = 0.0
            skill_mean = 0.0
        ss_lines.append(f"{d},{spread_mean:.6f},{skill_mean:.6f}")
    
    
    # ================================================================================
    # 3) write output tables
    # ================================================================================

    # 3.1 CRPS
    # 3.1b temporally averaged CRPS map
    if _crps_sum is not None and _crps_cnt is not None:
        mean_map = (_crps_sum / _crps_cnt.clamp(min=1.0)).cpu().numpy()
        np.savez_compressed(
            tables_dir / "prob_crps_mean_map.npz",
            crps_mean_map=mean_map,     # <-- correct key name
        )
    # Energy score (per day)
    (tables_dir / "prob_energy_daily.csv").write_text("\n".join(es_lines))
    # Variogram score (per day)
    (tables_dir / "prob_variogram_daily.csv").write_text("\n".join(vs_lines))

    # 3.2 PIT
    if all_pit:
        pit_all = np.concatenate(all_pit, axis=0)
    else:
        pit_all = np.array([], dtype=np.float32)
    np.savez_compressed(tables_dir / "prob_pit_values.npz", pit=pit_all)

    # 3.3 Rank
    if rank_acc is not None:
        np.savez_compressed(tables_dir / "prob_rank_histogram.npz", rank_hist=rank_acc.numpy())

    # 3.4 Reliability (aggregate across dates → write one file per threshold)
    for thr, lst in rel_acc.items():
        if not lst:
            continue
        agg = aggregate_reliability_bins(lst)
        bc = agg["bin_center"].numpy()
        pp = agg["prob_pred"].numpy()
        fo = agg["freq_obs"].numpy()
        cnt = agg["count"].numpy()

        lines = ["bin_center,prob_pred,freq_obs,count"]
        for b, p, f, c in zip(bc, pp, fo, cnt):
            lines.append(f"{b:.6f},{p:.6f},{f:.6f},{int(c)}")
        (tables_dir / f"prob_reliability_{thr:.1f}mm.csv").write_text("\n".join(lines))

    # 3.5 Spread–skill (per-date summary)
    (tables_dir / "prob_spread_skill.csv").write_text("\n".join(ss_lines))

    # ================================================================================
    # 4) plots
    # ================================================================================
    # Build member-based CRPS example payload for plotting (HR + members only)
    try:
        n_show = int(getattr(eval_cfg, "crps_examples_n_members", 4))
        seed = int(getattr(eval_cfg, "ensemble_member_seed", 1234))
        _build_and_save_crps_examples_members(
            resolver,
            tables_dir,
            n_members_to_show=n_show,
            member_seed=seed,
        )
    except Exception as e:
        logger.warning(f"[eval_probabilistic] Could not build CRPS member examples: {e}")

    plot_probabilistic(
        out_root,
        gen_root=Path(eval_cfg.gen_dir),
        thresholds=thresholds,
        pit_bins=pit_bins,
    )

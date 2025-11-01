from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import json
import logging

from sbgm.evaluate.data_resolver import EvalDataResolver
from sbgm.evaluate.evaluate_prcp.eval_probabilistic.evaluate_probabilistic import (run_probabilistic)

logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    gen_dir: str
    out_dir: str
    eval_land_only: bool = True
    prefer_phys: bool = True
    region_mask_path: Optional[str] = None
    grid_km_per_px: float = 2.5
    lr_grid_km_per_px: float = 31.0
    thresholds_mm: tuple = (1.0, 5.0, 10.0)
    fss_scales_km: tuple = (5, 10, 20)
    seasons: tuple = ("ALL", "DJF", "MAM", "JJA", "SON")
    reliability_bins: int = 10
    spread_skill_bins: int = 10
    pit_bins: int = 20

class EvaluationRunner:
    """
        Clean runner that uses EvalDataResolver to perform evaluations.
    """

    def __init__(
            self, 
            cfg_yaml: dict,
            eval_cfg: EvaluationConfig,
            device: torch.device,
            baseline_eval_dirs: Optional[Dict[str, str]] = None,
            plot_only: bool = False
    ):
        self.cfg_yaml = cfg_yaml
        self.eval_cfg = eval_cfg
        self.device = device
        self.baseline_eval_dirs = baseline_eval_dirs
        self.plot_only = plot_only

        # Data access
        self.data = EvalDataResolver(
            gen_root=eval_cfg.gen_dir,
            eval_land_only=eval_cfg.eval_land_only,
            roi_mask_path=eval_cfg.region_mask_path,
            prefer_phys=eval_cfg.prefer_phys
        )

        # Output setup
        self.out_root = Path(eval_cfg.out_dir)
        (self.out_root / "tables").mkdir(parents=True, exist_ok=True)
        (self.out_root / "figures").mkdir(parents=True, exist_ok=True)

    def should_compute(self, rel_table_name: str) -> bool:
        """
            Return False when running in plot_only mode and the table already exists.
        """
        table_path = self.out_root / "tables" / rel_table_name
        if self.plot_only and table_path.exists():
            logger.info(f"[EvaluationRunner] plot_only=True and table {table_path} exists; skipping computation.")
            return False
        return True

    # =====
    # Simple run example: list dates and write manifest
    # =====
    

    def run(self, tasks: Optional[List[str]] = None):
        """
            Run the evaluation tasks specified.

            Args:
                tasks: List of task names to run. If None, run all available tasks.
        """
        dates = self.data.list_dates()
        logger.info(f"[EvaluationRunner] Found {len(dates)} dates for evaluation.")

        manifest = {
            "gen_dir": self.eval_cfg.gen_dir,
            "out_dir": self.eval_cfg.out_dir,
            "n_dates": len(dates),
            "thresholds_mm": list(self.eval_cfg.thresholds_mm),
            "fss_scales_km": list(self.eval_cfg.fss_scales_km),
            "seasons": list(self.eval_cfg.seasons),
        }
        # Before writing manifest.json, check should_compute
        if self.should_compute("manifest.json"):
            (self.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        else:
            logger.info(f"[EvaluationRunner] Skipping manifest.json write (plot_only and file exists).")

        # ===== 
        # Dispatch to evaluation tasks
        # =====
        if tasks is None:
            # Sensible default: run prcipitation probabilistic evaluation
            tasks = ["prcp_probabilistic"]
        for task in tasks:
            # 1) Precipitation probabilistic evaluation
            if task in ("prcp_probabilistic", "prcp_prob", "prob", "probabilistic"):
                out_dir = self.out_root / "prcp" / "probabilistic"
                out_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"[EvaluationRunner] Running task '{task}' -> {out_dir}")
                
                run_probabilistic(
                    resolver=self.data,
                    eval_cfg=self.eval_cfg,
                    out_root=out_dir,
                    plot_only=self.plot_only
                )
                continue

        # To be implemented: calls to...
        # if "scale" in tasks: evaluate_scale.run(...)
        # ... etc.

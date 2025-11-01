from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Iterable

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

class EvalDataResolver:
    """
        Centralized access to the already generated data for evaluation.

        A cleaned up version of the previous ad-hoc data logic living in
        sbgm/evaluate_sbgm/evaluation_main.py but now in its own module
        for better reusability and testability.
    """
    def __init__(
            self,
            gen_root: str | Path,
            eval_land_only: bool = True,
            roi_mask_path: Optional[str | Path] = None,
            prefer_phys: bool = True
    ):
        self.gen_root = Path(gen_root)
        self.eval_land_only = eval_land_only
        self.roi_mask_path = Path(roi_mask_path) if roi_mask_path is not None else None
        self.prefer_phys = prefer_phys

        # Preferred physical-space data paths
        self.dir_ens_phys = self.gen_root / "ensembles_phys"
        self.dir_pmm_phys = self.gen_root / "pmm_phys"
        self.dir_lrhr_phys = self.gen_root / "lr_hr_phys"

        # Fallback model-space data paths
        self.dir_ens_model = self.gen_root / "ensembles"
        self.dir_pmm_model = self.gen_root / "pmm"
        self.dir_lrhr_model = self.gen_root / "lr_hr"

        # Masks
        self.dir_lsm = self.gen_root / "lsm"

        # === Load global land-sea mask if present ===
        self.mask_global: Optional[torch.Tensor] = None
        try:
            p = self.gen_root / "meta" / "land_mask.npz"
            if p.exists():
                arr = np.load(p, allow_pickle=True).get("lsm_hr", None)
                if arr is not None:
                    self.mask_global = torch.from_numpy(np.asarray(arr)).to(torch.bool)
                    logger.info(f"[EvalDataResolver] Loaded global land-sea mask from {p} with shape {self.mask_global.shape}")
        except Exception as e:
            p = self.gen_root / "meta" / "land_mask.npz"
            logger.warning(f"[EvalDataResolver] Failed to load global land-sea mask from {p}: {e}")

        # === Load Region-of-Interest mask if provided ===
        self.roi_mask: Optional[torch.Tensor] = None
        if roi_mask_path is not None:
            roi_path = Path(roi_mask_path)
            if roi_path.exists():
                try:
                    arr = np.load(roi_path, allow_pickle=True)
                    if isinstance(arr, np.lib.npyio.NpzFile): # type: ignore
                        a = arr.get("mask", None) or arr.get("lsm_hr", None) or arr.get("roi", None)
                    else:
                        a = arr
                    if a is not None:
                        m = torch.from_numpy(np.asarray(a)).to(torch.bool)
                        # normalize to [H,W]
                        if m.dim() == 4 and m.shape[:2] == (1,1):
                            m = m.squeeze(0).squeeze(0)
                        elif m.dim() == 3 and m.shape[0] == 1:
                            m = m.squeeze(0)
                        self.roi_mask = m
                        logger.info(f"[EvalDataResolver] Loaded ROI mask from {roi_path} with shape {self.roi_mask.shape}")
                except Exception as e:
                    logger.warning(f"[EvalDataResolver] Failed to load ROI mask from {roi_path}: {e}")


    # =====
    # Basic helpers
    # =====
    def list_dates(self) -> list[str]:
        """
            Return sorted date stems from PMM folder (physical preferred)
        """
        base = self.dir_pmm_phys if self.dir_pmm_phys.exists() and self.prefer_phys else self.dir_pmm_model
        dates = sorted([f.stem for f in base.glob("*.npz")])
        logger.info(f"[EvalDataResolver] Found {len(dates)} dates in PMM folder: {base}")
        return dates
    
    def _load_npz(self, folder: Path, date: str, key: str):
        p = folder / f"{date}.npz"
        # logger.info(f"[DEBUG EvalDataResolver] Attempting to load NPZ from {p} for key: {key}")
        if not p.exists():
            return None
        d = np.load(p, allow_pickle=True)
        # check existing keys and match with input
        # logger.info(f"[DEBUG EvalDataResolver] Loading {p}, available keys: {list(d.keys())}")
        # logger.info(f"[DEBUG EvalDataResolver] Extracting key: {key}")
        return d.get(key, None)
    
    # =====
    # Data loaders (HR, PMM, ensembles, LR, mask)
    # ===== 
    def load_obs(self, date: str) -> Optional[torch.Tensor]:
        """
            Load HR field for a given date.
            Prefer physical pairs under lr_hr_phys, fall back to lr_hr.
            Output shape: [H,W] torch.float
        """
        x = self._load_npz(self.dir_lrhr_phys, date, "hr")
        
        if x is None:
            x = self._load_npz(self.gen_root / "lr_hr", date, "hr")
        if x is None:
            return None
        t = torch.from_numpy(np.asarray(x)).squeeze(0)
        return t.squeeze(0)
    
    def load_pmm(self, date: str) -> Optional[torch.Tensor]:
        """
            Load PMM field for a given date.
            Output shape: [H,W] torch.float
        """
        x = self._load_npz(self.dir_pmm_phys, date, "pmm")
        if x is None:
            x = self._load_npz(self.dir_pmm_model, date, "pmm")
        if x is None:
            return None
        t = torch.from_numpy(np.asarray(x)).squeeze(0)
        return t.squeeze(0)


    def load_ens(self, date: str) -> Optional[torch.Tensor]:
        """
        Load ensemble for a given date.
        Output: [M,H,W] torch.float
        """
        x = self._load_npz(self.dir_ens_phys, date, "ens")
        if x is None:
            x = self._load_npz(self.dir_ens_model, date, "ens")
        if x is None:
            return None
        t = torch.from_numpy(np.asarray(x)).squeeze(1)  # [M,1,H,W] → [M,H,W]
        return t

    def load_lr(self, date: str) -> Optional[torch.Tensor]:
        """
        Load LR (native LR grid) for a given date.
        Output: [1,h,w] torch.float or None
        """
        x = self._load_npz(self.dir_lrhr_phys, date, "lr")
        if x is None:
            return None
        lr_t = torch.from_numpy(np.asarray(x))
        if lr_t.ndim == 3:
            lr_t = lr_t[0:1, ...]
        elif lr_t.ndim == 2:
            lr_t = lr_t.unsqueeze(0)
        return lr_t
    
    def load_mask(self, date: str) -> Optional[torch.Tensor]:
        """
        Prefer global land-sea mask over per-date masks under /lsm over None.
        Intersect with ROI mask if provided.
        Always normalize to [H,W].
        """
        if not self.eval_land_only:
            return None
        if self.mask_global is not None:
            m = self.mask_global.clone()
        else:
            p = self.dir_lsm / f"{date}.npz"
            if not p.exists():
                return None
            try:
                arr = np.load(p, allow_pickle=True).get("lsm_hr", None)
                if arr is None:
                    return None
                m = torch.from_numpy(np.asarray(arr)).to(torch.bool)
            except Exception as e:
                logger.warning(f"[EvalDataResolver] Failed to load land-sea mask from {p}: {e}")
                return None
        
        # normalize to [H,W]
        if m.dim() == 4 and m.shape[:2] == (1,1):
            m = m.squeeze(0).squeeze(0)
        elif m.dim() == 3 and m.shape[0] == 1:
            m = m.squeeze(0)

        # intersect with ROI mask if provided
        if self.roi_mask is not None:
            rm = self.roi_mask
            if rm.shape != m.shape:
                if rm.dim() == 4 and rm.shape[:2] == (1,1):
                    rm = rm.squeeze(0).squeeze(0)
                elif rm.dim() == 3 and rm.shape[0] == 1:
                    rm = rm.squeeze(0)
            if rm.shape == m.shape:
                m = m & rm
            else:
                logger.warning(f"[EvalDataResolver] ROI mask shape {rm.shape} does not match LSM shape {m.shape}, skipping intersection.")
        return m
                
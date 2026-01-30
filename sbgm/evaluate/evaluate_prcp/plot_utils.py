
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Optional

import os
import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _savefig(fig, out_path: Path, dpi: int = 300):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _nice():
    # lightweight, you can override with your global style later
    plt.rcParams.update({
        "figure.figsize": (5.5, 4.0),
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })

def _to_date_safe(s: str) -> Optional[datetime]:
    s = s.strip()
    # accept "YYYY-MM-DD" and "YYYYMMDD"
    try:
        if len(s) == 8 and s.isdigit():
            return datetime.strptime(s, "%Y%m%d")
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _season_from_month(m: int) -> str:
    if m in (12, 1, 2):
        return "DJF"
    if m in (3, 4, 5):
        return "MAM"
    if m in (6, 7, 8):
        return "JJA"
    return "SON"



# ------------------------------
# DK outline via LSM (cached)
# ------------------------------
_DK_LSM_CACHE: dict[tuple[int, int, int, int], np.ndarray] | np.ndarray | None = None

def _load_dk_lsm_outline(
    bounds: tuple[int, int, int, int] = (200, 328, 380, 508),
    base: str | None = None,
    rel_path: str = "data_lsm/truth_fullDomain/lsm_full.npz",
    key_candidates: tuple[str, ...] = ("lsm_hr", "lsm", "mask", "roi", "lsm_full", "data", "arr_0"),
) -> np.ndarray | None:
    """Load and crop a land-sea mask and return a boolean [H,W] mask for Denmark.
    bounds is interpreted as (y0, y1, x0, x1) with y1/x1 exclusive; e.g., (200,328,380,508) → 128x128.
    """
    try:
        # Resolve base path for data directory
        if base is None:
            base = os.environ.get("DATA_DIR", None)
        if base is None:
            logger.warning("[DK_LSM] DATA_DIR not set and no base path provided; cannot load LSM.")
            return None

        logger.info("[DEBUG] Loading DK LSM outline from %s/%s", base, rel_path)
        p = Path(base) / rel_path
        if not p.exists():
            logger.warning("[DEBUG] DK LSM outline file not found: %s", str(p))
            return None
        d = np.load(p, allow_pickle=True)
        # Print the keys available in the npz file for debugging
        arr = None
        if hasattr(d, "files"):
            for k in key_candidates:
                if k in d.files:
                    arr = d[k]
                    break
        if arr is None:
            logger.warning("[DEBUG] DK LSM outline: no suitable key found in %s", str(p))
            return None
        a = np.asarray(arr)
        # normalize to [H,W]
        if a.ndim == 4 and a.shape[:2] == (1, 1):
            a = a.squeeze(0).squeeze(0)
        elif a.ndim == 3 and a.shape[0] == 1:
            a = a.squeeze(0)
        y0, y1, x0, x1 = bounds
        a = np.flipud(a)  # flip vertically if needed
        a = a[y0:y1, x0:x1]
        m = (a >= 0.5)
        m = np.flipud(m)  # flip back to original orientation

        logger.info("[DEBUG] DK LSM outline loaded with shape %s", str(m.shape))
        return m.astype(bool, copy=False)
    except Exception as e:
        logger.exception("[DEBUG] Exception while loading DK LSM outline: %s", str(e))
        return None

def get_dk_lsm_outline(
    bounds: tuple[int, int, int, int] = (200, 328, 380, 508),
    base: str | None = None,
) -> np.ndarray | None:
    """
    Return cached DK outline mask (boolean [H,W]) for the requested `bounds`.
    Caches per-bounds so different crops return correctly sized masks.
    """
    global _DK_LSM_CACHE
    # For backward compat, treat _DK_LSM_CACHE as a dict if needed
    if isinstance(_DK_LSM_CACHE, dict):
        cache = _DK_LSM_CACHE
    else:
        # Single cache for default bounds (legacy)
        cache = {}
        if _DK_LSM_CACHE is not None:
            cache[(200, 328, 380, 508)] = _DK_LSM_CACHE
        _DK_LSM_CACHE = cache
    try:
        key = (int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3]))
    except Exception:
        key = (200, 328, 380, 508)
    if key in _DK_LSM_CACHE:
        return _DK_LSM_CACHE[key]
    m = _load_dk_lsm_outline(bounds=key, base=base)
    if m is not None:
        _DK_LSM_CACHE[key] = m
    return m

def overlay_outline(ax, mask: np.ndarray | None, *, color: str = "black", linewidth: float = 0.8):
    """Overlay a contour outline (level 0.5) on the given axes if mask is provided."""
    if mask is None:
        return
    try:
        ax.contour(mask.astype(float, copy=False), levels=[0.5], colors=color, linewidths=linewidth)
    except Exception:
        pass
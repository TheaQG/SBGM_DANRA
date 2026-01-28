"""model_tester_local.py

Dataset-backed smoke test for the Option-2 spatial conditioner (multi-scale + global pooled vector).

Runs:
  python -m sbgm.data.model_tester_local --cfg sbgm/config/paper2_data_test_local.yaml --split train --idx 0

What it does:
  1) builds the dataset/dataloader using training_utils.get_dataloader(cfg)
  2) pulls one batch and selects one sample (idx within batch)
  3) builds spatial_ctx_full by stacking *full-domain* LR variables (H_full,W_full)
  4) runs SpatialContextConditioner to obtain ctx_patch + ctx_global
  5) saves a plot: full-domain LR (channel 0) + HR rectangle, LR crop @ hr_points, ctx_patch, and optional HR patch
     with DK outline overlays (from LSM) to debug orientation issues.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

from matplotlib.patches import Rectangle

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from sbgm.training_utils import get_dataloader
from sbgm.score_unet import SpatialContextConditioner
from sbgm.plotting_utils import get_dk_lsm_outline, overlay_outline, imshow_variable, get_lsm_land_mask_full


def _as_int_points(p, *, sample_idx: int | None = None):
    """Return raw 4 ints from dataset points without assuming axis order."""
    import numpy as np
    import torch

    # CASE A: list-of-4 vectors (your current format)
    if isinstance(p, list) and len(p) == 4:
        if sample_idx is None:
            sample_idx = 0
        out = []
        for v in p:
            if torch.is_tensor(v):
                out.append(int(v[sample_idx].item()))
            else:
                out.append(int(np.asarray(v)[sample_idx]))
        return tuple(out)

    if torch.is_tensor(p):
        a = p.detach().cpu().numpy() # type: ignore
    else:
        a = np.asarray(p)

    # CASE B: [4]
    if a.ndim == 1 and a.size == 4:
        return tuple(int(x) for x in a.tolist())

    # CASE C: [B,4]
    if a.ndim == 2 and a.shape[1] == 4:
        if sample_idx is None:
            sample_idx = 0
        return tuple(int(x) for x in a[sample_idx].tolist())

    # CASE D: packed [4*N]
    f = np.asarray(a).reshape(-1).astype(int)
    if f.size >= 4 and (f.size % 4 == 0):
        return tuple(int(x) for x in f[:4].tolist())

    raise ValueError(f"Could not parse points. Got type={type(p)} shape={getattr(a,'shape',None)}")


def _choose_full_domain_lr_tensors(batch: Dict[str, Any], full_hw: Tuple[int, int]) -> Tuple[torch.Tensor, List[str]]:
    """Pick all keys ending with '_lr' whose spatial shape matches full_hw.

    Returns:
      x_full: [B, C_ctx, H_full, W_full]
      keys: list of keys used (for provenance)
    """
    Hf, Wf = int(full_hw[0]), int(full_hw[1])

    keys = []
    tensors = []
    for k, v in batch.items():
        if not isinstance(k, str) or not k.endswith("_lr"):
            continue
        if not torch.is_tensor(v):
            continue
        if v.ndim != 4:
            continue
        if tuple(v.shape[-2:]) == (Hf, Wf):
            tensors.append(v)
            keys.append(k)

    if len(tensors) == 0:
        lr_shapes = {}
        for k, v in batch.items():
            if isinstance(k, str) and k.endswith("_lr") and torch.is_tensor(v) and v.ndim == 4:
                lr_shapes[k] = tuple(v.shape)
        raise RuntimeError(
            "No full-domain LR tensors found in batch. "
            f"Expected some *_lr with spatial {(Hf, Wf)}. Found: {lr_shapes}"
        )

    x_full = torch.cat(tensors, dim=1)
    return x_full, keys

def _fix_order_xxyy(x1, x2, y1, y2):
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return (x1, x2, y1, y2)

def _resolve_points_to_xxyy(raw4, *, full_hw, hr_hw, cfg):
    """
    Resolve raw4 into canonical (x1,x2,y1,y2), handling ambiguity between:
      - raw4 as xxyy
      - raw4 as yyxx
    """
    Hf, Wf = int(full_hw[0]), int(full_hw[1])
    Hh, Wh = int(hr_hw[0]), int(hr_hw[1])
    a, b, c, d = map(int, raw4)

    cand1 = _fix_order_xxyy(a, b, c, d)   # assume raw4 is xxyy
    cand2 = _fix_order_xxyy(c, d, a, b)   # assume raw4 is yyxx

    def valid(c):
        x1, x2, y1, y2 = c
        if (x2-x1) != Wh or (y2-y1) != Hh:
            return False
        return (0 <= x1 < Wf) and (0 < x2 <= Wf) and (0 <= y1 < Hf) and (0 < y2 <= Hf)

    valids = [c for c in (cand1, cand2) if valid(c)]
    if len(valids) == 1:
        return valids[0]
    if len(valids) == 0:
        return cand1  # fallback

    # Prefer the one closest to configured DK bounds center (even if cfg itself is ambiguous)
    ref = None
    hr_cfg = (cfg or {}).get("highres", {})
    sc = (hr_cfg.get("stationary_cutout", {}) or {})
    if bool(sc.get("enabled", False)) and sc.get("bounds", None) is not None:
        ref = tuple(int(x) for x in sc["bounds"])
    elif hr_cfg.get("cutout_domains", None) is not None:
        ref = tuple(int(x) for x in hr_cfg["cutout_domains"])

    def center(c):
        x1,x2,y1,y2 = c
        return ((x1+x2)/2.0, (y1+y2)/2.0)

    def score(c):
        if ref is None or len(ref) != 4:
            return 0.0
        r0,r1,r2,r3 = ref
        ref1 = _fix_order_xxyy(r0,r1,r2,r3)  # ref as xxyy
        ref2 = _fix_order_xxyy(r2,r3,r0,r1)  # ref as yyxx
        cx,cy = center(c)
        rcx1,rcy1 = center(ref1)
        rcx2,rcy2 = center(ref2)
        d1 = (cx-rcx1)**2 + (cy-rcy1)**2
        d2 = (cx-rcx2)**2 + (cy-rcy2)**2
        return min(d1,d2)

    return min(valids, key=score)

# --- Helper to optionally flip y coordinates for plotting_utils (origin='lower') ---
def _maybe_flip_y_points(
    pts: Tuple[int, int, int, int],
    *,
    H_full: int,
    mode: str,
) -> Tuple[int, int, int, int]:
    """Optionally flip y coordinates from a top-origin convention to bottom-origin.

    We use the standard image-array indexing convention for slicing (y first, then x),
    but your plotting_utils / LSM outline are standardized around origin='lower'.

    If the dataset produces hr_points in a top-origin coordinate system, the correct
    mapping into origin='lower' coordinates is:
        (x1, x2, y1, y2) -> (x1, x2, H_full - y2, H_full - y1)

    mode:
      - 'flip' : always apply the mapping
      - 'none' : never apply the mapping
      - 'auto' : default to applying the mapping (prints a note so you can switch to 'none')
    """
    x1, x2, y1, y2 = map(int, pts)

    if mode not in {"auto", "flip", "none"}:
        raise ValueError(f"Unknown flip_y mode: {mode}")

    do_flip = (mode == "flip") or (mode == "auto")
    if not do_flip:
        return (x1, x2, y1, y2)

    y1n = int(H_full) - int(y2)
    y2n = int(H_full) - int(y1)

    # Ensure ordering
    if y2n < y1n:
        y1n, y2n = y2n, y1n

    return (x1, x2, y1n, y2n)


def _plot_full_and_patch(
    x_full_1ch: torch.Tensor,
    hr_points: Tuple[int, int, int, int],
    ctx_patch_1ch: torch.Tensor,
    outpath: Path,
    *,
    hr_patch_1ch: torch.Tensor | None = None,
    title: str = "Option2 spatial conditioner (dataset)",
    lsm_path: Optional[str] = None,
) -> None:
    """Save a plot that verifies placement + orientation.

    Panels:
      A) full-domain LR (ch0) + HR rectangle + DK outline
      B) LR crop at hr_points + DK outline
      C) ctx_patch feature channel 0
      D) optional: HR target patch + DK outline

    Notes:
      - Points are [x1,x2,y1,y2] and slicing is [y1:y2, x1:x2].
      - We always plot with origin='lower' to match sbgm.plotting_utils conventions.
    """
    full = x_full_1ch.detach().cpu().float().numpy()
    patch = ctx_patch_1ch.detach().cpu().float().numpy()

    x1, x2, y1, y2 = [int(v) for v in hr_points]

    Hf, Wf = int(full.shape[-2]), int(full.shape[-1])
    x1c, x2c = max(0, x1), min(Wf, x2)
    y1c, y2c = max(0, y1), min(Hf, y2)
    lr_crop = full[y1c:y2c, x1c:x2c]

    # --- DK outline masks (optional) ---
    mask_full = None
    mask_crop = None
    if lsm_path:
        try:
            mask_full = get_dk_lsm_outline(lsm_path=str(lsm_path), bounds=(0, Hf, 0, Wf))
            mask_crop = get_dk_lsm_outline(lsm_path=str(lsm_path), bounds=(y1c, y2c, x1c, x2c))
        except Exception:
            mask_full = None
            mask_crop = None

    have_hr = hr_patch_1ch is not None
    ncols = 4 if have_hr else 3
    fig, axs = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    # A) full + rect + outline
    im0 = imshow_variable(
        axs[0],
        full,
        variable="prcp",  # only used to pick a sane default cmap; this is just a debugging view
        bounds=(0, Hf, 0, Wf),
        lsm_path=str(lsm_path) if lsm_path else None,
        add_dk_outline=False,
    )
    axs[0].set_title("LR full (ch0) + HR rect")
    rect = Rectangle((x1, y1), width=(x2 - x1), height=(y2 - y1), fill=False, linewidth=2.0)
    axs[0].add_patch(rect)
    axs[0].scatter([(x1 + x2) / 2.0], [(y1 + y2) / 2.0], s=25)
    overlay_outline(axs[0], mask_full, color="darkgrey", linewidth=0.8)
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    # B) LR crop + outline
    im1 = axs[1].imshow(lr_crop, origin="lower")
    axs[1].set_title(f"LR crop @ hr_points\nshape={lr_crop.shape}")
    overlay_outline(axs[1], mask_crop, color="darkgrey", linewidth=0.8)
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    # C) ctx_patch
    im2 = axs[2].imshow(patch, origin="lower")
    axs[2].set_title("ctx_patch (feat ch0)")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    # D) HR patch (optional) + outline
    if have_hr:
        hrp = hr_patch_1ch.detach().cpu().float().numpy()
        im3 = axs[3].imshow(hrp, origin="lower")
        axs[3].set_title(f"HR target patch\nshape={hrp.shape}")
        overlay_outline(axs[3], mask_crop, color="darkgrey", linewidth=0.8)
        plt.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150)  # type: ignore
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True, help="Path to YAML config")
    ap.add_argument("--split", type=str, default="train", choices=["train", "valid", "gen"], help="Which loader to sample from")
    ap.add_argument("--batch_idx", type=int, default=0, help="Which batch to take (0 = first)")
    ap.add_argument("--idx", type=int, default=0, help="Index within selected batch (0..B-1)")
    ap.add_argument("--out", type=str, default=None, help="Optional output dir; defaults to cfg.paths.path_save")
    ap.add_argument("--inspect_points", action="store_true", help="Print hr_points/lr_points raw contents and exit")
    ap.add_argument(
        "--flip_y",
        type=str,
        default="auto",
        choices=["auto", "flip", "none"],
        help=(
            "Whether to flip y in hr_points before cropping/plotting. "
            "Use 'flip' if the HR rectangle appears mirrored vertically; use 'none' if it appears correct. "
            "'auto' currently defaults to applying the flip and prints a note."
        ),
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.cfg).read_text())

    train_loader, val_loader, gen_loader = get_dataloader(cfg, verbose=True)
    loader = {"train": train_loader, "valid": val_loader, "gen": gen_loader}[args.split]

    batch = None
    for bi, b in enumerate(loader):
        if bi == args.batch_idx:
            batch = b
            break
    if batch is None:
        raise RuntimeError(f"Could not fetch batch_idx={args.batch_idx} from loader '{args.split}'.")

    if args.inspect_points:
        print("\n[inspect_points] batch keys:", list(batch.keys()))
        for k in ["hr_points", "lr_points"]:
            if k not in batch:
                print(f"[inspect_points] missing key: {k}")
                continue
            v = batch[k]
            print(f"\n[inspect_points] {k}: type={type(v)}")
            if torch.is_tensor(v):
                print(f"[inspect_points] {k}: shape={tuple(v.shape)} dtype={v.dtype}")
            else:
                nshow = min(4, len(v))
                for i in range(nshow):
                    print(f"  {k}[{i}]: {v[i]}")
        return

    # --- Select full-domain tensors and points ---
    full_hw = tuple(cfg["lowres"]["full_domain_dims"])
    x_full, used_keys = _choose_full_domain_lr_tensors(batch, full_hw)
    x_full_1ch = x_full[args.idx, 0]  # [H,W]

    hr_raw4 = _as_int_points(batch["hr_points"], sample_idx=args.idx)
    lr_raw4 = _as_int_points(batch["lr_points"], sample_idx=args.idx)

    hr_points = _resolve_points_to_xxyy(
        hr_raw4,
        full_hw=full_hw,
        hr_hw=tuple(cfg["highres"]["data_size"]),
        cfg=cfg,
    )

    print(f"[dataset opt2] hr_points_raw    : {hr_raw4}")
    print(f"[dataset opt2] hr_points(xxyy)  : {hr_points}")

    # Build spatial conditioner
    cond = SpatialContextConditioner(
        in_channels=x_full.shape[1],
        c_base=32,
        c_feat=16,
        c_global=128,
        n_down=3,
        hr_size=(cfg["highres"]["data_size"][0], cfg["highres"]["data_size"][1]),
    )

    # Run conditioner on full-domain LR, crop patch at hr_points
    ctx_patch, ctx_global = cond(x_full, hr_points=hr_points)
    ctx_patch_1ch = ctx_patch[0, 0]  # [H_hr, W_hr]

    # Optionally: get HR patch for panel D
    hr_patch_1ch = None
    if "prcp_hr" in batch:
        hr_patch = batch["prcp_hr"][args.idx]
        if hr_patch.ndim == 3:
            hr_patch_1ch = hr_patch[0]
        elif hr_patch.ndim == 2:
            hr_patch_1ch = hr_patch

    outdir = Path(args.out) if args.out else Path(cfg["paths"]["path_save"])
    outpath = outdir / f"spatial_ctx_debug_{args.split}_idx{args.idx}.png"

    _plot_full_and_patch(
        x_full_1ch,
        hr_points,
        ctx_patch_1ch,
        outpath,
        hr_patch_1ch=hr_patch_1ch,
        title=f"Option2 spatial conditioner (dataset, split={args.split})\nLR keys: {', '.join(used_keys)}",
        lsm_path=str(cfg.get("paths", {}).get("lsm_path", "")) or None,
    )

    print(f"[dataset opt2] saved plot -> {outpath}")


if __name__ == "__main__":
    main()
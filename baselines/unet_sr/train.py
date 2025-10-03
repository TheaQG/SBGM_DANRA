# sbgm/baselines/unet_sr/train.py
from __future__ import annotations
import numpy as np
import json
import logging
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from baselines.unet_sr.model import TinyUNet
from baselines.plotting import plotting_enabled, plotting_params, resolve_samples_dir, plot_triplet

logger = logging.getLogger(__name__)

def evaluate(model, loader, loss_fn, device):
    """
        Evaluate model on the given data split using the provided loss function.
        Returns average loss over the split.
    """
    logger.debug("[UNetSR][eval] Starting evaluation...")
    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x_in, y, lsm = batch.x_in.to(device), batch.y.to(device), batch.lsm.to(device).bool()
            yhat = model(x_in)
            perpx = loss_fn(yhat, y)  # [B,1,H,W]
            loss  = (perpx[lsm]).mean().item()
            tot += loss; cnt += 1
    model.train()
    return tot / max(cnt,1)


def save_split_outputs(model, loader, out_root: Path, device: torch.device, cfg):
    """
        Saves model predictions on the given data split to out_root in a generation-like folder structure:
            out_root/
                pmm/{date}.npz -> {'pmm': [1,1,H,W]}
                lr_hr/{date}.npz -> {'hr': [1,1,H,W], 'lr_hr': [1,1,H,W]}
                lsm/{date}.npz -> {'lsm': [1,1,H,W]}
                meta/manifest.json -> {'date': {date}, 'shape': [1,1,H,W]}
    """
    hr_var = cfg.get('highres', {}).get('variable', cfg.get('data', {}).get('var','prcp'))
    logger.info(f"[UNetSR] Saving outputs to: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / 'pmm').mkdir(parents=True, exist_ok=True)
    (out_root / 'lr_hr').mkdir(parents=True, exist_ok=True)
    (out_root / 'lsm').mkdir(parents=True, exist_ok=True)
    (out_root / 'meta').mkdir(parents=True, exist_ok=True)

    # plotting setup (optional)
    want_plot = plotting_enabled(cfg)
    params = plotting_params(cfg) if want_plot else {}
    plotted = 0
    if want_plot:
        # assume test split for now; adjust if you call with different split names
        sample_dir = resolve_samples_dir(cfg, baseline_type="unet_sr", split="test")
        logger.info(f"[UNetSR][plot] Enabled. Saving examples to: {sample_dir}")
    else:
        sample_dir = None

    logger.info("[UNetSR][save] Generating predictions")
    model.eval()
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            x_in, y, lsm = batch.x_in.to(device), batch.y, batch.lsm
            dates = batch.date
            yhat = model(x_in).cpu().numpy()
            y_np = y.cpu().numpy()
            lsm_np = lsm.cpu().numpy()
            lr_np  = batch.lr_up.cpu().numpy()

            if b_idx == 0:
                B, _, H, W = yhat.shape if isinstance(yhat, np.ndarray) else (len(dates),) + tuple(batch.y.shape[1:])
                logger.info(f"[UNetSR][save] First batch: B={len(dates)}, HxW~{batch.y.shape[-2:]}")
            for i, d in enumerate(dates):
                np.savez_compressed(out_root / 'pmm' / f'{d}.npz', pmm=yhat[i:i+1])
                np.savez_compressed(out_root / 'lr_hr' / f'{d}.npz', hr=y_np[i:i+1], lr_hr=lr_np[i:i+1])
                np.savez_compressed(out_root / 'lsm' / f'{d}.npz', lsm=lsm_np[i:i+1])
                if (i + 1) % 200 == 0:
                    logger.info(f"[UNetSR][save] Saved {i+1} samples in current batch...")

                # optional plotting
                # optional plotting
                max_plots = int(params.get("max_plots", 0))
                if want_plot and plotted < max_plots and sample_dir is not None:
                    try:
                        vmax_param = params.get("vmax", None)
                        if isinstance(vmax_param, str):
                            try:
                                vmax_param = float(vmax_param)
                            except ValueError:
                                vmax_param = None
                        cmap_param = params.get("cmap", "Blues")
                        plot_triplet(hr_var=hr_var,
                            date=d,
                            pmm=yhat[i:i+1], hr=y_np[i:i+1], lr=lr_np[i:i+1], lsm=lsm_np[i:i+1],
                            out_dir=sample_dir,
                            vmax=vmax_param,
                            title_suffix="(UNet-SR)"
                        )
                        plotted += 1
                    except Exception as e:
                        logger.warning(f"[UNetSR][plot] Failed to plot sample {d}: {e}")

    logger.info("[UNetSR][save] Completed writing predictions.")


def run_unet_sr(cfg, adapter_train, adapter_val, adapter_test, out_root: Path):
    logger.info("[UNetSR] Initializing training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hp = cfg.get('baseline', {}).get('unet_sr', {})
    in_ch   = hp.get('in_channels', 3)
    out_ch  = hp.get('out_channels', 1)
    width   = hp.get('width', 48)
    depth   = hp.get('depth', 4)
    act     = hp.get('act', 'SiLU')
    residual= hp.get('residual', True)
    loss_nm = hp.get('loss', 'L1')
    lr      = hp.get('lr', 1.5e-3)
    bs      = hp.get('batch_size', 8)
    steps   = hp.get('max_steps', 20000)
    amp     = hp.get('amp', True)

    n_workers = cfg.get('num_workers', 4)

    logger.info(f"[UNetSR] Hyperparameters: in_ch={in_ch}, width={width}, depth={depth}, residual={residual}, loss={loss_nm}, lr={lr}, bs={bs}, steps={steps}, amp={amp}")

    # loaders
    tr_loader = adapter_train.make_loader(batch_size=bs, shuffle=True,  num_workers=n_workers)
    va_loader = adapter_val.make_loader(batch_size=bs, shuffle=False, num_workers=n_workers)
    te_loader = adapter_test.make_loader(batch_size=bs, shuffle=False, num_workers=n_workers)
    logger.info(f"[UNetSR] Data loaders ready (bs={bs}), num_workers={n_workers}")

    # === Infer input channels from a probe batch BEFORE creating model ===
    try:
        probe = next(iter(tr_loader))
        inferred_in_ch = int(probe.x_in.shape[1])
        if inferred_in_ch != in_ch:
            logger.warning(f"[UNetSR] Warning: inferred in_ch={inferred_in_ch} from data, but config specifies in_ch={in_ch}. Using inferred value.")
            in_ch = inferred_in_ch
        logger.debug(f"[UNetSR] Inferred input channels from data: in_ch={in_ch}")
    except Exception as e:
        logger.error(f"[UNetSR] Failed to infer input channels from data: {e}. Proceeding with config in_ch={in_ch}.")
    
    model = TinyUNet(in_ch, out_ch, width=width, depth=depth, act=act, residual=residual).to(device)

    # loss/opt
    loss_fn = nn.L1Loss(reduction='none') if loss_nm.upper()=='L1' else nn.MSELoss(reduction='none')
    opt = optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=amp)

    # train loop
    step, best = 0, 1e9
    model.train()
    while step < steps:
        for batch in tr_loader:
            x_in, y, lsm = batch.x_in.to(device), batch.y.to(device), batch.lsm.to(device).bool()
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                yhat = model(x_in)
                loss = (loss_fn(yhat, y)[lsm]).mean()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            step += 1
            if step % 100 == 0:
                logger.info(f"[UNetSR] Progress: step {step}/{steps}")
            if step % 20 == 0:
                logger.info(f"[{step}] train loss={loss.item():.4f}")
            if step % 200 == 0:
                v = evaluate(model, va_loader, loss_fn, device)
                logger.info(f"[{step}] val loss={v:.4f}")
                if v < best:
                    best = v
            if step >= steps:
                break

    logger.info("[UNetSR] Training complete. Saving predictions on test set...")
    run_name = cfg.get('experiment_name', 'baseline_unet_sr')

    logger.info(f"[UNetSR] Output base dir: {out_root}")
    # save_split_outputs(model, va_loader, out_root / 'val',  device)
    save_split_outputs(model, te_loader, out_root, device, cfg)
    logger.info("[UNetSR] Test split predictions saved.")

    (out_root / 'meta').mkdir(parents=True, exist_ok=True)
    manifest = {
        "baseline": "unet_sr",
        "steps": steps,
        "best_val": best,
        "in_channels": in_ch,
        "width": width, "depth": depth, "residual": residual,
    }
    (out_root / 'meta' / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return {"manifest": manifest, "out_root": str(out_root)}
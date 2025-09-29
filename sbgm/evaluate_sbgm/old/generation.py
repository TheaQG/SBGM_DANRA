'''
    Script to run evaluations of trained ddpm models on test DANRA dataset.
    Default evalutes on evaluation set of size equal to two years of data (730 samples), 2001-2002.
    The default size is 64x64.
    Script only creates samples, does not evaluate or plot.

    !!! MISSING: No shift in image, steady over specific area of DK
    


'''

import torch
import os
import logging

import numpy as np
from matplotlib import pyplot as plt

from sbgm.score_sampling import pc_sampler, edm_sampler
from sbgm.score_unet import marginal_prob_std_fn, diffusion_coeff_fn
from sbgm.utils import extract_samples, get_model_string, get_first_sample_dict
from sbgm.plotting_utils import plot_samples_and_generated

logger = logging.getLogger(__name__)

def maybe_inverse_transform(k, arr, back_transforms):
    """    
        Apply inverse transformation if available for the given key.
    """
    logger.info(f"Applying inverse transformation for key: {k}")
    if back_transforms and k in back_transforms:
        logger.info(f"Found inverse transformation for key: {k}")
        return back_transforms[k](arr)
    logger.info(f"No inverse transformation found for key: {k}")
    return arr




class SampleGenerator:
    def __init__(self, cfg, model, dataloader, back_transforms, device):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.back_transforms = back_transforms
        self.device = device
        self.model.eval()

        self.model_name_str = get_model_string(cfg)
        self.output_dir = os.path.join(cfg.paths.sample_dir, 'generation', self.model_name_str)
        self.fig_path = os.path.join(self.output_dir, 'generated_figures')
        self.sample_path = os.path.join(self.output_dir, 'generated_samples')
        os.makedirs(self.fig_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)

    # Helper to extract LR upsampled baseline in HR z-space if using residual prediction (like in training)
    def _build_lr_ups_baseline(self, cond_images: torch.Tensor | None) -> torch.Tensor | None:
        # Return [B,1,H,W] baseline in HR z-space if enabled; else None
        if not bool(getattr(self.cfg, 'edm', {}).get('predict_residual', False)):
            return None
        if cond_images is None:
            raise ValueError("cond_images is None; cannot extract residual baseline.")

        hr_var = self.cfg.highres.variable
        lr_vars = list(self.cfg.lowres.condition_variables)
        # Handle common aliases
        alias = {('prcp','tp'),('tp','prcp'),('temp','t2m'),('t2m','temp')}
        main_name = hr_var if hr_var in lr_vars else next((b for a,b in alias if a==hr_var and b in lr_vars), None)
        if main_name is None:
            raise ValueError(f"Target '{hr_var}' not found among LR variables {lr_vars} for baseline.")

        # Figure out which channel(s) in cond_images correspond to main_name (dual-LR aware)
        dual_lr = bool(self.cfg.lowres.get('dual_lr', False))
        idx_main = lr_vars.index(main_name)
        # Walk channels: dual main consumes 2 channels
        ch = 0
        sl = None
        for j, v in enumerate(lr_vars):
            span = 2 if (dual_lr and v == main_name) else 1
            if j == idx_main:
                sl = slice(ch, ch+span)  # span==1 or 2
                break
            ch += span
        if sl is None:
            raise RuntimeError("Could not determine slice for main_name in cond_images.")
        lr_in_lr_space = cond_images[:, sl, :, :]
        # If dual, the first channel is HR+LR z-space, the second is LR-only; we want LR-only as starting point.
        if lr_in_lr_space.shape[1] == 2:
            lr_in_lr_space = lr_in_lr_space[:, 1:2]  # pick LR-space channel
        else:
            lr_in_lr_space = lr_in_lr_space[:, 0:1]

        # Map LR-space → HR z-space using your stats (same as training)
        from sbgm.special_transforms import lr_baseline_to_hr_zspace
        dom_hr = f"{self.cfg.highres.full_domain_dims[0]}x{self.cfg.highres.full_domain_dims[1]}" if self.cfg.highres.full_domain_dims else "full_domain"
        dom_lr = f"{self.cfg.lowres.full_domain_dims[0]}x{self.cfg.lowres.full_domain_dims[1]}" if self.cfg.lowres.full_domain_dims else "full_domain"
        crop_hr = '_'.join(map(str, self.cfg.highres.cutout_domains)) if self.cfg.highres.cutout_domains else "no_crop"
        crop_lr = '_'.join(map(str, self.cfg.lowres.cutout_domains)) if self.cfg.lowres.cutout_domains else "no_crop"

        baseline_hr_z = lr_baseline_to_hr_zspace(
            lr_chan_norm             = lr_in_lr_space,
            lr_variable              = hr_var,
            lr_model                 = self.cfg.lowres.model,
            lr_domain_str            = dom_lr,
            lr_crop_region_str       = crop_lr,
            lr_split                 = self.cfg.transforms.get('scaling_split', 'train'),
            lr_scaling_method        = self.cfg.lowres.scaling_methods[ idx_main ],
            lr_buffer_frac           = self.cfg.lowres.get('buffer_frac', 0.0),
            lr_stats_dir_root        = self.cfg.paths.stats_load_dir,
            hr_variable              = hr_var,
            hr_model                 = self.cfg.highres.model,
            hr_domain_str            = dom_hr,
            hr_crop_region_str       = crop_hr,
            hr_split                 = self.cfg.transforms.get('scaling_split', 'train'),
            hr_scaling_method        = self.cfg.highres.scaling_method,
            hr_buffer_frac           = self.cfg.highres.get('buffer_frac', 0.0),
            hr_stats_dir_root        = self.cfg.paths.stats_load_dir,
            eps                      = self.cfg.transforms.get('prcp_eps', 0.01),
        )
        return baseline_hr_z

    def _run_sampler(self, batch_size, y, cond_img, lsm_cond, topo_cond):
        """
            Run the correct sampler depending on whether EDM is enabled.
        """
        use_edm = bool(getattr(self.cfg, 'edm', {}).get('enabled', False))

        if use_edm:
            edm_cfg = getattr(self.cfg, 'edm', {})
            lr_ups_baseline = self._build_lr_ups_baseline(cond_img)  # [B,1,H,W] or None

            logger.info("[Sampler] Using EDM sampler...")
            gen_sample = edm_sampler(score_model=self.model,
                                     batch_size=batch_size,
                                     num_steps=self.cfg.sampler.n_timesteps,
                                     device=self.device,
                                     img_size=self.cfg.highres.data_size[0],
                                     # conditioning
                                     y=y,
                                     cond_img=cond_img,
                                     lsm_cond=lsm_cond,
                                     topo_cond=topo_cond,
                                     lr_ups=lr_ups_baseline,
                                     # EDM schedule and churn params w safe defaults
                                     sigma_min=float(edm_cfg.get('sigma_min', 0.002)),
                                     sigma_max=float(edm_cfg.get('sigma_max', 80.0)),
                                     rho=float(edm_cfg.get('rho', 7.0)),
                                     S_churn=float(edm_cfg.get('S_churn', 0.0)),
                                     S_min=float(edm_cfg.get('S_min', 0.0)),
                                     S_max=float(edm_cfg.get('S_max', 99999.0)),
                                     S_noise=float(edm_cfg.get('S_noise', 1.0)),
                                     )
        else:
            logger.info("[Sampler] Using VE-DSM predictor-corrector sampler...")

            gen_sample = pc_sampler(
                score_model=self.model,
                marginal_prob_std=marginal_prob_std_fn,
                diffusion_coeff=diffusion_coeff_fn,
                batch_size=batch_size,
                num_steps=self.cfg.sampler.n_timesteps,
                device=self.device,
                img_size=self.cfg.highres.data_size[0],
                y=y,
                cond_img=cond_img,
                lsm_cond=lsm_cond,
                topo_cond=topo_cond,
                )
            
        gen = gen_sample.detach().cpu()

        # Normalize to [B, H, W] explicitly
        if gen.ndim == 4 and gen.shape[1] == 1:
            gen = gen[:, 0] # [B, H, W]
        elif gen.ndim == 3:
            pass # [B, H, W], all good
        elif gen.ndim == 2:
            gen = gen.unsqueeze(0) # add batch dim back
        else:
            raise ValueError(f"Unknown generated sample shape: {gen.shape}")

        return gen

    def _save_cond_images_per_var(self, cond_images: torch.Tensor):
        if cond_images is None:
            return
        lr_vars = list(self.cfg.lowres.condition_variables)
        hr_var = self.cfg.highres.variable
        dual_lr = bool(self.cfg.lowres.get('dual_lr', False))
        # alias mapping
        alias = {('prcp','tp'),('tp','prcp'),('temp','t2m'),('t2m','temp')}
        main_name = hr_var if hr_var in lr_vars else next((b for a,b in alias if a==hr_var and b in lr_vars), None)

        c = 0
        for v in lr_vars:
            span = 2 if (dual_lr and v == main_name) else 1
            chans = cond_images[:, c:c+span, :, :]
            if span == 1:
                npz = chans[:, 0].cpu().numpy()  # [B,H,W]
                path = os.path.join(self.sample_path, f'cond_samples_{v}.npz')
                np.savez_compressed(path, npz)
                logger.info(f"Saved cond var '{v}' to {path}")
            else:
                # save split channels for clarity
                npz0 = chans[:, 0].cpu().numpy()  # [B,H,W]
                npz1 = chans[:, 1].cpu().numpy()  # [B,H,W]
                np.savez_compressed(os.path.join(self.sample_path, f'cond_samples_{v}_ch0.npz'), npz0)
                np.savez_compressed(os.path.join(self.sample_path, f'cond_samples_{v}_ch1.npz'), npz1)
                logger.info(f"Saved cond var '{v}' channels to {self.sample_path}")
            c += span

    def _apply_backtransforms(self, x, generated, cond_images, seasons):
        """
        Back-transform HR target and generated to physical space using `{hr_var}_hr`.
        If `cfg.evaluation.transform_lr_conditions` is True, also back-transform
        LR condition channels using available keys `{var}_lr` (LR stats) and `{var}_hr` (HR stats).
        This avoids relying on non-existent keys like `lr_lr`/`lr_hr`.
        Dual-LR aware: main LR var contributes 2 channels: ch0 (HR+LR z) and ch1 (LR z).
        We map ch0 using `cfg.evaluation.lr_dual_bt_target` (default 'HR') and ch1 using `{var}_lr`.
        """

        # ---------- helpers ----------
        def _bt_batch_with_key(var_key: str, arr):
            """Back-transform a batch of 2D arrays using back_transforms[var_key] if present; else passthrough."""
            if arr is None:
                return None
            t = arr if isinstance(arr, torch.Tensor) else torch.as_tensor(arr)
            # normalize to [B,H,W]
            if t.ndim == 2:
                t = t.unsqueeze(0)
            if t.ndim == 4 and t.shape[1] == 1:
                t = t[:, 0]
            elif t.ndim != 3:
                raise ValueError(f"Unexpected tensor shape for key {var_key}: {t.shape}")
            out = []
            for i in range(t.shape[0]):
                out_i = maybe_inverse_transform(var_key, t[i], self.back_transforms) if (self.back_transforms and (var_key in self.back_transforms)) else t[i]
                out_i = torch.as_tensor(out_i) if not torch.is_tensor(out_i) else out_i
                out.append(out_i)
            return torch.stack(out, dim=0)  # [B,H,W]
        def _bt_hr(arr):
            return _bt_batch_with_key(self.cfg.highres.variable + '_hr', arr)
        
        # ---------- HR back-transform (always) ----------
        x_bt   = _bt_hr(x)
        gen_bt = _bt_hr(generated)
        # ---------- LR conditions (optional) ----------
        transform_lr_conds = bool(getattr(self.cfg, 'evaluation', {}).get('transform_lr_conditions', False))
        if (cond_images is None) or (not transform_lr_conds):
            return x_bt, gen_bt, cond_images

        cond = cond_images if isinstance(cond_images, torch.Tensor) else torch.as_tensor(cond_images)
        if cond.ndim != 4:
            logger.warning(f"cond_images has unexpected shape {cond.shape}; skipping LR back-transform.")
            return x_bt, gen_bt, cond_images

        lr_vars = list(self.cfg.lowres.condition_variables)
        hr_var  = self.cfg.highres.variable
        dual_lr = bool(self.cfg.lowres.get('dual_lr', False))

        # Identify which LR var is the 'main' paired with HR target
        alias = {('prcp','tp'),('tp','prcp'),('temp','t2m'),('t2m','temp')}
        main_name = hr_var if hr_var in lr_vars else next((b for a,b in alias if a==hr_var and b in lr_vars), None)

        # Configurable mapping for dual ch0 (HR+LR z): target space for inverse
        lr_dual_bt_target = str(getattr(self.cfg, 'evaluation', {}).get('lr_dual_bt_target', 'HR')).upper()  # 'HR' or 'LR'
        if lr_dual_bt_target not in ('HR','LR'):
            lr_dual_bt_target = 'HR'

        B, C, H, W = cond.shape
        parts = []
        c = 0
        for v in lr_vars:
            if dual_lr and (v == main_name):
                # Expect two channels: ch0 (HR+LR z), ch1 (LR z)
                if c + 1 >= C:
                    logger.warning(f"Dual-LR expected two channels for '{v}', but found C={C}; skipping remaining.")
                    break
                ch0 = cond[:, c+0, :, :]
                ch1 = cond[:, c+1, :, :]
                # ch0 → choose mapping key based on cfg (default HR)
                key0 = f"{v}_hr" if lr_dual_bt_target == 'HR' else f"{v}_lr"
                key1 = f"{v}_lr"  # ch1 is LR z → physical via LR stats
                ch0_bt = _bt_batch_with_key(key0, ch0)
                ch1_bt = _bt_batch_with_key(key1, ch1)
                if ch0_bt is not None:
                    parts.append(ch0_bt.unsqueeze(1))
                if ch1_bt is not None:
                    parts.append(ch1_bt.unsqueeze(1))
                c += 2
            else:
                # Single channel variable; use LR stats to go to physical
                if c >= C:
                    logger.warning(f"Channel index {c} exceeds cond C={C} for var '{v}'.")
                    break
                ch = cond[:, c, :, :]
                key = f"{v}_lr" if v != main_name else (f"{v}_lr" if not dual_lr else f"{v}_lr")
                ch_bt = _bt_batch_with_key(key, ch)
                if ch_bt is not None:
                    parts.append(ch_bt.unsqueeze(1))
                c += 1

        if len(parts) == 0:
            return x_bt, gen_bt, cond_images

        cond_bt = torch.cat(parts, dim=1)
        return x_bt, gen_bt, cond_bt


    # def _apply_backtransforms(self, x, generated, cond_images, seasons):
    #     """
    #     Back-transform ONLY the HR target (x) and the generated field, in HR variable space.
    #     - Do NOT transform LR conditionals: with dual-LR, channels live in different z-spaces.
    #     - Robust to shapes:
    #         x:         [B,1,H,W] | [B,H,W] | [H,W]
    #         generated: [B,1,H,W] | [B,H,W] | [H,W]  (B may differ from x in repeated mode)
    #     Returns:
    #         x_bt, generated_bt, cond_images (unchanged)
    #     """
    #     hr_var_name = self.cfg.highres.variable + '_hr'

    #     def _bt_batch(arr):
    #         # Normalize to [B,H,W]
    #         if torch.is_tensor(arr):
    #             t = arr
    #         else:
    #             t = torch.as_tensor(arr)
    #         if t.ndim == 2:                        # [H,W]
    #             t = t.unsqueeze(0)                 # [1,H,W]
    #         if t.ndim == 4 and t.shape[1] == 1:    # [B,1,H,W]
    #             t = t[:, 0]                        # [B,H,W]
    #         elif t.ndim != 3:
    #             raise ValueError(f"Unexpected tensor shape in back-transform: {t.shape}")

    #         # Back-transform per sample, preserving batch size
    #         out_list = []
    #         for i in range(t.shape[0]):
    #             out_i = maybe_inverse_transform(hr_var_name, t[i], self.back_transforms)  # expects [H,W]
    #             out_i = torch.as_tensor(out_i) if not torch.is_tensor(out_i) else out_i
    #             out_list.append(out_i)
    #         return torch.stack(out_list, dim=0)     # [B,H,W]

    #     x_bt = _bt_batch(x)
    #     gen_bt = _bt_batch(generated)

    #     # Leave cond_images as-is (model-space), because dual-LR channels cannot be safely mapped here.
    #     return x_bt, gen_bt, cond_images

    def _plot_and_save(self,
                        samples,
                        generated,
                        name_suffix,
                        n_samples=None,
                        transform_back_bf_plot=False,
                        back_transforms=None,
                        epoch: int | None = None
                        ):
        """
        Plot and save generated samples in a way that mirrors the training pipeline.
        - Respects a plotting cap (n_samples) so we don't try to draw huge batches.
        - Ensures `generated` has a channel dimension for the plotter ([B,1,H,W]).
        - Uses the same entry-point `plot_samples_and_generated(samples, generated, cfg, ...)`.
        - Keeps the `samples` dict structure intact (keys like 'img', 'img_cond', 'lsm', 'topo', 'lsm_hr', etc.).
        """
        # ---- Determine plot batch size ----
        # Infer batch from any tensor in samples
        def _infer_batch(d):
            for v in d.values():
                if torch.is_tensor(v) and v.ndim >= 3:
                    return v.shape[0]
            return 1

        B_all = _infer_batch(samples)
        B_cap = n_samples if (n_samples is not None) else getattr(self.cfg.evaluation, 'n_samples_threshold_plot', 16)
        B_plot = min(B_all, B_cap)

        # ---- Slice a view of samples to B_plot ----
        samples_plot: dict[str, torch.Tensor | object] = {}
        for k, v in samples.items():
            if torch.is_tensor(v):
                if v.ndim >= 3 and v.shape[0] > B_plot:
                    samples_plot[k] = v[:B_plot].detach().cpu()
                else:
                    samples_plot[k] = v.detach().cpu()
            else:
                samples_plot[k] = v  # leave as is (e.g., list of strings)

        # ---- Prepare generated to [B_plot,1,H,W] ----
        gen = generated
        if isinstance(gen, torch.Tensor):
            gen = gen.detach().cpu()
            # Normalize to [B,H,W]
            if gen.ndim == 2:
                gen = gen.unsqueeze(0)                    # [1,H,W]
            elif gen.ndim == 4 and gen.shape[1] == 1:
                gen = gen[:, 0]                           # [B,H,W]
            elif gen.ndim != 3:
                raise ValueError(f"Unexpected generated shape {gen.shape} for plotting")
            # Slice to B_plot if needed
            if gen.shape[0] > B_plot:
                gen = gen[:B_plot]
            # Add channel for plotter: [B,1,H,W]
            gen_plot = gen.unsqueeze(1)
        else:
            raise TypeError("generated must be a torch.Tensor")

        # ---- Call the shared plotter ----
        fig, _ = plot_samples_and_generated(
            samples_plot,
            gen_plot,
            cfg=self.cfg,
            transform_back_bf_plot=transform_back_bf_plot,
            back_transforms=back_transforms,
            n_samples_threshold=B_plot,
        )

        # === Save figure ===
        suffix = name_suffix if epoch is None else f"e{epoch:03d}_{name_suffix}"
        out_path = os.path.join(self.fig_path, f'gen_samples_{suffix}.png')
        fig.savefig(out_path, dpi=300)

        if getattr(self.cfg.evaluation, 'show_plots', False):
            plt.show()
        else:
            plt.close(fig)
        
        logger.info(f"Saved generated samples figure to {out_path}")

    def _save_npz(self, data_dict, name_suffix):
        for key, value in data_dict.items():
            if value is not None:
                path = os.path.join(self.sample_path, f'{key}_{name_suffix}.npz')
                np.savez_compressed(path, value.cpu().numpy() if torch.is_tensor(value) else value)
                logger.info(f"Saved {key}, {name_suffix} to {path}")

    def generate_multiple(self):
        '''
        Generate cfg.evaluation.batch_size different samples, based on different inputs from dataset
        '''
        samples = next(iter(self.dataloader))
        x, seasons, cond_images, lsm_hr, lsm, sdf, topo, hr_points, lr_points = extract_samples(samples, self.device)

        n = x.shape[0]
        generated = self._run_sampler(n, seasons, cond_images, lsm, topo)

        # Plotting before back transform to avoid confussion on transformations
        if self.cfg.evaluation.plot_examples:
            logger.info("Plotting examples...\n")
            self._plot_and_save(samples,
                                generated,
                                f"multi_n_{n}",
                                n_samples=self.cfg.evaluation.n_samples_threshold_plot,
                                transform_back_bf_plot=self.cfg.evaluation.transform_back,
                                back_transforms=self.back_transforms
                                )

        if self.cfg.evaluation.transform_back:
            x, generated, cond_images = self._apply_backtransforms(x, generated, cond_images, seasons)

        self._save_npz({
            "gen_samples": generated,
            "eval_samples": x,
            "lsm_samples": lsm,
            "seasons": seasons,
        }, f"multi_n_{n}")

        if cond_images is not None:
            # Ensure cond_images is a torch.Tensor before passing
            if not isinstance(cond_images, torch.Tensor):
                cond_images_tensor = torch.stack([torch.tensor(im) for im in cond_images])
            else:
                cond_images_tensor = cond_images
            self._save_cond_images_per_var(cond_images=cond_images_tensor)

        return

    def generate_single(self):
        '''
            Generate one (1) sample from the dataset
        '''
        samples = next(iter(self.dataloader))
        x, seasons, cond_images, lsm_hr, lsm, sdf, topo, hr_points, lr_points = extract_samples(samples, self.device)

        # Take only the first sample
        x = x[:1]
        seasons = seasons[:1] if seasons is not None else None
        lsm = lsm[:1] if lsm is not None else None
        topo = topo[:1] if topo is not None else None
        cond_images = cond_images[:1] if cond_images is not None else None
        
        generated = self._run_sampler(1, seasons, cond_images, lsm, topo)

        samples_single = get_first_sample_dict(samples)

        # Plotting before back transform to avoid confussion on transformations
        if self.cfg.evaluation.plot_examples:
            logger.info(f"Plotting examples...\n")
            self._plot_and_save(samples_single,
                                generated,
                                "single",
                                n_samples=1,
                                transform_back_bf_plot=self.cfg.evaluation.transform_back,
                                back_transforms=self.back_transforms
                                )
        logger.info(f"[DEBUG] Shape of generated before back transform: {generated.shape}")

        if self.cfg.evaluation.transform_back:
            x, generated, cond_images = self._apply_backtransforms(x, generated, cond_images, seasons)
            if generated is not None:
                logger.info(f"[DEBUG] Shape of generated after back transform: {generated.shape}")
            else:
                logger.info("[DEBUG] Generated is None after back transform.")

        self._save_npz({
            "gen_samples": generated,
            "eval_samples": x,
            "lsm_samples": lsm,
            "seasons": seasons,
        }, "single")

        if cond_images is not None:
            # Ensure cond_images is a torch.Tensor before passing
            if not isinstance(cond_images, torch.Tensor):
                cond_images_tensor = torch.stack([torch.tensor(im) for im in cond_images])
            else:
                cond_images_tensor = cond_images
            self._save_cond_images_per_var(cond_images=cond_images_tensor)

        return

    def generate_repeated(self):
        '''
            Generate cfg.evaluation.n_repeats samples from the same single sample from dataset
        '''
        samples = next(iter(self.dataloader))
        x, seasons, cond_images, lsm_hr, lsm, sdf, topo, hr_points, lr_points = extract_samples(samples, self.device)

        # Take only the first sample
        x = x[:1]
        seasons = seasons[:1] if seasons is not None else None
        lsm = lsm[:1] if lsm is not None else None
        topo = topo[:1] if topo is not None else None
        cond_images = cond_images[:1] if cond_images is not None else None

        n_repeats = self.cfg.evaluation.n_repeats
        generated_list = [self._run_sampler(1, seasons, cond_images, lsm, topo) for _ in range(n_repeats)]
        generated = torch.stack(generated_list)
        
        samples_repeated = {k: v[0].repeat(n_repeats, *[1 for _ in v.shape[1:]]) if torch.is_tensor(v) else v for k, v in samples.items()}

        if self.cfg.evaluation.plot_examples:
            self._plot_and_save(samples_repeated,
                                generated,
                                f"repeated_{n_repeats}",
                                n_samples=self.cfg.evaluation.n_samples_threshold_plot,
                                transform_back_bf_plot=self.cfg.evaluation.transform_back,
                                back_transforms=self.back_transforms
                                )

        if self.cfg.evaluation.transform_back:
            x, generated, cond_images = self._apply_backtransforms(x, generated, cond_images, seasons)  


        self._save_npz({
            "gen_samples": generated,
            "eval_samples": x,
            "lsm_samples": lsm,
            "seasons": seasons,
        }, f"repeated_{n_repeats}")

        
        if cond_images is not None:
            # Ensure cond_images is a torch.Tensor before passing
            if not isinstance(cond_images, torch.Tensor):
                cond_images_tensor = torch.stack([torch.tensor(im) for im in cond_images])
            else:
                cond_images_tensor = cond_images
            self._save_cond_images_per_var(cond_images=cond_images_tensor)
        return
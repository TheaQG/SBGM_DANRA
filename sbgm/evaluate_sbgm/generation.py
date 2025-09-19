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

    def _run_sampler(self, batch_size, y, cond_img, lsm_cond, topo_cond):
        """
            Run the correct sampler depending on whether EDM is enabled.
        """
        use_edm = bool(getattr(self.cfg, 'edm', {}).get('enabled', False))

        if use_edm:
            edm_cfg = getattr(self.cfg, 'edm', {})
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
                                     # EDM schedule and churn params w safe defaults
                                     sigma_min=float(edm_cfg.get('sigma_min', 0.002)),
                                     sigma_max=float(edm_cfg.get('sigma_max', 80.0)),
                                     rho=float(edm_cfg.get('rho', 7.0)),
                                     S_churn=float(edm_cfg.get('S_churn', 0.0)),
                                     S_min=float(edm_cfg.get('S_min', 0.0)),
                                     S_max=float(edm_cfg.get('S_max', 99999.0)),
                                     S_noise=float(edm_cfg.get('S_noise', 1.0)),
                                     lr_ups=None
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
            
        gen_sample = gen_sample.squeeze().detach().cpu()

        # Normalize output shape to [B, H, W]
        if gen_sample.ndim == 4:
            gen_sample = gen_sample.squeeze(1) # remove channel dim only !!! NOTE IF REWRITING TO MULTI CHANNEL OUTPUT THIS NEEDS TO GO !!!
        elif gen_sample.ndim == 3:
            pass # [B, H, W], all good
        elif gen_sample.ndim == 2:
            gen_sample = gen_sample.unsqueeze(0) # add batch dim back
        else:
            raise ValueError(f"Unknown generated sample shape: {gen_sample.shape}")

        return gen_sample

    def _apply_backtransforms(self, x, generated, cond_images, seasons):
        n = x.shape[0]
        hr_var_name = self.cfg.highres.variable + '_hr'

        # Make sure the back transform doesn't squeeze the generated if single sample (shape [H, W]) by expanding to [1, H, W]
        if generated.ndim == 2:
            generated = generated.unsqueeze(0)
        x = torch.stack([maybe_inverse_transform(hr_var_name, x[i], self.back_transforms) for i in range(n)])
        generated = torch.stack([maybe_inverse_transform(hr_var_name, generated[i], self.back_transforms) for i in range(n)])

        if cond_images is not None:
            cond_keys = self.cfg.lowres.condition_variables or []
            cond_images_btrans = []
            for i in range(n):
                cond_sample = cond_images[i]
                cond_var_btrans = []
                for k, cond_var in zip(cond_keys, cond_sample):
                    key = k + '_lr'
                    cond_var_btrans.append(maybe_inverse_transform(key, cond_var, self.back_transforms))
                cond_images_btrans.append(cond_var_btrans)
            cond_images = cond_images_btrans

        return x, generated, cond_images

    def _plot_and_save(self,
                        samples,
                        generated,
                        name_suffix,
                        n_samples=None,
                        transform_back_bf_plot=False,
                        back_transforms=None
                        ):
        '''
            Plot and save generated samples, supporting flexible sample count
        '''

        # Loop through samples and print shape
        for k, v in samples.items():
            if torch.is_tensor(v):
                logger.info(f"[DEBUG] Sample '{k}' shape: {v.shape}")
            else:
                logger.info(f"[DEBUG] Sample '{k}' value: {v}")

        logger.info(f"[DEBUG] Generated shape: {generated.shape}")

        # Ensure samples is a list of single-sample dicts
        if isinstance(samples, dict):
            batch_size = next((v.shape[0] for v in samples.values() if isinstance(v, torch.Tensor)), 1)
            sample_list = []
            for i in range(batch_size):
                sample_i = {}
                for k, v in samples.items():
                    if torch.is_tensor(v):
                        vi = v[i]
                        if vi.ndim == 2:
                            sample_i[k] = vi.unsqueeze(0)  # add batch dim
                        elif vi.ndim == 3:
                            sample_i[k] = vi  # already [C,H,W] or [B,H,W]
                        else:
                            sample_i[k] = vi  # leave as is; plotting will handle or skip
                    else:
                        sample_i[k] = v
                sample_list.append(sample_i)

        # Ensure generated is 3D or 4D with batch dim
        if isinstance(generated, torch.Tensor) and generated.ndim == 2:
            generated = generated.unsqueeze(0)

        # Determine how many samples to plot. Fallback is self.cfg.evaluation.n_samples_threshold_plot (we don't want to accidentally plot 5000 samples)
        threshold = n_samples if n_samples is not None else self.cfg.evaluation.n_samples_threshold_plot

        # Ensure generated is correct shape: [B, H. W]
        if isinstance(generated, torch.Tensor):
            if generated.ndim == 2:
                generated = generated.unsqueeze(0) # [1, H, W]
            elif generated.ndim == 4 and generated.shape[1] == 1:
                generated = generated.squeeze(1) # [B, H, W]

        fig, _ = plot_samples_and_generated(samples,
                                            generated.unsqueeze(1) if generated.ndim == 3 else generated,
                                            cfg=self.cfg,
                                            transform_back_bf_plot=transform_back_bf_plot,
                                            back_transforms=back_transforms,
                                            n_samples_threshold=threshold
                                            )
        fig.savefig(os.path.join(self.fig_path, f'gen_samples_{name_suffix}.png'), dpi=300)
        if self.cfg.evaluation.show_plots:
            plt.show()
        else:
            plt.close(fig)

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
            cond_keys = self.cfg.lowres.condition_variables or []
            for i, cond_key in enumerate(cond_keys):
                cond_tensor = torch.stack([torch.tensor(im[i]) for im in cond_images])
                self._save_npz({f'cond_samples_{cond_key}': cond_tensor}, f"multi_n_{n}")
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
            logger.info(f"[DEBUG] Shape of generated after back transform: {generated.shape}")

        self._save_npz({
            "gen_samples": generated,
            "eval_samples": x,
            "lsm_samples": lsm,
            "seasons": seasons,
        }, "single")

        if cond_images is not None:
            cond_keys = self.cfg.lowres.condition_variables or []
            for i, cond_key in enumerate(cond_keys):
                cond_tensor = torch.tensor(cond_images[0][i])
                self._save_npz({f"cond_samples_{cond_key}": cond_tensor}, "single")
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
            cond_keys = self.cfg.lowres.condition_variables or []
            for i, cond_key in enumerate(cond_keys):
                cond_tensor = torch.stack([torch.tensor(im[i]) for im in cond_images])
                self._save_npz({f"cond_samples_{cond_key}": cond_tensor}, f"repeated_{n_repeats}")
        return
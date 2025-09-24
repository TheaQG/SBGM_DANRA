"""
    TODO:
        - Implement mixed precision training 
        - Make precipitation evaluations only when precipitation is the target variable
"""

import os
import torch
import copy
import pickle
import tqdm
import logging 

import torch.nn as nn
import matplotlib.pyplot as plt

from torch.cuda.amp import autocast, GradScaler


from sbgm.special_transforms import build_back_transforms_from_stats, remap_between_scalings_from_stats, lr_baseline_to_hr_zspace
from sbgm.utils import get_model_string, extract_samples
from sbgm.plotting_utils import (
    get_cmaps,
    plot_samples_and_generated,
    plot_live_training_metrics,
    plot_fss_history,
    plot_psd_slope_history,
    plot_quantiles_wetday_history,
    )
from sbgm.monitoring import (
    report_precip_extremes,
    edm_cosine_metric,
    compute_fss_at_scales,
    compute_psd_slope,
    compute_q95_q99_and_wet_day,
    tensor_stats,
    save_histogram,
    plot_saved_histograms,
    )
from sbgm.score_sampling import Euler_Maruyama_sampler, pc_sampler, ode_sampler, edm_sampler
from sbgm.training_utils import get_loss_fn, apply_cfg_dropout
from sbgm.variable_utils import get_units

# Speed up conv algo selection on fixed input sizes
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
'''
    ToDo:
        - Add support for mixed precision training
        - Add support for EMA (Exponential Moving Average) of the model
        - Add support for custom weight initialization
'''

# Set up logging
logger = logging.getLogger(__name__)


class TrainingPipeline_general:
    '''
        Class for building a training pipeline for the SBGM.
        To run through the training batches in one epoch.
    '''

    def __init__(self,
                 model,
                 marginal_prob_std_fn,
                 diffusion_coeff_fn,
                 optimizer,
                 device,
                 lr_scheduler,
                 cfg
                 ):
        '''
            Initialize the training pipeline.
            Args:
                model: PyTorch model to be trained. 
                loss_fn: Loss function for the model. 
                optimizer: Optimizer for the model.
                device: Device to run the model on.
                weight_init: Weight initialization method.
                custom_weight_initializer: Custom weight initialization method.
                sdf_weighted_loss: Boolean to use SDF weighted loss.
                with_ema: Boolean to use Exponential Moving Average (EMA) for the model.
        '''
        # Store the full configuration for later use
        self.cfg = cfg

        self.writer = None  # Placeholder for TensorBoard writer, if needed

        # Set class variables
        self.model = model
        # Set debug_pre_sigma_div from cfg if exists, else default to True
        self.model.debug_pre_sigma_div = cfg['training'].get('debug_pre_sigma_div', False)

        self.marginal_prob_std_fn = marginal_prob_std_fn
        self.diffusion_coeff_fn = diffusion_coeff_fn
        self.optimizer = optimizer
        # self.loss_fn = loss_fn
        self.loss_fn = get_loss_fn(self.cfg, marginal_prob_std_fn_in=getattr(self, 'marginal_prob_std_fn', None))

        self.lr_scheduler = lr_scheduler

        self.scaling = cfg['transforms']['scaling']
        self.global_prcp_eps = cfg['transforms'].get('prcp_eps', 0.01)

        self.hr_var = cfg['highres']['variable']
        self.hr_scaling_method = cfg['highres']['scaling_method']
        self.full_domain_dims_hr = cfg['highres']['full_domain_dims']
        self.crop_region_hr = cfg['highres']['cutout_domains']

        self.lr_vars = cfg['lowres']['condition_variables']
        self.lr_scaling_methods = cfg['lowres']['scaling_methods']
        self.full_domain_dims_lr = cfg['lowres']['full_domain_dims']
        self.crop_region_lr = cfg['lowres']['cutout_domains']

        # Cache strings for stats lookups
        self._dom_hr_str = f"{self.full_domain_dims_hr[0]}x{self.full_domain_dims_hr[1]}" if self.full_domain_dims_hr is not None else "full_domain"
        self._dom_lr_str = f"{self.full_domain_dims_lr[0]}x{self.full_domain_dims_lr[1]}" if self.full_domain_dims_lr is not None else "full_domain"
        self._crop_hr_str = '_'.join(map(str, self.crop_region_hr)) if self.crop_region_hr is not None else "no_crop"
        self._crop_lr_str = '_'.join(map(str, self.crop_region_lr)) if self.crop_region_lr is not None else "no_crop"
        self._stats_root = self.cfg['paths']['stats_load_dir']
        self._hr_method_for_target = self.hr_scaling_method
        # Assume LR scaling methods is a list aligned with lr_vars; get method for the target variable
        if self.hr_var in self.lr_vars:
            idx_t = self.lr_vars.index(self.hr_var)
            self._lr_method_for_target = self.lr_scaling_methods[idx_t]
        else:
            self._lr_method_for_target = None  # Target variable not in LR vars
            logger.warning(f"HR target variable '{self.hr_var}' not found in LR condition variables {self.lr_vars}. Cannot determine LR scaling method for target - residuals may not be aligned.")

        # inject into dicts
        self.bt_gen_key = "generated"

        # --------------------------------------------------- assemble key order
        self.bt_hr_key = f"{self.hr_var}_hr"
        self.bt_lr_keys = [f"{var}_lr" for var in self.lr_vars]

        self.weight_init = cfg['training']['weight_init']
        self.custom_weight_initializer = cfg['training']['custom_weight_initializer']
        self.sdf_weighted_loss = cfg['training']['sdf_weighted_loss']
        
        # EMA parameters
        self.with_ema = cfg['training']['with_ema']
        self.ema_decay = float(cfg['training'].get('ema_decay', 0.9999)) # Default to 0.9999 if not specified
        if self.with_ema:
            self._init_ema()

        # Classifier free guidance config
        self.cfg_guidance = self.cfg.get('classifier_free_guidance', {})

        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Initialize weights if needed
        if self.weight_init:
            if self.custom_weight_initializer is not None:
                # Use custom weight initializer if provided
                self.model.apply(self.custom_weight_initializer)
            else:
                self.model.apply(self.xavier_init_weights)
            logger.info(f"→ Model weights initialized with {self.custom_weight_initializer.__name__ if self.custom_weight_initializer else 'Xavier uniform'} initialization.")

        # Set up checkpoint directory, name and path
        self.checkpoint_dir = cfg['paths']['checkpoint_dir']
        self.checkpoint_name = get_model_string(cfg) + '.pth.tar' 
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)

        # Create the checkpoint directory if it does not exist
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            logger.info(f"→ Checkpoint directory created at {self.checkpoint_dir}")
        else:
            logger.info(f"→ Checkpoint directory already exists at {self.checkpoint_dir}")

        # Set the model string based on the configuration
        self.model_string = get_model_string(cfg)

        # Set path to figures, samples, losses
        self.path_samples = cfg['paths']['path_save'] + '/samples/' + self.model_string
        self.path_losses = cfg['paths']['path_save'] + '/losses'
        self.path_figures = self.path_samples + '/Figures'
        # Metrics path
        self.path_metrics = os.path.join(self.path_figures, 'metrics')

        # Create the directories if they do not exist
        if not os.path.exists(self.path_samples):
            os.makedirs(self.path_samples)
            logger.info(f"→ Samples directory created at {self.path_samples}")
        if not os.path.exists(self.path_losses):
            os.makedirs(self.path_losses)
            logger.info(f"→ Losses directory created at {self.path_losses}")
        if not os.path.exists(self.path_figures):
            os.makedirs(self.path_figures)
            logger.info(f"→ Figures directory created at {self.path_figures}")
        if not os.path.exists(self.path_metrics):
            os.makedirs(self.path_metrics)
            logger.info(f"→ Metrics directory created at {self.path_metrics}")



        # === Monitoring: extreme precipitation values in generated samples ===
        monitor_cfg = cfg.get('monitoring', {})
        monitor_prcp = monitor_cfg.get('extreme_prcp', {})
        self.extreme_enabled = bool(monitor_prcp.get('enabled', True))
        self.extreme_threshold_mm = float(monitor_prcp.get('threshold_mm', 500.0)) # Threshold in mm for extreme precipitation
        self.extreme_every_step = int(monitor_prcp.get('every_steps', 50)) # Monitor every n steps
        self.extreme_backtransform = bool(monitor_prcp.get('back_transform', True)) # Backtransform samples before checking extremes
        self.extreme_log_first_n = int(monitor_prcp.get('log_first_n', 5)) # Log the first n extreme values in detail
        self.extreme_in_validation = bool(monitor_prcp.get('check_in_validation', True)) # Check extreme values in validation set as well
        self.extreme_clamp_in_gen = bool(monitor_prcp.get('clamp_in_generation', True)) # Clamp extreme values in generated samples to threshold

        try:
            full_domain_dims_str_hr = f"{self.full_domain_dims_hr[0]}x{self.full_domain_dims_hr[1]}" if self.full_domain_dims_hr is not None else "full_domain"
            full_domain_dims_str_lr = f"{self.full_domain_dims_lr[0]}x{self.full_domain_dims_lr[1]}" if self.full_domain_dims_lr is not None else "full_domain"
            crop_region_hr_str = '_'.join(map(str, self.crop_region_hr)) if self.crop_region_hr is not None else "no_crop"
            crop_region_lr_str = '_'.join(map(str, self.crop_region_lr)) if self.crop_region_lr is not None else "no_crop"

            self.back_transforms_train = build_back_transforms_from_stats(
                hr_var=self.hr_var,
                hr_model=cfg['highres']['model'],
                domain_str_hr=full_domain_dims_str_hr,
                crop_region_str_hr=crop_region_hr_str,
                hr_scaling_method=self.hr_scaling_method,
                hr_buffer_frac=cfg['highres']['buffer_frac'] if 'buffer_frac' in cfg['highres'] else 0.0,
                lr_vars=self.lr_vars,
                lr_model=cfg['lowres']['model'],
                lr_scaling_methods=self.lr_scaling_methods,
                domain_str_lr=full_domain_dims_str_lr,
                crop_region_str_lr=crop_region_lr_str,
                lr_buffer_frac=cfg['lowres']['buffer_frac'] if 'buffer_frac' in cfg['lowres'] else 0.0,
                split="all", # For now "all", but NOTE: needs to be "train" in future
                stats_dir_root=cfg['paths']['stats_load_dir'],
                eps=self.global_prcp_eps
            )
        except Exception as e:
            logger.warning(f"[monitor] Could not build back transforms for sentinel; will skip back_transform in training. Error: {e}")
            self.back_transforms_train = None


        # === EDM flags (used for residual baseline handling) ===
        self.edm_enabled = bool((cfg.get('edm', {}).get('enabled', False)))
        self.edm_predict_residual = bool((cfg.get('edm', {}).get('predict_residual', False)))

        # === Live, lightweight monitors (append in loop; plot occasionally) ===
        moncfg = cfg.get('monitoring', {})
        self.monitor_plot_every_n_epochs = int(moncfg.get('plot_every_n_epochs', 5))
        self.live_metrics = {
            'steps': [],
            'edm_cosine': [],
            'hr_lr_corr': []
        }
        self.eval_land_only = bool(cfg.get('evaluation', {}).get('eval_land_only', False))

        # Persistent histories of epoch-level monitors
        self.fss_hist: list[dict] = []
        self.psd_hist: list[dict] = []
        self.q_hist: list[dict] = []

        # Monitoring configuration
        cfg_mon__end_of_epoch = moncfg.get('end_of_epoch', {})
        self.fss_scales_km = list(cfg_mon__end_of_epoch.get('fss_km', [5, 10, 20])) # Scales in km for FSS
        self.fss_threshold_mm = float(cfg_mon__end_of_epoch.get('fss_threshold_mm', 1.0)) # Threshold in mm for FSS
        self.pixel_km = float(cfg_mon__end_of_epoch.get('grid_km_per_px', 2.5)) # Grid spacing in km/px
        self.wetday_thresh = float(cfg_mon__end_of_epoch.get('wet_day_threshold_mm', 0.1)) # Wet day threshold in mm/day
        self.psd_compare_to_hr = bool(cfg_mon__end_of_epoch.get('psd_compare_to_hr', True)) # Whether to compare LR/HR PSD slopes
        self.quantiles_compare_to_hr = bool(cfg_mon__end_of_epoch.get('quantiles_compare_to_hr', True)) # Whether to compare LR/HR quantiles


    def _build_lr_ups_baseline(self, cond_images: torch.Tensor | None):
        """
            Extract LR baseline channel (same variable as HR target) from cond_images and upsample to HR resolution.
            Ensure it is expressed in HR z-space (or HR min-max space) before using for residual EDM.
            Returns [B, 1, H, W] or raises if unavailable when predict_residual is True.
        """
        if cond_images is None:
            raise ValueError("cond_images is None, cannot extract LR baseline for residual prediction.")
        
        cond_vars = self.cfg['lowres']['condition_variables']
        target_var = self.hr_var
        if target_var not in cond_vars:
            raise ValueError(f"Target variable '{target_var}' not found in condition variables {cond_vars}, cannot extract LR baseline for residual prediction.")
        
        idx = cond_vars.index(target_var)
        if cond_images.shape[1] <= idx:
            raise ValueError(f"cond_images has shape {cond_images.shape}, cannot extract channel index {idx} for variable '{target_var}'.")
        lr_in_lr_space = cond_images[:, idx:idx+1, :, :]  # [B, 1, h, w] - cond images already upsampled to HR size

        if self.cfg.get('edm', {}).get('baseline_space', 'hr') == 'lr':
            logger.info(f"baseline_space requested is 'lr'; using LR baseline channel as-is in LR space for residual prediction.")
            return lr_in_lr_space  # Already in LR space, just upsampled to HR size
        
        # Else, need to convert from LR space to HR space 
        
        # Find the LR scaling method corresponding to baseline channel
        lr_method_for_baseline = self._lr_method_for_target 

        # Ensure lr_method_for_baseline is a string
        if lr_method_for_baseline is None:
            raise ValueError("LR scaling method for baseline is None. Cannot proceed with lr_baseline_to_hr_zspace. Please check your configuration.")

        logger.info(f"Converting LR baseline channel from LR space to HR space using lr_baseline_to_hr_zspace with LR method '{lr_method_for_baseline}' and HR method '{self.hr_scaling_method}'.")
        # Remap using transform/back-transform stack
        lr_in_hr_space = lr_baseline_to_hr_zspace(
            lr_chan_norm=lr_in_lr_space,
            # LR meta
            lr_variable=self.hr_var,
            lr_model=self.cfg['lowres']['model'],
            lr_domain_str=self._dom_lr_str,
            lr_crop_region_str=self._crop_lr_str,
            lr_split=self.cfg['transforms'].get('scaling_split', 'train'),
            lr_scaling_method=lr_method_for_baseline,
            lr_buffer_frac=self.cfg['lowres'].get('buffer_frac', 0.0),
            lr_stats_dir_root=self.cfg['paths']['stats_load_dir'],
            # HR meta
            hr_variable=self.hr_var,
            hr_model=self.cfg['highres']['model'],
            hr_domain_str=self._dom_hr_str,
            hr_crop_region_str=self._crop_hr_str,
            hr_split=self.cfg['transforms'].get('scaling_split', 'train'),
            hr_scaling_method=self.hr_scaling_method,
            hr_buffer_frac=self.cfg['highres'].get('buffer_frac', 0.0),
            hr_stats_dir_root=self.cfg['paths']['stats_load_dir'],

            eps=self.global_prcp_eps
        )

        return lr_in_hr_space

    def _assert_all_finite(self, name, t):
        if t is not None and not torch.isfinite(t).all():
            mn = t[torch.isfinite(t)].min().item() if torch.isfinite(t).any() else float('nan')
            mx = t[torch.isfinite(t)].max().item() if torch.isfinite(t).any() else float('nan')
            raise ValueError(f"Input '{name}' contains non-finite values. Min: {mn}, Max: {mx}")


    def xavier_init_weights(self, m):
        '''
            Xavier weight initialization.
            Args:
                m: Model to initialize weights for.
        '''

        # Check if the layer is a linear or convolutional layer
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            # Initialize weights with Xavier uniform
            nn.init.xavier_uniform_(m.weight)
            # If model has bias, initialize with 0.01 constant
            if m.bias is not None and torch.is_tensor(m.bias):
                m.bias.data.fill_(0.01)
    
    def _init_ema(self):
        """ 
            Initialize Exponential Moving Average (EMA) model as a deepcopy of the current model and freeze it.
        """
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.to(self.device)
        self.ema_model.eval() # Set to eval mode

        # Detach the EMA model parameters to not update them
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
        logger.info(f"→ EMA model initialized with decay {self.ema_decay}")
    
    @torch.no_grad()
    def _update_ema(self):
        """
            Exponential moving average (EMA) update: ema = d*ema + (1-d)*model
        """
        if not getattr(self, 'ema_model', None):
            return  # EMA not initialized
        d = self.ema_decay
        msd = self.model.state_dict()  # model state dict
        esd = self.ema_model.state_dict()  # ema model state dict
        for k in esd.keys():
            # Only update if floating-point tensors:
            if k in msd and esd[k].dtype.is_floating_point:
                esd[k].mul_(d).add_(msd[k], alpha=1 - d)

    def load_checkpoint(self,
                        checkpoint_path,
                        load_ema=False,
                        # If load_ema is True, load the EMA model parameters
                        # If load_ema is False, load the model parameters
                        device=None
                        ):
        '''
            Load a checkpoint from the given path. If load_ema = True and EMA exists, load EMA parameters into self.model
            Also restore the EMA model when enabled.
            Args:
                checkpoint_path: Path to the checkpoint file.
                device: Device to load the checkpoint on. If None, uses the current device.
        '''
        # Check if device is provided, if not, use the current device
        if device is None:
            device = self.device
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net_sd = checkpoint.get('network_params', None)  # Network state dict
        ema_sd = checkpoint.get('ema_network_params', None)  # EMA state dict if exists

        if load_ema and (ema_sd is not None):
            self.model.load_state_dict(ema_sd)
            logger.info(f"→ Loaded EMA model weights into the main model from checkpoint {checkpoint_path}")
        elif net_sd is not None:
            self.model.load_state_dict(net_sd)
            logger.info(f"→ Loaded model weights into the main model from checkpoint {checkpoint_path}")
        else:
            raise KeyError(f"Checkpoint at {checkpoint_path} does not contain 'network_params' or 'ema_network_params'.")
        


    def save_model(self,
                   dirname='./model_params',
                   filename='SBGM.pth'
                   ):
        '''
            Save the model parameters and EMA parameters (if available)
            Args:
                dirname: Directory to save the model parameters.
                filename: Filename to save the model parameters.
        '''
        # Create directory if it does not exist
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # Set state dictionary to save
        state_dicts = {
            'network_params': self.model.state_dict(),
            'optimizer_params': self.optimizer.state_dict()
        }

        if self.with_ema and hasattr(self, 'ema_model'):
            state_dicts['ema_network_params'] = self.ema_model.state_dict()

        return torch.save(state_dicts, os.path.join(dirname, filename))
    
    def train_batches(self,
              dataloader,
              epochs=10,
              current_epoch=1,
              verbose=True,
              use_mixed_precision=False
              ):
        '''
            Method to run through the training batches in one epoch.
            Args:
                dataloader: Dataloader to run through.
                verbose: Boolean to print progress.
                PLOT_FIRST: Boolean to plot the first image.
                SAVE_PATH: Path to save the image.
                SAVE_NAME: Name of the image to save.
                use_mixed_precision: Boolean to use mixed precision training.
        '''
                # If plot first, then plot an example of the data

        
        # Set model to training mode
        self.model.train()

        # Set initial loss to 0
        loss_sum = 0.0

        # Check if cuda is available and set scaler for mixed precision training if needed
        self.scaler = GradScaler() if torch.cuda.is_available() and use_mixed_precision else None

        # Set the progress bar
        pbar = tqdm.tqdm(dataloader, desc=f"Epoch {current_epoch}/{epochs}", unit="batch")
        # Iterate through batches in dataloader (tuple of images and seasons)
        for idx, samples in enumerate(pbar):
            # Samples is a dict with following available keys: 'img', 'classifier', 'img_cond', 'lsm', 'sdf', 'topo', 'points'
            # Extract samples
            x, seasons, cond_images, lsm_hr, lsm, sdf, topo, hr_points, lr_points = extract_samples(samples, self.device)

            # === EDM: build lr_ups_baseline if needed ===
            lr_ups_baseline = None
            if self.edm_enabled and self.edm_predict_residual:
                lr_ups_baseline = self._build_lr_ups_baseline(cond_images)  # [B, 1, H, W]



            # === Diagnostics checks: asserts and 
            # Check raw inputs for NaNs or Infs
            self._assert_all_finite('x', x)
            self._assert_all_finite('cond_images', cond_images)
            self._assert_all_finite('lr_ups_baseline', lr_ups_baseline)

            cfg_diagnostics = self.cfg.get("diagnostics", {})
            do_log = bool(cfg_diagnostics.get("per_batch_stats", False))
            every = int(cfg_diagnostics.get("log_every", 100))

            hr = x
            lr_hr = lr_ups_baseline
            # Get the lr_lr as the cond_image that corresponds to the hr_var, if available
            if cond_images is not None and (self.hr_var in self.cfg['lowres']['condition_variables']):
                idx_hr_in_cond = self.cfg['lowres']['condition_variables'].index(self.hr_var)
                lr_lr = cond_images[:, idx_hr_in_cond:idx_hr_in_cond+1, :, :]  # [B, 1, H, W]
            else:
                lr_lr = None

            residual = hr - lr_hr if (hr is not None and lr_hr is not None) else None
            if do_log and (idx % every == 0):
                tensor_stats(hr, "train/hr_norm")
                if lr_hr is not None:
                    tensor_stats(lr_hr, "train/lr_hr_norm")
                if residual is not None:
                    tensor_stats(residual, "train/residual_hr_space")
                
                # if (idx % (10 * every)) == 0:
                #     # Log histograms less frequently
                #     save_histogram(hr, "train/hr_norm", self.path_metrics, bins=100, range=None)
                #     if lr_hr is not None:
                #         save_histogram(lr_hr, "train/lr_hr_norm", self.path_metrics, bins=100, range=None)
                #     if residual is not None:
                #         save_histogram(residual, "train/residual_hr_space", self.path_metrics, bins=100, range=None)
                #     if lr_lr is not None:
                #         tensor_stats(lr_lr, "train/lr_lr_norm")
                #         save_histogram(lr_lr, "train/lr_lr_norm", self.path_metrics, bins=100, range=None)

            # OPTIONAL: Clamp warnings
            clamp_warn = float(cfg_diagnostics.get("warn_if_abs_gt", 15.0))
            if do_log and (idx % every == 0) and (clamp_warn > 0.0):
                mx = float(residual.abs().amax().item()) if residual is not None else float('nan')
                if mx > clamp_warn:
                    logger.warning(f"[diagnostics][train] Batch {idx}: |residual| max {mx:.2f} exceeds warn_if_abs_gt {clamp_warn}. Consider residual normalization, tail clamp or loss robustification.")



            # # === CFG dropout (training) ===
            cfg_guidance = self.cfg_guidance
            cfg_dropout_result = apply_cfg_dropout(
                cond_images, lsm, topo, seasons, lr_ups_baseline, cfg_guidance
            )
            if len(cfg_dropout_result) == 5:
                cond_images, lsm, topo, seasons, lr_ups_baseline = cfg_dropout_result
            elif len(cfg_dropout_result) == 4:
                cond_images, lsm, topo, seasons = cfg_dropout_result
            else:
                raise ValueError(f"apply_cfg_dropout returned unexpected tuple length: {len(cfg_dropout_result)}")

            # Zero gradients
            self.optimizer.zero_grad()

            # Log the shapes of the inputs for debugging
            for name, tensor in zip(['x', 'seasons', 'cond_images', 'lsm', 'topo'], [x, seasons, cond_images, lsm, topo]):
                if tensor is not None:
                    assert tensor.device == x.device, f"{name} is on device {tensor.device}, expected {x.device}"
            
            if hasattr(self, 'scaler') and self.scaler:
                with autocast():
                    # Pass the score model and samples+conditions to the loss_fn
                    batch_loss = self.loss_fn(self.model, # NOTE: Is this correct? Should I set ema_model somewhere?
                                               x,
                                               y=seasons,
                                               cond_img=cond_images,
                                               lsm_cond=lsm,
                                               topo_cond=topo,
                                               sdf_cond=sdf,
                                               lr_ups=lr_ups_baseline
                                               )
            else:
                # No mixed precision, just pass the score model and samples+conditions to the loss_fn
                batch_loss = self.loss_fn(self.model,
                                           x,
                                           y=seasons,
                                           cond_img=cond_images,
                                           lsm_cond=lsm,
                                           topo_cond=topo,
                                           sdf_cond=sdf,
                                           lr_ups=lr_ups_baseline
                                       )
            # Make sure loss is finite
            self._assert_all_finite('batch_loss', batch_loss)


            # === Cosine monitoring (lightweight) ===
            monitor_cfg = self.cfg.get('monitoring', {})
            log_every = monitor_cfg.get('edm_metrics_every', 50)
            global_step = (current_epoch - 1) * len(dataloader) + idx
            edm_on = self.cfg.get('edm', {}).get('enabled', False)

            if edm_on and log_every > 0 and (global_step % log_every == 0):
                cos = edm_cosine_metric(self.loss_fn, self.model, x, y=seasons, cond_img=cond_images, lsm_cond=lsm, topo_cond=topo, lr_ups=lr_ups_baseline)
                self.live_metrics['steps'].append(global_step)
                self.live_metrics['edm_cosine'].append(float(cos)) # type: ignore
                if cos is not None:
                    pbar.set_postfix(loss=loss_sum / (idx + 1), edm_cosine=cos)
                    if verbose:
                        logger.info(f"→ [monitor][train] Step {idx}: EDM cosine metric: {cos:.4f}")
                    if self.writer is not None:
                        self.writer.add_scalar('monitoring/edm_cosine_metric_train', cos, (current_epoch - 1) * len(dataloader) + idx)

                # HR <-> LR alignment corr (cheap). Use lr_ups_baseline already built for residual path, else derive it here if needed
                hr_lr_corr_val = float('nan')
                try: 
                    if lr_ups_baseline is None and cond_images is not None and (self.hr_var in self.cfg['lowres']['condition_variables']):
                        lr_ups_baseline = self._build_lr_ups_baseline(cond_images)  # [B, 1, H, W]
                    if lr_ups_baseline is not None:
                        # Optional land-only mask
                        mask = None
                        if self.eval_land_only and (lsm_hr is not None):
                            mask = (lsm_hr >= 0.5).float()  # [B, 1, H, W]
                        from sbgm.monitoring import _masked_corrcoef_per_sample
                        hr_lr_corr_val = _masked_corrcoef_per_sample(x, lr_ups_baseline.expand_as(x), mask=mask).item()
                        if verbose:
                            logger.info(f"[monitor][train] Step {idx}: HR-LR correlation: {hr_lr_corr_val:.4f}")
                except Exception as e:
                    logger.warning(f"[monitor][train] Could not compute HR-LR correlation at step {idx}. Error: {e}")
                    hr_lr_corr_val = float('nan')
                finally:
                    self.live_metrics['hr_lr_corr'].append(hr_lr_corr_val)
                    if self.writer is not None:
                        self.writer.add_scalar('monitoring/hr_lr_corr_train', hr_lr_corr_val, (current_epoch - 1) * len(dataloader) + idx)
                    

            # Backward pass
            batch_loss.backward()
            # Update weights
            self.optimizer.step()
            # Update EMA model if enabled
            if self.with_ema:
                self._update_ema()

            # Add batch loss to total loss
            loss_sum += batch_loss.item()
            # Update the bar
            if idx % self.cfg['training'].get('train_postfix_every', 10) == 0:
                pbar.set_postfix(loss=loss_sum / (idx + 1))
        
        # Calculate average loss
        avg_loss = loss_sum / len(dataloader)

        # Print average loss if verbose
        if verbose:
            logger.info(f"→ Epoch {getattr(self, 'epoch', '?')} completed: Avg. training Loss: {avg_loss:.4f}")

        return avg_loss
    
    def train(self,
              train_dataloader,
              val_dataloader,
              gen_dataloader,
              cfg,
              epochs=1,
              verbose=True,
              use_mixed_precision=False
              ):
        '''
            Method to run through the training batches in one epoch.
            Args:
                train_dataloader: Dataloader to run through.
                val_dataloader: Dataloader to run through for validation.
                epochs: Number of epochs to train for.
                verbose: Boolean to print progress.
                PLOT_FIRST: Boolean to plot the first image.
                SAVE_PATH: Path to save the image.
                SAVE_NAME: Name of the image to save.
                use_mixed_precision: Boolean to use mixed precision training.
        '''

        # === Classifier-Free Guidance (CFG) parameters ===
        logger.info(f"→ Classifier-Free Guidance (CFG) enabled: {self.cfg_guidance.get('enabled', False)}")
        if self.cfg_guidance.get('enabled', False):
            logger.info(f"   ▸ Dropout probability for LR conditions: {self.cfg_guidance.get('drop_prob', 0.1)}")
            logger.info(f"   ▸ Dropout probability for static geo: {self.cfg_guidance.get('drop_prob_geo', self.cfg_guidance.get('drop_prob', 0.1))}")

        # Log EMA
        logger.info(f"→ EMA enabled: {self.with_ema}; decay: {getattr(self, 'ema_decay', None)}; eval_use_ema: {cfg['training'].get('eval_use_ema', True)}")

        train_losses = []
        val_losses = []

        # set best loss to infinity
        train_loss = float('inf')
        val_loss = float('inf')
        best_loss = float('inf')

        # Iterate through epochs
        for epoch in range(1, epochs + 1):
            # Set epoch attribute
            self.epoch = epoch 
            # Print epoch number if verbose
            if verbose:
                logger.info(f"\n\n      ▸ Starting epoch {epoch}/{epochs}...")

            # Train on batches
            train_loss = self.train_batches(train_dataloader,
                                            epochs=epochs,
                                            current_epoch=epoch,
                                            verbose=verbose,
                                            use_mixed_precision=use_mixed_precision)

            # Append training loss to list
            train_losses.append(train_loss)

            val_loss = self.validate_batches(val_dataloader, verbose)
            # Append validation loss to list
            val_losses.append(val_loss)

            # Step the learning rate scheduler if provided
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)  # Step with validation loss
                else:
                    self.lr_scheduler.step()  # Regular step
                if verbose:
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    logger.info(f"→ Learning rate after epoch {epoch}: {current_lr:.6f}")

            # Capture improvement before updating best loss
            improved = val_loss < best_loss

            # If validation loss is lower than best loss, save the model
            if improved:
                best_loss = val_loss
                # Save the model
                self.save_model(dirname=self.checkpoint_dir, filename=self.checkpoint_name)
                logger.info(f"→ Best model saved with validation loss: {best_loss:.4f} at epoch {epoch}.")
                logger.info(f"→ Checkpoint saved to {os.path.join(self.checkpoint_dir, self.checkpoint_name)}")


            # Pickle dump the losses
            losses = {
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            with open(os.path.join(self.path_losses, 'losses' + f'_{self.model_string}.pkl'), 'wb') as f:
                pickle.dump(losses, f)
            
            if cfg['visualization']['create_figs'] and cfg['visualization']['plot_losses']:
                # Plot the losses
                self.plot_losses(train_losses,
                                 val_losses=val_losses,
                                 save_path=self.path_figures,
                                 save_name=f'losses_plot_{self.model_string}.png',
                                 show_plot=cfg['visualization']['show_figs'])
            # Plot in-loop timeseries occasionally
            if (self.monitor_plot_every_n_epochs > 0) and (epoch % self.monitor_plot_every_n_epochs == 0):
                try:
                    self._plot_live_metrics(self.path_metrics)
                except Exception as e:
                    logger.warning(f"[monitor] Could not plot live metrics at epoch {epoch}. Error: {e}")
            
            # Generate and save samples, if create_figs is True
            if cfg['visualization']['create_figs'] and cfg['data_handling']['n_gen_samples'] > 0:
                # Only generate and plot if loss improved or every n epochs if configured
                gen_every_n_epochs = int(cfg['visualization'].get('gen_and_plot_every_n_epochs', 1) or 1)

                if gen_every_n_epochs < 1:
                    gen_every_n_epochs = 1  # Ensure at least every 
                    
                on_schedule = (epoch % gen_every_n_epochs == 0)
                if improved or on_schedule:
                    if on_schedule:
                        logger.info(f"→ Generating and plotting samples at epoch {epoch} (every {gen_every_n_epochs} epochs)...")
                    if improved:
                        logger.info(f"→ Generating and plotting samples at epoch {epoch} (new best model)...")
                    self.generate_and_plot_samples(gen_dataloader,
                                                   cfg=cfg,
                                                   epoch=epoch)

            logger.info(f"→ Epoch {epoch}/{epochs} completed. \n\n")

        return train_loss, val_loss

    def validate_batches(self,
                    dataloader,
                    epochs=1,
                    current_epoch=1,
                    verbose=True
                 ):
        '''
            Method to run through the validation batches in one epoch.
            Args:
                dataloader: Dataloader to run through.
                verbose: Boolean to print progress.
        '''

        # Set model to evaluation mode
        self.model.eval()
        edm_on = bool(self.cfg.get('edm', {}).get('enabled', False))

        # Choose eval model (EMA if enabled and configured)
        use_ema_for_val = bool(self.cfg['training'].get('eval_use_ema', True))
        model_eval = self.ema_model if (self.with_ema and use_ema_for_val and hasattr(self, 'ema_model')) else self.model

        # Set initial loss to 0
        loss = 0.0
        # Set the progress bar
        pbar = tqdm.tqdm(dataloader, desc=f"Epoch {current_epoch}/{epochs}", unit="batch")

        # Iterate through batches in dataloader (tuple of images and seasons)
        for idx, samples in enumerate(pbar):
            # Samples is a dict with following available keys: 'img', 'classifier', 'img_cond', 'lsm', 'sdf', 'topo', 'points'
            # Extract samples
            x, seasons, cond_images, lsm_hr, lsm, sdf, topo, hr_points, lr_points = extract_samples(samples, self.device)



            # Setup lr_ups_baseline if needed
            lr_ups_baseline = None
            if edm_on and self.edm_predict_residual:
                lr_ups_baseline = self._build_lr_ups_baseline(cond_images)  # [B, 1, H, W]

            # No gradients needed for validation
            with torch.inference_mode(): #torch.no_grad(): # New in PyTorch 1.9, slightly faster than torch.no_grad()
                # Use mixed precision training if needed
                if hasattr(self, 'scaler') and self.scaler:
                    with autocast():
                        # Pass the score model and samples+conditions to the loss_fn
                        batch_loss = self.loss_fn(model_eval,
                                             x,
                                             y=seasons,
                                             cond_img=cond_images,
                                             lsm_cond=lsm,
                                             topo_cond=topo,
                                             sdf_cond=sdf,
                                             lr_ups=lr_ups_baseline
                                             )
                else:
                    # No mixed precision, just pass the score model and samples+conditions to the loss_fn
                    batch_loss = self.loss_fn(model_eval,
                                         x,
                                         y=seasons,
                                         cond_img=cond_images,
                                         lsm_cond=lsm,
                                         topo_cond=topo,
                                         sdf_cond=sdf,
                                         lr_ups=lr_ups_baseline
                                     )
                # === Cosine monitoring (validation; lightweight) ===
                monitor_cfg = self.cfg.get('monitoring', {})
                log_every = monitor_cfg.get('edm_metrics_every', 50)
                if edm_on and log_every > 0 and (idx % log_every == 0):
                    cos_val = edm_cosine_metric(self.loss_fn, model_eval, x, y=seasons, cond_img=cond_images, lsm_cond=lsm, topo_cond=topo, lr_ups=lr_ups_baseline)
                    if cos_val is not None:
                        if verbose:
                            logger.info(f"→ [monitor][val] Step {idx}: EDM cosine metric: {cos_val:.4f}")
                        if self.writer is not None:
                            self.writer.add_scalar('monitoring/edm_cosine_metric_val', cos_val, (current_epoch - 1) * len(dataloader) + idx)

            # Add batch loss to total loss
            loss += batch_loss.item()
            # Update the bar
            if idx % self.cfg['training'].get('train_postfix_every', 10) == 0:
                pbar.set_postfix(loss=loss / (idx + 1))

        # Calculate average loss
        avg_loss = loss / len(dataloader)

        # Print average loss if verbose
        if verbose:
            logger.info(f'→ Validation Loss: {avg_loss:.4f}')

        return avg_loss
    
    def generate_and_plot_samples(self,
                            gen_dataloader,
                            cfg,
                            epoch,
                          ):
        
        # Load the best model (EMA or network) from checkpoint WITHOUT altering training weights
        model_sd_backup = copy.deepcopy(self.model.state_dict())  # Backup current model state dict

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        net_sd = checkpoint.get('network_params', None) # Network state dict
        ema_sd = checkpoint.get('ema_network_params', None) # EMA state dict if exists
        use_ema_for_gen = bool(cfg['training'].get('eval_use_ema', True))

        if self.with_ema and use_ema_for_gen and (ema_sd is not None):
            self.model.load_state_dict(ema_sd)
            logger.info(f"→ Loaded EMA model weights into the main model from checkpoint {self.checkpoint_path} for sampling.")
        elif net_sd is not None:
            self.model.load_state_dict(net_sd)
            logger.info(f"→ Loaded model weights into the main model from checkpoint {self.checkpoint_path} for sampling.")
        else:
            logger.warning(f"→ No EMA weights in checkpoint; using network weights for sampling.")

        # Keep a synced EMA model handy if enabled
        if self.with_ema:
            if not hasattr(self, 'ema_model'):
                self._init_ema()  # Initialize EMA if not already
            if ema_sd is not None:
                self.ema_model.load_state_dict(ema_sd)  # Sync EMA model

        # Set model to evaluation mode (set back to training mode after sampling)
        self.model.eval()

        # Set up sampler 
        edm_on = bool((cfg.get('edm', {}).get('enabled', False)))
        if edm_on:
            sampler_edm = edm_sampler
            sampler = None
            logger.info("→ Sampling using EDM sampler...")
            
        else:
            sampler_edm = None
            if cfg['sampler']['sampler_type'] == 'pc_sampler':
                sampler = pc_sampler 
            elif cfg['sampler']['sampler_type'] == 'Euler_Maruyama_sampler':
                sampler = Euler_Maruyama_sampler
            elif cfg['sampler']['sampler_type'] == 'ode_sampler':
                sampler = ode_sampler
            else:
                raise ValueError(f"Sampler type {cfg['sampler']['sampler_type']} not recognized. Please choose from 'pc_sampler', 'Euler_Maruyama_sampler', or 'ode_sampler'.")
        
        
        full_domain_dims_str_hr = f"{self.full_domain_dims_hr[0]}x{self.full_domain_dims_hr[1]}" if self.full_domain_dims_hr is not None else "full_domain"
        full_domain_dims_str_lr = f"{self.full_domain_dims_lr[0]}x{self.full_domain_dims_lr[1]}" if self.full_domain_dims_lr is not None else "full_domain"
        crop_region_hr_str = '_'.join(map(str, self.crop_region_hr)) if self.crop_region_hr is not None else "no_crop"
        crop_region_lr_str = '_'.join(map(str, self.crop_region_lr)) if self.crop_region_lr is not None else "no_crop"
        

        back_transforms = build_back_transforms_from_stats(
                            hr_var              = cfg['highres']['variable'],
                            hr_model            = cfg['highres']['model'],
                            domain_str_hr       = full_domain_dims_str_hr,
                            crop_region_str_hr  = crop_region_hr_str,
                            hr_scaling_method   = cfg['highres']['scaling_method'],
                            hr_buffer_frac      = cfg['highres']['buffer_frac'] if 'buffer_frac' in cfg['highres'] else 0.0,
                            lr_vars             = cfg['lowres']['condition_variables'],
                            lr_model            = cfg['lowres']['model'],
                            domain_str_lr       = full_domain_dims_str_lr,
                            crop_region_str_lr  = crop_region_lr_str,
                            lr_scaling_methods  = cfg['lowres']['scaling_methods'],
                            lr_buffer_frac      = cfg['lowres']['buffer_frac'] if 'buffer_frac' in cfg['lowres'] else 0.0,
                            split               = 'train',
                            stats_dir_root      = cfg['paths']['stats_load_dir'],
                            eps=self.global_prcp_eps
                            )

        # Setup units and cmaps
        hr_unit, lr_units = get_units(cfg)
        hr_cmap_name, lr_cmap_dict = get_cmaps(cfg)

        p_bar = tqdm.tqdm(gen_dataloader, desc=f"Generating samples for epoch {epoch}", unit="batch") # type: ignore
        # Iterate through batches in dataloader
        for idx, samples in enumerate(p_bar):
            # Samples is a dict with following available keys: 'img', 'classifier', 'img_cond', 'lsm', 'sdf', 'topo', 'points'
            # Extract samples
            x_gen, seasons_gen, cond_images_gen, lsm_hr_gen, lsm_gen, sdf_gen, topo_gen, hr_points_gen, lr_points_gen = extract_samples(samples, self.device)
            logger.info(f"→ Generating {len(x_gen)} samples at epoch {epoch}, batch {idx}...")
            logger.info(f"      Only plotting first {min(cfg['visualization'].get('n_plot_samples', 4), cfg['data_handling']['n_gen_samples'])} samples.")

            # Setup lr_ups_baseline if needed
            lr_ups_baseline = None
            if edm_on and self.edm_predict_residual:
                lr_ups_baseline = self._build_lr_ups_baseline(cond_images_gen)  # [B, 1, H, W]

            if edm_on and sampler_edm is not None:
                edm_cfg = cfg.get('edm', {}) or {}
                guidance_cfg = cfg.get('classifier_free_guidance', {})

                generated_samples = sampler_edm(score_model=self.model,
                                            batch_size=cfg['data_handling']['n_gen_samples'],
                                            num_steps=edm_cfg.get('sampling_steps', 18),
                                            device=self.device,
                                            img_size=cfg['highres']['data_size'][0],
                                            y=seasons_gen,
                                            cond_img=cond_images_gen,
                                            lsm_cond=lsm_gen,
                                            topo_cond=topo_gen,
                                            sigma_min=float(edm_cfg.get('sigma_min', 0.002)),
                                            sigma_max=float(edm_cfg.get('sigma_max', 80)),
                                            rho=float(edm_cfg.get('rho', 7.0)),
                                            S_churn=float(edm_cfg.get('S_churn', 0.0)),
                                            S_min=float(edm_cfg.get('S_min', 0.0)),
                                            S_max=float(edm_cfg.get('S_max', float('inf'))),
                                            S_noise=float(edm_cfg.get('S_noise', 1.0)),
                                            lr_ups=lr_ups_baseline,
                                            cfg_guidance=guidance_cfg if guidance_cfg.get('enabled', False) else None,
                )
            elif sampler is not None:
                generated_samples = sampler(
                    score_model=self.model,
                    marginal_prob_std=self.marginal_prob_std_fn,
                    diffusion_coeff=self.diffusion_coeff_fn,
                    batch_size=cfg['data_handling']['n_gen_samples'],
                    num_steps=cfg['sampler']['n_timesteps'],
                    device=self.device,
                    img_size=cfg['highres']['data_size'][0],
                    y=seasons_gen,
                    cond_img=cond_images_gen,
                    lsm_cond=lsm_gen,
                    topo_cond=topo_gen,
                )
            else:
                raise ValueError("No valid sampler found. Please check the configuration.")

            # Keep sampler output in model space on CPU (preserve batch dim!)
            gen_model = generated_samples.detach().cpu().float()

            # === Back-transform for metrics (NOT plotting) ===
            # 1) Get HR and generated in physical space if possible (else use model space)
            if back_transforms is not None:
                # Expect a callable for HR back-transform under key self.bt_gen_key
                bt_gen = back_transforms.get(self.bt_gen_key, None)
                bt_hr = back_transforms.get(self.bt_hr_key, None)
                if callable(bt_gen):
                    logger.info("[monitor] Applying HR back-transform to generated samples.")
                    gen_phys = bt_gen(gen_model)
                else:
                    logger.warning("[monitor] HR back-transform not callable; using model space for generated samples.")
                    gen_phys = gen_model

                if callable(bt_hr):
                    logger.info("[monitor] Applying HR back-transform to ground-truth HR samples.")
                    hr_phys = bt_hr(x_gen)
                else:
                    logger.warning("[monitor] HR back-transform not callable; using model space for ground-truth HR samples.")
                    hr_phys = x_gen
            else:
                logger.info("[monitor] No back-transforms available; using model space for generated samples.")
                gen_phys = gen_model
                hr_phys = x_gen

            # === Diagnostics check: physical units exceedance ===
            if gen_phys is not None and hr_phys is not None:
                if not isinstance(gen_phys, torch.Tensor):
                    gen_phys = torch.tensor(gen_phys)
                tensor_stats(gen_phys, f"eval/x_phys")
                warn_hi = float(cfg.get('diagnostics', {}).get('warn_if_phys_gt', 300.0))
                if float(gen_phys.max().item()) > warn_hi:
                    logger.warning(f"[diagnostics][eval] Generated samples exceed {warn_hi} {hr_unit} in physical space. Max: {float(gen_phys.max().item()):.2f} {hr_unit}. Consider adjusting back-transform, data scaling, or adding clamping.")

            # TODO: Clamp gen_model in model space for injection to plotting? Maybe add clamper in sampling instead?
            try:
                # Extreme sentinel (and optional clamp) in PHYSICAL space
                mon_cfg = cfg.get('monitoring', {}).get('extreme_prcp', {})
                thr = float(mon_cfg.get('threshold_mm', self.extreme_threshold_mm))

                # Ensure gen_phys is a torch.Tensor before passing to report_precip_extremes
                if not isinstance(gen_phys, torch.Tensor):
                    gen_phys = torch.tensor(gen_phys)
                chk = report_precip_extremes(x_bt=gen_phys, name="generated_hr", cap_mm_day=thr)
                if chk.get('has_extreme', False):
                    extreme_values = chk.get('extreme_values', [])
                    mx = max(extreme_values) if isinstance(extreme_values, list) and extreme_values else None
                    cnt = len(extreme_values) if isinstance(extreme_values, list) else None
                    logger.warning(f"[monitor][gen] Extreme precip: max={mx:.1f} mm/day, count={cnt}, thr={thr} mm/day")

                    if mon_cfg.get('clamp_in_generation', self.extreme_clamp_in_gen):
                        clamp_max = float(mon_cfg.get('clamp_max_mm', thr))
                        gen_phys = torch.clamp(gen_phys, min=0.0, max=clamp_max)
                        logger.warning(f"[monitor][gen] Clamped generated samples to ≤ {clamp_max} mm/day.")
                        logger.warning(f"[monitor][gen] Note: clamping is not done on plotted samples, only on gen_phys used for metrics. Consider adding clamping in sampling instead if desired.")
            except Exception as e:
                logger.warning(f"[monitor] Could not run extreme sentinel. Error: {e}")

            # === Epoch-level metrics (always use PHYSICAL for fairness) ===
            try:
                # Make sure everything is on CPU and detached
                if not isinstance(gen_phys, torch.Tensor):
                    gen_phys = torch.tensor(gen_phys)
                gen_phys = gen_phys.detach().cpu()
                if not isinstance(hr_phys, torch.Tensor):
                    hr_phys = torch.tensor(hr_phys)
                hr_phys = hr_phys.detach().cpu()
                if lsm_hr_gen is not None:
                    if not isinstance(lsm_hr_gen, torch.Tensor):
                        lsm_hr_gen = torch.tensor(lsm_hr_gen)
                    lsm_hr_gen = lsm_hr_gen.detach().cpu()

                # Optional land-only mask at HR resolution
                mask = None
                if self.eval_land_only and (lsm_hr_gen is not None):
                    mask = (lsm_hr_gen >= 0.5).to(dtype=torch.float32).detach().cpu()

                # Ensure gen_phys and hr_phys are torch.Tensor
                if not isinstance(gen_phys, torch.Tensor):
                    gen_phys = torch.tensor(gen_phys)
                if not isinstance(hr_phys, torch.Tensor):
                    hr_phys = torch.tensor(hr_phys)

                # 1) FSS @ scales
                fss_dict = compute_fss_at_scales(
                    gen_phys, hr_phys, mask=mask,
                    fss_km=self.fss_scales_km,
                    grid_km_per_px=self.pixel_km,
                    thr_mm=self.fss_threshold_mm
                )
                self.fss_hist.append(fss_dict)
                # Plot only history, not per-epoch
                plot_fss_history(self.fss_hist, save_dir=self.path_metrics,
                                filename="fss_history.png",
                                title="FSS history" + (" (land-only)" if self.eval_land_only else ""))
                # 2) PSD slope
                psd_dict = compute_psd_slope(gen_phys, hr_bt=hr_phys if self.psd_compare_to_hr else None, mask=mask)
                self.psd_hist.append(psd_dict)
                # Plot only history, not per-epoch
                plot_psd_slope_history(self.psd_hist, save_dir=self.path_metrics,
                                    filename="psd_history.png",
                                    title="PSD slope history" + (" (land-only)" if self.eval_land_only else ""))

                # 3) Q95/Q99 + wet-day
                q_dict = compute_q95_q99_and_wet_day(gen_phys,
                                                    hr_bt=hr_phys if self.quantiles_compare_to_hr else None,
                                                    mask=mask,
                                                    wet_threshold_mm=self.wetday_thresh)
                self.q_hist.append(q_dict)
                # Plot only history, not per-epoch
                plot_quantiles_wetday_history(self.q_hist, save_dir=self.path_metrics,
                                            filename="quantiles_history.png",
                                            title="Quantiles & wet-day history" + (" (land-only)" if self.eval_land_only else ""))
            except Exception as e:
                logger.warning(f"[monitor] Could not compute epoch-level metrics at epoch {epoch}. Error: {e}")

            # === Plot samples ===
            if cfg['visualization']['create_figs']:
                # Use gen_model and samples (both in model space) for plotting and transform in plotting function if needed
                fig, _ = plot_samples_and_generated(
                    samples=samples,
                    generated=gen_model, 
                    cfg=cfg,
                    transform_back_bf_plot=cfg['visualization']['transform_back_bf_plot'],
                    back_transforms=back_transforms
                )
                if cfg['visualization']['save_figs']:
                    fig.savefig(os.path.join(self.path_figures, f'epoch_{epoch}_generatedSamples.png'),
                                dpi=300, bbox_inches='tight')
                    logger.info(f"→ Figure saved to {os.path.join(self.path_figures, f'epoch_{epoch}_generatedSamples.png')}")
                plt.close(fig)
                break  # one batch per epoch                    
        # Restore training weights and mode
        self.model.load_state_dict(model_sd_backup)  # Restore original model weights
        self.model.train()  # Set back to training mode

    def plot_losses(self,
                    train_losses,
                    val_losses=None,
                    save_path=None,
                    save_name='losses_plot.png',
                    show_plot=False,
                    verbose=True):
        '''
            Plot the training and validation losses.
            Args:
                train_losses: List of training losses.
                val_losses: List of validation losses.
                save_path: Path to save the plot.
                save_name: Name of the plot file.
                show_plot: Boolean to show the plot.
        '''
        # Plot the losses
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(train_losses, label='Training Loss', color='blue')
        if val_losses is not None:
            ax.plot(val_losses, label='Validation Loss', color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Losses')
        ax.legend()

        # Show the plot
        if show_plot:
            plt.show()
            
        # Save the plot
        if save_path is not None:
            fig.savefig(os.path.join(save_path, save_name), dpi=300, bbox_inches='tight')
            if verbose:
                logger.info(f"→ Losses plot saved to {os.path.join(save_path, save_name)}")

        plt.close(fig)

    def _plot_live_metrics(self, save_dir: str):
        """
        Internal method to plot live training metrics if enabled in the configuration.
        Args:
            save_dir (str): Directory where the metrics plot will be saved.
        """
        if len(self.live_metrics['steps']) == 0:
            return

        out = os.path.join(self.path_metrics, 'inLoop_metrics_timeseries.png')

        try:
            plot_live_training_metrics(
                self.live_metrics['steps'],
                self.live_metrics['edm_cosine'],
                self.live_metrics['hr_lr_corr'],
                save_dir=self.path_metrics,
                filename='inLoop_metrics_timeseries.png',
                show=self.cfg['visualization'].get('show_figs', False),
                title="In-loop training metrics (EDM cosine, HR-LR corr)"
            )
            logger.info(f"→ Live metrics plot saved to {out}")

        except Exception as e:
            logger.error(f"[Monitor] Could not save live metrics plot to {out}. Error: {e}")

        
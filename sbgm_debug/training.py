"""
    TODO:
        - Implement mixed precision training 
        - Make precipitation evaluations only when precipitation is the target variable
    ToDo:
        - Add support for mixed precision training
        - Add support for EMA (Exponential Moving Average) of the model
        - Add support for custom weight initialization

"""

import os
import torch
import copy
import pickle
import tqdm
import logging 
import math

import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt


from typing import Optional
from torch.cuda.amp import autocast, GradScaler

from sbgm_debug.cfg_params import (
    PathsParams,
    DataParams,
    TrainParams,
    EDMParams,
    RainGateParams,
    GuidanceParams,
    DiagnosticsParams,
    EndOfEpochParams,
    VisualizationsParams,
    ExtremePrcpParams,
    MonitoringParams,
    SamplerParams
)

from sbgm_debug.heads.rain_gate import RainGate
from sbgm_debug.special_transforms import build_back_transforms_from_stats, lr_baseline_to_hr_zspace
from sbgm_debug.utils import get_model_string, extract_samples
from sbgm_debug.plotting_utils import (
    get_cmaps,
    plot_samples_and_generated,
    plot_live_training_metrics,
    plot_fss_history,
    plot_psd_slope_history,
    plot_quantiles_wetday_history,
    )
from sbgm_debug.monitoring import (
    report_precip_extremes,
    compute_fss_at_scales,
    compute_psd_slope,
    compute_p95_p99_and_wet_day,
    tensor_stats,
    in_loop_metrics,
    _save_weight_map_viz,
    _plot_reliability_curve
    )
from sbgm_debug.score_sampling import Euler_Maruyama_sampler, pc_sampler, ode_sampler, edm_sampler
from sbgm_debug.training_utils import get_loss_fn, apply_cfg_dropout
from sbgm_debug.variable_utils import get_units

# Speed up conv algo selection on fixed input sizes
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Set up logging
logger = logging.getLogger(__name__)


def _safe_bool(d, k, default=False):
    return bool(d.get(k, default))
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
        self.cfg = cfg
        self.model = model
        self.marginal_prob_std_fn = marginal_prob_std_fn
        self.diffusion_coeff_fn = diffusion_coeff_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # === DATA params ===
        high = cfg['highres']
        low = cfg['lowres']
        trans = cfg.get('transforms', {})
        paths = cfg['paths']
        train = cfg['training']
        ema_cfg = train.get('ema', {})

        self.data = DataParams(
            hr_var              = high['variable'],
            hr_scaling_method   = high['scaling_method'],
            hr_model            = high['model'],
            full_domain_dims_hr = high.get('full_domain_dims', None),
            crop_region_hr      = high.get('cutout_domains', None),
            lr_vars             = low['condition_variables'],
            lr_scaling_methods  = low['scaling_methods'],
            lr_model            = low['model'],
            full_domain_dims_lr = low.get('full_domain_dims', None),
            crop_region_lr      = low.get('cutout_domains', None),
            buffer_frac_hr      = float(high.get('buffer_frac', 0.0)),
            buffer_frac_lr      = float(low.get('buffer_frac', 0.0)),
            scaling_split       = str(trans.get('scaling_split', 'train')),
            prcp_eps            = float(trans.get('prcp_eps', 0.01)),
        )

        self.trainp = TrainParams(
            epochs              = int(train.get('epochs', 1)),
            mixed_precision     = bool(train.get('mixed_precision', train.get('use_mixed_precision', False))),
            eval_use_ema        = bool(ema_cfg.get('eval_use_ema', True)),
            train_postfix_every = int(train.get('train_postfix_every', 10)),
            verbose             = bool(train.get('verbose', True)),
        )

        edm_cfg = cfg.get('edm', {})
        self.edm = EDMParams(
            enabled          = _safe_bool(edm_cfg, 'enabled', False),
            predict_residual = _safe_bool(edm_cfg, 'predict_residual', False),
            baseline_space   = str(edm_cfg.get('baseline_space', 'hr')),
        )

        rg = cfg.get('rain_gate', {})
        self.rg = RainGateParams(
            enabled                 = _safe_bool(rg, 'enabled', False),
            include_lsm             = _safe_bool(rg, 'include_lsm', True),
            include_topo            = _safe_bool(rg, 'include_topo', True),
            include_lr_baseline     = _safe_bool(rg, 'include_lr_baseline', True),
            wet_threshold_mm        = float(rg.get('wet_threshold_mm', cfg.get('monitoring', {}).get('end_of_epoch', {}).get('wet_day_threshold_mm', 0.1))),
            wet_threshold_model_space = float(rg.get('wet_threshold_modelSpace', 0.1)),
            reweight_enabled        = _safe_bool(rg, 'reweight_enabled', False),
            warm_start_epochs       = int(rg.get('reweight_warm_start_epochs', 5)),
            ramp_epochs             = int(rg.get('reweight_ramp_epochs', 0)),
            loss_weight_bce         = float(rg.get('loss_weight_bce', 0.1)),
            pos_weight              = float(rg.get('pos_weight', 2.0)),
            learning_rate           = float(rg.get('learning_rate', self.optimizer.param_groups[0]['lr'] if self.optimizer is not None else 1e-4)),
            c_hidden                = int(rg.get('c_hidden', 16)),
            weight_strategy         = str(rg.get('weight_strategy', 'prob')).lower(),
            weight_alpha            = float(rg.get('weight_alpha', 2.0)),
            prob_gamma              = float(rg.get('prob_gamma', 1.0)),
            clip_max                = float(rg.get('clip_max', 5.0)),
            detach_weights          = _safe_bool(rg, 'detach_weights', True),
            binary_threshold        = float(rg.get('binary_threshold', 0.5)),
        )

        # --- Derived strings for stats lookup ---
        self._dom_hr_str = f"{self.data.full_domain_dims_hr[0]}x{self.data.full_domain_dims_hr[1]}" if self.data.full_domain_dims_hr is not None else "full_domain"
        self._dom_lr_str = f"{self.data.full_domain_dims_lr[0]}x{self.data.full_domain_dims_lr[1]}" if self.data.full_domain_dims_lr is not None else "full_domain"
        self._crop_hr_str = '_'.join(map(str, self.data.crop_region_hr)) if self.data.crop_region_hr is not None else "no_crop"
        self._crop_lr_str = '_'.join(map(str, self.data.crop_region_lr)) if self.data.crop_region_lr is not None else "no_crop"

        # --- Paths & names ---
        model_str = get_model_string(cfg)
        samples_dir = os.path.join(paths['path_save'], 'samples', model_str)
        figures_dir = os.path.join(samples_dir, 'Figures')
        metrics_dir = os.path.join(figures_dir, 'metrics')
        diagnostics_dir = os.path.join(metrics_dir, 'debug')
        for d in [paths['checkpoint_dir'], samples_dir, figures_dir, metrics_dir, diagnostics_dir, os.path.join(paths['path_save'], 'losses')]:
            os.makedirs(d, exist_ok=True)

        self.paths = PathsParams(
            checkpoint_dir = paths['checkpoint_dir'],
            stats_dir      = paths['stats_load_dir'],
            path_save_root = paths['path_save'],
            samples_dir    = samples_dir,
            figures_dir    = figures_dir,
            metrics_dir    = metrics_dir,
            diagnostics_dir= diagnostics_dir,
        )

        self.model_string = model_str
        self.checkpoint_name = self.model_string + '.pth.tar'
        self.checkpoint_path = os.path.join(self.paths.checkpoint_dir, self.checkpoint_name)


        moncfg = cfg.get("monitoring", {})
        self.mon = MonitoringParams.from_cfg(moncfg).validate(
            allowed_step=["loss", "hr_lr_corr", "edm_cosine"],
            allowed_epoch=["loss", "initial", "samples", "fss", "psd_slope", "quantiles", "weight_map"],
        )
        # Sanity: edm_metrics_every should not exceed dataloader length by orders of magnitude
        if self.edm_metrics_every < 0:
            self.edm_metrics_every = 0

        # Sanity: FSS band bounds
        lo, hi = self.mon.end_of_epoch.psd_band
        if not (0.0 <= lo < hi <= 0.5):
            import warnings
            warnings.warn(f"[monitor] psd_band {self.mon.end_of_epoch.psd_band} outside (0, 0.5]; resetting to [0.05, 0.40].")
            self.mon = MonitoringParams.from_cfg({**moncfg, "end_of_epoch": {**moncfg.get("end_of_epoch", {}), "psd_band": [0.05, 0.40]}})

        # Cached, hot-path fields
        self.eval_land_only         = self.mon.land_only
        self.edm_metrics_every      = self.mon.edm_metrics_every
        self.monitor_plot_every_n_epochs = self.mon.visualizations.plot_every_n_epochs

        # End-of-epoch metrics config (used in generation/eval)
        self.fss_scales_km          = self.mon.end_of_epoch.fss_km
        self.fss_threshold_mm       = self.mon.end_of_epoch.fss_threshold_mm
        self.pixel_km               = self.mon.end_of_epoch.grid_km_per_px
        self.psd_compare_to_hr      = self.mon.end_of_epoch.psd_compare_to_hr
        self.quantiles_compare_to_hr= self.mon.end_of_epoch.quantiles_compare_to_hr
        self.wetday_thresh          = self.mon.end_of_epoch.wet_day_threshold

        # Diagnostics (used in loops)
        self.diag_per_batch         = self.mon.diagnostics.per_batch_stats
        self.diag_log_every         = self.mon.diagnostics.log_every
        self.diag_viz_every_epochs  = self.mon.diagnostics.viz_every_n_epochs
        self.diag_warn_abs          = self.mon.diagnostics.warn_if_abs_gt
        self.diag_warn_phys         = self.mon.diagnostics.warn_if_phys_gt

        # Extreme precipitation sentinel for generation
        self.extreme_enabled        = self.mon.extreme_prcp.enabled
        self.extreme_threshold_mm   = self.mon.extreme_prcp.threshold_mm
        self.extreme_backtransform  = self.mon.extreme_prcp.back_transform
        self.extreme_in_validation  = self.mon.extreme_prcp.check_in_validation
        self.extreme_clamp_in_gen   = self.mon.extreme_prcp.clamp_in_generation
        self.extreme_clamp_max      = self.mon.extreme_prcp.clamp_max_mm


        # Training-time convenience
        self.bt_gen_key = "generated"
        self.bt_hr_key = f"{self.data.hr_var}_hr"

        # Loss function (built once)
        self.loss_fn = get_loss_fn(self.cfg, marginal_prob_std_fn_in=getattr(self, 'marginal_prob_std_fn', None))

        # EMA
        self.with_ema = bool(train.get('with_ema', False))
        self.ema_decay = float(train.get('ema_decay', 0.9999)) # Default to 0.9999 if not specified
        if self.with_ema:
            self._init_ema()

        # Classifier free guidance 
        self.guidance = GuidanceParams.from_cfg(self.cfg.get('classifier_free_guidance', {}))

        # Handy aliases (avoid giant edits)
        self.hr_var = self.data.hr_var
        self.hr_scaling_method = self.data.hr_scaling_method
        self.full_domain_dims_hr = self.data.full_domain_dims_hr
        self.crop_region_hr = self.data.crop_region_hr
        self.lr_vars = self.data.lr_vars
        self.lr_scaling_methods = self.data.lr_scaling_methods
        self.full_domain_dims_lr = self.data.full_domain_dims_lr
        self.crop_region_lr = self.data.crop_region_lr
        self.global_prcp_eps = self.data.prcp_eps
        
        # Some strings for unified use
        self.full_domain_dims_str_hr = f"{self.full_domain_dims_hr[0]}x{self.full_domain_dims_hr[1]}" if self.full_domain_dims_hr is not None else "full_domain"
        self.full_domain_dims_str_lr = f"{self.full_domain_dims_lr[0]}x{self.full_domain_dims_lr[1]}" if self.full_domain_dims_lr is not None else "full_domain"
        self.crop_region_hr_str = '_'.join(map(str, self.crop_region_hr)) if self.crop_region_hr is not None else "no_crop"
        self.crop_region_lr_str = '_'.join(map(str, self.crop_region_lr)) if self.crop_region_lr is not None else "no_crop"

        # Setup units and cmaps
        self.hr_unit, self.lr_units = get_units(cfg)
        self.hr_cmap_name, self.lr_cmap_dict = get_cmaps(cfg)

        if self.hr_var in self.lr_vars:
            idx_t = self.lr_vars.index(self.hr_var)
            self._lr_method_for_target = self.lr_scaling_methods[idx_t]
        else:
            self._lr_method_for_target = None  # Target variable not in LR vars
            logger.warning(f"HR target variable '{self.hr_var}' not found in LR condition variables {self.lr_vars}. Cannot determine LR scaling method for target - residuals may not be aligned.")        

        try:
            self.back_transforms_train = build_back_transforms_from_stats(
                hr_var=self.data.hr_var,
                hr_model=self.data.hr_model,
                domain_str_hr=self._dom_hr_str,
                crop_region_str_hr=self._crop_hr_str,
                hr_scaling_method=self.data.hr_scaling_method,
                hr_buffer_frac=self.data.buffer_frac_hr,
                lr_vars=self.data.lr_vars,
                lr_model=self.data.lr_model,
                lr_scaling_methods=self.data.lr_scaling_methods,
                domain_str_lr=self._dom_lr_str,
                crop_region_str_lr=self._crop_lr_str,
                lr_buffer_frac=self.data.buffer_frac_lr,
                split=self.data.scaling_split,
                stats_dir_root=self.paths.stats_dir,
                eps=self.global_prcp_eps
            )
        except Exception as e:
            logger.warning(f"[monitor] Could not build back transforms for sentinel; will skip back_transform in training. Error: {e}")
            self.back_transforms_train = None

        self.rain_gate: RainGate | None = None
        c_in = None  # Ensure c_in is always defined
        if self.rg.enabled:
            c_in = len(self.data.lr_vars)
            if bool(low.get('dual_lr', False)) and (self.data.hr_var in self.data.lr_vars):
                c_in += 1
            if self.rg.include_lsm: c_in += 1
            if self.rg.include_topo: c_in += 1
            if self.rg.include_lr_baseline: c_in += 1

            self.rain_gate = RainGate(c_in=c_in, c_hidden=self.rg.c_hidden).to(self.device)
            if self.optimizer is not None:
                self.optimizer.add_param_group({'params': self.rain_gate.parameters(), 'lr': self.rg.learning_rate})
                import torch.optim.lr_scheduler as _schedulers
                if isinstance(self.lr_scheduler, _schedulers.ReduceLROnPlateau):
                    n_groups = len(self.optimizer.param_groups)
                    if len(self.lr_scheduler.min_lrs) < n_groups:
                        tail = self.lr_scheduler.min_lrs[-1] if len(self.lr_scheduler.min_lrs) > 0 else 0.0
                        self.lr_scheduler.min_lrs += [tail] * (n_groups - len(self.lr_scheduler.min_lrs))
        logger.info(f"→ Rain gating head enabled: {self.rg.enabled}, c_in: {c_in if self.rg.enabled else 'N/A'}")

        # Sampler logic
        self.sampler = SamplerParams.from_cfg(cfg)

        # === Live, lightweight monitors (append in loop; plot occasionally) ===
        self.monitor_plot_every_n_epochs = int(self.mon.visualizations.plot_every_n_epochs)
        self.live_metrics = {
            'steps': [],
            'edm_cosine': [],
            'hr_lr_corr': []
        }

        # Persistent histories of epoch-level monitors
        self.fss_hist: list[dict] = []
        self.psd_hist: list[dict] = []
        self.q_hist: list[dict] = []
        self.epoch_list = []

    def _build_lr_ups_baseline(self, cond_images: torch.Tensor | None):
        """
            Extract LR baseline channel (same variable as HR target) from cond_images and upsample to HR resolution.
            Ensure it is expressed in HR z-space (or HR min-max space) before using for residual EDM.
            Returns [B, 1, H, W] or raises if unavailable when predict_residual is True.
        """
        if cond_images is None:
            raise ValueError("cond_images is None, cannot extract LR baseline for residual prediction.")
        
        cond_vars = self.lr_vars
        target_var = self.hr_var
        if target_var not in cond_vars:
            raise ValueError(f"Target variable '{target_var}' not found in condition variables {cond_vars}, cannot extract LR baseline for residual prediction.")
        
        idx = cond_vars.index(target_var)
        if cond_images.shape[1] <= idx:
            raise ValueError(f"cond_images has shape {cond_images.shape}, cannot extract channel index {idx} for variable '{target_var}'.")
        lr_in_lr_space = cond_images[:, idx:idx+1, :, :]  # [B, 1, h, w] - cond images already upsampled to HR size

        if self.edm.baseline_space == 'lr':
            logger.info(f"baseline_space requested is 'lr'; using LR baseline channel as-is in LR space for residual prediction.")
            return lr_in_lr_space  # Already in LR space, just upsampled to HR size
        
        # Else, need to convert from LR space to HR space 
        
        # Find the LR scaling method corresponding to baseline channel
        lr_method_for_baseline = self._lr_method_for_target 

        # Ensure lr_method_for_baseline is a string
        if lr_method_for_baseline is None:
            raise ValueError("LR scaling method for baseline is None. Cannot proceed with lr_baseline_to_hr_zspace. Please check your configuration.")

        # logger.info(f"Converting LR baseline channel from LR space to HR space using lr_baseline_to_hr_zspace with LR method '{lr_method_for_baseline}' and HR method '{self.hr_scaling_method}'.")
        # Remap using transform/back-transform stack
        lr_in_hr_space = lr_baseline_to_hr_zspace(
            lr_chan_norm=lr_in_lr_space,
            # LR meta
            lr_variable=self.hr_var,
            lr_model=self.data.lr_model,
            lr_domain_str=self._dom_lr_str,
            lr_crop_region_str=self._crop_lr_str,
            lr_split=self.data.scaling_split,
            lr_scaling_method=lr_method_for_baseline,
            lr_buffer_frac=self.data.buffer_frac_lr,
            lr_stats_dir_root=self.paths.stats_dir,
            # HR meta
            hr_variable=self.hr_var,
            hr_model=self.data.hr_model,
            hr_domain_str=self._dom_hr_str,
            hr_crop_region_str=self._crop_hr_str,
            hr_split=self.data.scaling_split,
            hr_scaling_method=self.hr_scaling_method,
            hr_buffer_frac=self.data.buffer_frac_hr,
            hr_stats_dir_root=self.paths.stats_dir,

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
        
        # Load rain-gate parameters if present in checkpoint
        try:
            rg_sd = checkpoint.get('rain_gate_params', None)
            if (rg_sd is not None) and self.rg.enabled and hasattr(self, 'rain_gate') and (self.rain_gate is not None): 
                self.rain_gate.load_state_dict(rg_sd)
                logger.info(f"→ Loaded rain-gate head weights from checkpoint {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Could not load rain-gate head weights from checkpoint {checkpoint_path}. Error: {e}")
        


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
        # Include rain-gate head params if enabled
        if self.rg.enabled and (self.rain_gate is not None):
            state_dicts['rain_gate_params'] = self.rain_gate.state_dict()

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
            if self.edm.enabled and self.edm.predict_residual:
                lr_ups_baseline = self._build_lr_ups_baseline(cond_images)  # [B, 1, H, W]

            # === Rain-gate auxiliary supervision (before classifier free guidance dropout affects inputs) ===
            rg_aux_loss = None
            wet_logits = None  # Ensure wet_logits is always defined
            if self.rg.enabled and (self.rain_gate is not None):
                # Build gate inputs by concatenation at HR resolution
                gate_inputs = []
                if cond_images is not None:
                    gate_inputs.append(cond_images)  # LR condition channels already upsampled to HR in dataset
                if self.rg.include_lsm and (lsm is not None):
                    gate_inputs.append(lsm)
                if self.rg.include_topo and (topo is not None):
                    gate_inputs.append(topo)
                if self.rg.include_lr_baseline and (lr_ups_baseline is not None):
                    gate_inputs.append(lr_ups_baseline)
                logger.info(f"[rain_gate debug] cond_images={cond_images is not None}, lsm={lsm is not None}, topo={topo is not None}, lr_ups_baseline={lr_ups_baseline is not None}")
                if len(gate_inputs) > 0:
                    gate_x = torch.cat(gate_inputs, dim=1)  # [B, C_in, H, W]
                    # Predict rain probabilities (logits) from gate inputs
                    wet_logits = self.rain_gate(gate_x)  # [B, 1, H, W]

                    # Target wet mask from HR **physical** space if possible, else model space fallback
                    with torch.no_grad():
                        wet_target = None
                        thr = float(self.rg.wet_threshold_mm)
                        bt_hr = None
                        try:
                            if self.back_transforms_train is not None:
                                bt_hr = self.back_transforms_train.get(self.bt_hr_key, None)
                        except Exception:
                            bt_hr = None
                        if callable(bt_hr):
                            x_phys = bt_hr(x) # Backtransform to physical space [B, 1, H, W] in mm/day
                            if not isinstance(x_phys, torch.Tensor):
                                x_phys = torch.tensor(x_phys, dtype=torch.float32)
                            wet_target = (x_phys > thr).to(dtype=torch.float32)  # [B, 1, H, W] binary mask
                        else:
                            # Fallback: use model space with global eps
                            logger.warning(f"[rain_gate] back_transforms_train missing or invalid; using model-space thresholding for rain gate target.")
                            wet_target = (x > self.rg.wet_threshold_model_space).to(dtype=torch.float32)  # [B, 1, H, W] binary mask
                        # Match shapes
                        if wet_target.shape[1] != 1:
                            wet_target = wet_target[:, :1, :, :]  # Ensure single channel

                    # Class imbalance handling via pos_weight
                    pos_w = torch.tensor(self.rg.pos_weight, device=x.device, dtype=torch.float32)
                    bce = F.binary_cross_entropy_with_logits(wet_logits, wet_target, pos_weight=pos_w)
                    rg_aux_loss = bce * self.rg.loss_weight_bce  # Scale BCE loss
                else:
                    rg_aux_loss = None  # No inputs for rain gate
            # === Optional: Use gate to reweight main loss on wet pixels (with warm start and ramp) ===
            pixel_weight_map = None
            if self.rg.enabled and (self.rain_gate is not None):
                
                # Decide whether to apply reweighting based on epoch
                do_reweight = self.rg.reweight_enabled and (current_epoch > self.rg.warm_start_epochs)
                if do_reweight and ('wet_logits' in locals()) and (wet_logits is not None):
                    p = torch.sigmoid(wet_logits)  # [B, 1, H, W] probabilities

                    # Base weighting shape from config
                    strategy = str(self.rg.weight_strategy)  # 'prob' or 'binary'
                    alpha = float(self.rg.weight_alpha)  # Weighting strength
                    clip_max = float(self.rg.clip_max)  # Max clip for weights
                    detach_w = bool(self.rg.detach_weights)  # Detach weights from gradient flow

                    if strategy == 'binary':
                        thr_p = float(self.rg.binary_threshold)
                        w_core = 1.0 + alpha * (p >= thr_p).to(dtype=p.dtype)  # [B, 1, H, W]
                    else:
                        gamma = float(self.rg.prob_gamma) # Exponent for probability weighting
                        w_core = 1.0 + alpha * (p.clamp(0,1) ** gamma)  # [B, 1, H, W]

                    # Apply warm-start ramp factor in [0,1]
                    if self.rg.ramp_epochs > 0:
                        # Cosine ramp from 0 to 1 over rg_ramp epochs after rg_warm_start
                        phase = min(1.0, max(0.0, (current_epoch - self.rg.warm_start_epochs) / max(1, self.rg.ramp_epochs)))
                        ramp_prog = 0.5 * (1 - math.cos(math.pi * phase))  # Cosine ramp from 0 to 1
                    else:
                        ramp_prog = 1.0

                    # Blend towards identity weight = 1 using ramp_prog
                    w = 1.0 + (w_core - 1.0) * ramp_prog
                    
                    w = w.clamp(min=1.0, max=clip_max)
                    if detach_w:
                        w = w.detach()
                    pixel_weight_map = w  # [B, 1, H, W]

            # === Diagnostics checks: asserts and 
            # Check raw inputs for NaNs or Infs
            self._assert_all_finite('x', x)
            self._assert_all_finite('cond_images', cond_images)
            self._assert_all_finite('lr_ups_baseline', lr_ups_baseline)

            do_log = self.diag_per_batch
            every = self.diag_log_every
            
            # Debug: save viz occasionally (show probs if no weight map yet) - only per ten epochs to limit storage
            viz_every_n_epochs = self.diag_viz_every_epochs
            if do_log and (wet_logits is not None) and (current_epoch % viz_every_n_epochs == 0):
                _save_weight_map_viz(
                    weight_map=pixel_weight_map if pixel_weight_map is not None else torch.sigmoid(wet_logits),
                    wet_probs=torch.sigmoid(wet_logits),
                    wet_target=locals().get('wet_target', None),
                    epoch=current_epoch, step=idx, prefix='train', save_path=self.paths.diagnostics_dir
                )

            hr = x
            lr_hr = lr_ups_baseline
            # Get the lr_lr as the cond_image that corresponds to the hr_var, if available
            if cond_images is not None and (self.hr_var in self.lr_vars):
                idx_hr_in_cond = self.lr_vars.index(self.hr_var)
                lr_lr = cond_images[:, idx_hr_in_cond:idx_hr_in_cond+1, :, :]  # [B, 1, H, W]
            else:
                lr_lr = None

            residual = hr - lr_hr if (hr is not None and lr_hr is not None) else None
            if do_log and (current_epoch % every == 0):
                tensor_stats(hr, "train/hr_norm")
                if lr_hr is not None:
                    tensor_stats(lr_hr, "train/lr_hr_norm")
                if residual is not None:
                    tensor_stats(residual, "train/residual_hr_space")
                
            # OPTIONAL: Clamp warnings
            clamp_warn = self.diag_warn_abs
            if do_log and (idx % every == 0) and (clamp_warn > 0.0):
                mx = float(residual.abs().amax().item()) if residual is not None else float('nan')
                if mx > clamp_warn:
                    logger.warning(f"[diagnostics][train] Batch {idx}: |residual| max {mx:.2f} exceeds warn_if_abs_gt {clamp_warn}. Consider residual normalization, tail clamp or loss robustification.")



            # # === CFG dropout (training) ===
            cfg_dropout_result = apply_cfg_dropout(
                cond_images, lsm, topo, seasons, lr_ups_baseline, self.guidance
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
                                               lr_ups=lr_ups_baseline,
                                               pixel_weight_map=pixel_weight_map
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
                                           lr_ups=lr_ups_baseline,
                                           pixel_weight_map=pixel_weight_map
                                       )
            # Add rain-gate auxiliary loss if available
            if rg_aux_loss is not None:
                batch_loss = batch_loss + rg_aux_loss
            # Make sure loss is finite
            self._assert_all_finite('batch_loss', batch_loss)


            # === In-loop monitoring (lightweight): cosine and HR-LR correlation ===
            log_every = self.edm_metrics_every
            global_step = (current_epoch - 1) * len(dataloader) + idx
            edm_on = self.edm.enabled

            if edm_on and log_every > 0 and (global_step % log_every == 0):
                metrics = in_loop_metrics(loss_obj=self.loss_fn, model=self.model,
                    x0=x, y=seasons, cond_img=cond_images, lsm_cond=lsm, topo_cond=topo,
                    lr_ups=lr_ups_baseline, eval_land_only=self.eval_land_only)

                self.live_metrics['steps'].append(global_step)
                self.live_metrics['edm_cosine'].append(float(metrics.get('edm_cosine', float('nan')))) # type: ignore
                self.live_metrics['hr_lr_corr'].append(float(metrics.get('hr_lr_corr', float('nan')))) # type: ignore

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
            if idx % self.trainp.train_postfix_every == 0:
                pbar.set_postfix(loss=loss_sum / (idx+1), rg_bce=float(rg_aux_loss.item()) if rg_aux_loss is not None else None)
        
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
        logger.info(f"→ Classifier-Free Guidance (CFG) enabled: {self.guidance.enabled}")
        if self.guidance.enabled:
            logger.info(f"   ▸ Dropout probability for LR conditions:   {self.guidance.prob_drop_lr}")
            logger.info(f"   ▸ Dropout probability for static geo:      {self.guidance.drop_prob_geo}")
            logger.info(f"   ▸ Guidance scale (max):                    {self.guidance.guidance_scale} ({self.guidance.guidance_scale_max})")

        # Log EMA
        logger.info(f"→ EMA enabled: {self.with_ema}; decay: {getattr(self, 'ema_decay', None)}; eval_use_ema: {self.trainp.eval_use_ema}")

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
                self.save_model(dirname=self.paths.checkpoint_dir, filename=self.checkpoint_name)
                logger.info(f"→ Best model saved with validation loss: {best_loss:.4f} at epoch {epoch}.")
                logger.info(f"→ Checkpoint saved to {os.path.join(self.paths.checkpoint_dir, self.checkpoint_name)}")


            # Pickle dump the losses
            losses = {
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            with open(os.path.join(self.paths.metrics_dir, 'losses' + f'_{self.model_string}.pkl'), 'wb') as f:
                pickle.dump(losses, f)

            vis = self.mon.visualizations
            
            if vis.enabled and ('loss' in vis.plots_n_epoch):
                # Plot the losses
                self.plot_losses(train_losses,
                                 val_losses=val_losses,
                                 save_path=self.paths.metrics_dir,
                                 save_name=f'losses_plot_{self.model_string}.png',
                                 show_plot=vis.show)
            # Plot in-loop timeseries occasionally
            if (self.monitor_plot_every_n_epochs > 0) and (epoch % self.monitor_plot_every_n_epochs == 0):
                try:
                    self._plot_live_metrics(self.paths.metrics_dir, n_samples=vis.n_plot_samples)
                except Exception as e:
                    logger.warning(f"[monitor] Could not plot live metrics at epoch {epoch}. Error: {e}")
            
            # Generate and save samples, if create_figs is True
            if vis.enabled and vis.n_plot_samples > 0:
                # Only generate and plot if loss improved or every n epochs if configured
                gen_every_n_epochs = max(1, vis.gen_and_plot_every_n_epochs)

                if gen_every_n_epochs < 1:
                    gen_every_n_epochs = 1  # Ensure at least every 
                    
                on_schedule = (epoch % gen_every_n_epochs == 0)
                if improved or on_schedule:
                    if on_schedule:
                        logger.info(f"→ Generating and plotting samples at epoch {epoch} (every {gen_every_n_epochs} epochs)...")
                    if improved:
                        logger.info(f"→ Generating and plotting samples at epoch {epoch} (new best model)...")
                    self.generate_and_plot_samples(gen_dataloader,
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
        edm_on = self.edm.enabled

        # Choose eval model (EMA if enabled and configured)
        use_ema_for_val = self.trainp.eval_use_ema
        model_eval = self.ema_model if (self.with_ema and use_ema_for_val and hasattr(self, 'ema_model')) else self.model

        # Set initial loss to 0
        loss = 0.0
        # Set the progress bar
        pbar = tqdm.tqdm(dataloader, desc=f"Epoch {current_epoch}/{epochs}", unit="batch")

        # Reliability buffers (collect across validation epoch)
        rel_probs: list[torch.Tensor] = []
        rel_targets: list[torch.Tensor] = []

        # Iterate through batches in dataloader (tuple of images and seasons)
        for idx, samples in enumerate(pbar):
            # Samples is a dict with following available keys: 'img', 'classifier', 'img_cond', 'lsm', 'sdf', 'topo', 'points'
            # Extract samples
            x, seasons, cond_images, lsm_hr, lsm, sdf, topo, hr_points, lr_points = extract_samples(samples, self.device)



            # Setup lr_ups_baseline if needed
            lr_ups_baseline = None
            if edm_on and self.edm.predict_residual:
                lr_ups_baseline = self._build_lr_ups_baseline(cond_images)  # [B, 1, H, W]

            # === Optional: gate-based diagnostics and (optionally) reweighting in validation ===
            pixel_weight_map = None
            wet_logits_val = None

            if self.rg.enabled and (self.rain_gate is not None):
                # Build gate inputs by concatenation at HR resolution
                gate_inputs = []
                if cond_images is not None: gate_inputs.append(cond_images)
                if self.rg.include_lsm and (lsm is not None): gate_inputs.append(lsm)
                if self.rg.include_topo and (topo is not None): gate_inputs.append(topo)
                if self.rg.include_lr_baseline and (lr_ups_baseline is not None): gate_inputs.append(lr_ups_baseline)

                if len(gate_inputs) > 0 and self.rain_gate is not None:
                    gate_x = torch.cat(gate_inputs, dim=1)  # [B, C_in, H, W]

                    # Always compute logits for diagnostics (even if not reweighting)
                    with torch.no_grad():
                        wet_logits_val = self.rain_gate(gate_x)
                        p = torch.sigmoid(wet_logits_val)
                else:
                    wet_logits_val = None


                # Build pixel_weight_map only if reweighting is enabled and past warm-start
                do_reweight = self.rg.reweight_enabled and (current_epoch > self.rg.warm_start_epochs)
                if do_reweight and (wet_logits_val is not None):
                    p = torch.sigmoid(wet_logits_val)  # [B, 1, H, W] probabilities

                    # Base weighting shape from config
                    strategy = str(self.rg.weight_strategy)  # 'prob' or 'binary'
                    alpha = float(self.rg.weight_alpha)  # Weighting strength
                    clip_max = float(self.rg.clip_max)  # Max clip for weights

                    if strategy == 'binary':
                        thr_p = float(self.rg.binary_threshold)
                        core = (p >= thr_p).to(dtype=p.dtype)
                    else:
                        gamma = float(self.rg.prob_gamma)
                        core = (p.clamp(0,1) ** gamma)

                    # Optional ramp (keep consistent with train)
                    if self.rg.ramp_epochs > 0:
                        phase = min(1.0, max(0.0, (current_epoch - self.rg.warm_start_epochs) / max(1, self.rg.ramp_epochs)))
                        ramp_prog = 0.5 * (1 - math.cos(math.pi * phase))
                    else:
                        ramp_prog = 1.0

                    w = 1.0 + ( (1.0 + alpha * core) - 1.0 ) * ramp_prog
                    pixel_weight_map = w.clamp(min=1.0, max=clip_max).detach()

            # Reliability buffers + debug viz (now independent of reweighting)
            rg_val_bce = None
            rg_val_bce_t = None
            if wet_logits_val is not None:
                # Build wet_target for reliability/BCE logging
                try:
                    with torch.no_grad():
                        thr = float(self.rg.wet_threshold_mm)
                        bt_hr = self.back_transforms_train.get(self.bt_hr_key, None) if self.back_transforms_train is not None else None
                        if callable(bt_hr):
                            x_phys = bt_hr(x)
                            if not isinstance(x_phys, torch.Tensor):
                                x_phys = torch.tensor(x_phys, dtype=torch.float32)
                            wet_target = (x_phys > thr).to(dtype=torch.float32)
                        else:
                            wet_target = (x > self.rg.wet_threshold_model_space).to(dtype=torch.float32)
                        if wet_target.shape[1] != 1:
                            wet_target = wet_target[:, :1, :, :]
                    pos_w = torch.tensor(self.rg.pos_weight, device=x.device, dtype=torch.float32)
                    # Tensor BCE for adding to loss
                    rg_val_bce_t = F.binary_cross_entropy_with_logits(wet_logits_val, wet_target, pos_weight=pos_w)
                    # Optional scalr for logging
                    rg_val_bce = float(rg_val_bce_t.item())
                except Exception as e:
                    logger.warning(f"[rain_gate] Could not compute validation BCE or target. Error: {e}")
                    wet_target = None
                    rg_val_bce_t = None
                    rg_val_bce = None

                rel_probs.append(torch.sigmoid(wet_logits_val).detach().flatten())
                rel_targets.append(wet_target.detach().flatten() if wet_target is not None else torch.zeros_like(wet_logits_val.detach()).flatten())

                if (idx % max(1, self.diag_log_every) == 0):
                    _save_weight_map_viz(
                        weight_map=pixel_weight_map if pixel_weight_map is not None else torch.sigmoid(wet_logits_val),
                        wet_probs=torch.sigmoid(wet_logits_val),
                        wet_target=wet_target,
                        epoch=current_epoch, step=idx, prefix='val', save_path=self.paths.diagnostics_dir
                    )


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
                                             lr_ups=lr_ups_baseline,
                                             pixel_weight_map=pixel_weight_map
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
                                         lr_ups=lr_ups_baseline,
                                         pixel_weight_map=pixel_weight_map
                                     )
                # Add rain-gate BCE to loss 
                if (self.rg.enabled and (rg_val_bce_t is not None) and (self.rg.loss_weight_bce > 0.0)):
                    batch_loss = batch_loss + rg_val_bce_t * self.rg.loss_weight_bce

                # === Cosine monitoring (validation; lightweight) ===
                log_every = self.edm_metrics_every
                if edm_on and log_every > 0 and (idx % log_every == 0):
                    metrics = in_loop_metrics(loss_obj=self.loss_fn, model=self.model,
                        x0=x, y=seasons, cond_img=cond_images, lsm_cond=lsm, topo_cond=topo,
                        lr_ups=lr_ups_baseline, eval_land_only=self.eval_land_only)
                    if verbose and metrics is not None:
                        logger.info(f"→ [monitor][val] Step {idx}: EDM cosine metric: {metrics.get('edm_cosine', float('nan')):.4f}")
                        logger.info(f"→ [monitor][val] Step {idx}: HR-LR corr: {metrics.get('hr_lr_corr', float('nan')):.4f}")

            # Add batch loss to total loss
            loss += batch_loss.item()
            # Update the bar
            if idx % self.trainp.train_postfix_every == 0:
                pbar.set_postfix(loss=loss/(idx+1), rg_bce=rg_val_bce if wet_logits_val is not None else None)

        # Plot reliability for this validation epoch if data collected
        if len(rel_probs) > 0 and len(rel_targets) > 0:
            try:
                probs_all = torch.cat(rel_probs, dim=0)
                targets_all = torch.cat(rel_targets, dim=0)
                rel_path = os.path.join(self.paths.checkpoint_dir, f'reliability_epoch{current_epoch:03d}.png')
                _plot_reliability_curve(probs_all, targets_all, bins=15, save_path=rel_path, title=f'Rain Gate Reliability Epoch {current_epoch}')
                logger.info(f"[debug][rain_gate] Saved reliability plot to {rel_path}")
            except Exception as e:
                logger.warning(f"[debug][rain_gate] Could not plot reliability at epoch {current_epoch}. Error: {e}")

        # Calculate average loss
        avg_loss = loss / len(dataloader)

        # Print average loss if verbose
        if verbose:
            logger.info(f'→ Validation Loss: {avg_loss:.4f}')

        return avg_loss
    
    def generate_and_plot_samples(self,
                            gen_dataloader,
                            epoch,
                          ):
        
        # Load the best model (EMA or network) from checkpoint WITHOUT altering training weights
        model_sd_backup = copy.deepcopy(self.model.state_dict())  # Backup current model state dict

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        net_sd = checkpoint.get('network_params', None) # Network state dict
        ema_sd = checkpoint.get('ema_network_params', None) # EMA state dict if exists
        use_ema_for_gen = self.trainp.eval_use_ema

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

        # NOTE: Remove any VE-DSM sampler references here; only use EDM 
        sampler_fn = None  # Ensure sampler_fn is always defined
        if self.sampler.use_edm:
            sampler_edm = edm_sampler
        else:
            sampler_edm = None
            st = self.sampler.sampler_type
            if st == 'pc_sampler':
                sampler_fn = pc_sampler
            elif st == 'Euler_Maruyama_sampler':
                sampler_fn = Euler_Maruyama_sampler
            elif st == 'ode_sampler':
                sampler_fn = ode_sampler
            else: raise ValueError(f"Sampler type {st} not recognized. Please choose from 'pc_sampler', 'Euler_Maruyama_sampler', or 'ode_sampler'.")

        back_transforms = build_back_transforms_from_stats(
                            hr_var              = self.hr_var,
                            hr_model            = self.data.hr_model,
                            domain_str_hr       = self.full_domain_dims_str_hr,
                            crop_region_str_hr  = self.crop_region_hr_str,
                            hr_scaling_method   = self.data.hr_scaling_method,
                            hr_buffer_frac      = self.data.buffer_frac_hr,
                            lr_vars             = self.lr_vars,
                            lr_model            = self.data.lr_model,
                            domain_str_lr       = self.full_domain_dims_str_lr,
                            crop_region_str_lr  = self.crop_region_lr_str,
                            lr_scaling_methods  = self.data.lr_scaling_methods,
                            lr_buffer_frac      = self.data.buffer_frac_lr,
                            split               = 'all',
                            stats_dir_root      = self.paths.stats_dir,
                            eps=self.global_prcp_eps
                            )
        
        # Set visualization parameters
        vis = self.mon.visualizations

        p_bar = tqdm.tqdm(gen_dataloader, desc=f"Generating samples for epoch {epoch}", unit="batch") # type: ignore
        # Iterate through batches in dataloader
        for idx, samples in enumerate(p_bar):
            # Get dates for titles
            if 'date' in samples and isinstance(samples['date'], (list, tuple)) and len(samples['date']) > 0:
                dates = samples['date']
            else:
                dates = None

            # Samples is a dict with following available keys: 'img', 'classifier', 'img_cond', 'lsm', 'sdf', 'topo', 'points'
            # Extract samples
            x_gen, seasons_gen, cond_images_gen, lsm_hr_gen, lsm_gen, sdf_gen, topo_gen, hr_points_gen, lr_points_gen = extract_samples(samples, self.device)

            # Restrict batch for plotting
            bs = int(max(1, min(getattr(vis, 'n_plot_samples', x_gen.shape[0]), 3)))
            x_gen = x_gen[:bs]
            seasons_gen = seasons_gen[:bs] if seasons_gen is not None else None
            cond_images_gen = cond_images_gen[:bs] if cond_images_gen is not None else None
            lsm_hr_gen = lsm_hr_gen[:bs] if lsm_hr_gen is not None else None
            lsm_gen = lsm_gen[:bs] if lsm_gen is not None else None
            topo_gen = topo_gen[:bs] if topo_gen is not None else None

            logger.info(f"→ Generating {bs} samples at epoch {epoch}...")

            # LR baseline for residual EDM
            lr_ups_baseline = None
            if self.edm.enabled and self.edm.predict_residual:
                lr_ups_baseline = self._build_lr_ups_baseline(cond_images_gen)
                if lr_ups_baseline is not None:
                    lr_ups_baseline = lr_ups_baseline[:bs]

            # Run the sampler
            if self.edm.enabled and sampler_edm is not None:
                guidance_dict = None
                if self.guidance.enabled:
                    guidance_dict = {
                        'enabled': True,
                        'guidance_scale': float(self.guidance.guidance_scale),
                        'guidance_scale_max': float(self.guidance.guidance_scale_max),
                        'sigma_weighted': bool(self.guidance.sigma_weighted),
                        'drop_lr_ups_in_uncond': bool(self.guidance.drop_lr_ups_in_uncond),
                        'prob_drop_lr': float(self.guidance.prob_drop_lr),
                        'drop_prob_geo': float(self.guidance.drop_prob_geo),
                    }

                generated_samples = sampler_edm(
                    score_model=self.model,
                    batch_size=bs,
                    num_steps=int(self.sampler.steps),
                    device=self.device,
                    img_size=int(x_gen.shape[-1]),
                    y=seasons_gen,
                    cond_img=cond_images_gen,
                    lsm_cond=lsm_gen,
                    topo_cond=topo_gen,
                    sigma_min=float(self.sampler.sigma_min),
                    sigma_max=float(self.sampler.sigma_max),
                    rho=float(self.sampler.rho),
                    S_churn=float(self.sampler.S_churn),
                    S_min=float(self.sampler.S_min),
                    S_max=float(self.sampler.S_max if math.isfinite(self.sampler.S_max) else 0.0),
                    S_noise=float(self.sampler.S_noise),
                    lr_ups=lr_ups_baseline,
                    cfg_guidance=guidance_dict,
                )
            elif sampler_fn is not None:
                generated_samples = sampler_fn(
                    score_model=self.model,
                    marginal_prob_std=self.marginal_prob_std_fn,
                    diffusion_coeff=self.diffusion_coeff_fn,
                    batch_size=bs,
                    num_steps=int(self.sampler.steps),
                    device=self.device,
                    img_size=int(x_gen.shape[-1]),
                    y=seasons_gen,
                    cond_img=cond_images_gen,
                    lsm_cond=lsm_gen,
                    topo_cond=topo_gen,
                )
            else:
                raise ValueError("No valid sampler found. Check the sampler configuration.")

            gen_model = generated_samples.detach().cpu().float()  # Keep sampler output in model space on CPU (preserve batch dim!)

            # Back-transform for metrics
            if back_transforms is not None:
                bt_gen = back_transforms.get(self.bt_gen_key, None)
                bt_hr = back_transforms.get(self.bt_hr_key, None)
                gen_phys = bt_gen(gen_model) if callable(bt_gen) else gen_model
                hr_phys = bt_hr(x_gen) if callable(bt_hr) else x_gen
            else:
                gen_phys, hr_phys = gen_model, x_gen

            # Physical exceedance diagnostics
            if not isinstance(gen_phys, torch.Tensor):
                gen_phys = torch.tensor(gen_phys)
            tensor_stats(gen_phys, "eval/x_phys")
            if float(gen_phys.max().item()) > self.diag_warn_phys:
                logger.warning(f"[diagnostics][eval] Generated samples exceed {self.diag_warn_phys} {self.hr_unit}. Max: {float(gen_phys.max().item()):.2f} {self.hr_unit}.")

            # Extreme sentinel (optional clamp)
            try:
                thr = self.extreme_threshold_mm
                chk = report_precip_extremes(x_bt=gen_phys, name="generated_hr", cap_mm_day=thr)
                if chk.get('has_extreme', False) and self.extreme_clamp_in_gen:
                    clamp_max = self.extreme_clamp_max if self.extreme_clamp_max is not None else thr
                    gen_phys = torch.clamp(gen_phys, min=0.0, max=clamp_max)
                    logger.warning(f"[monitor][gen] Clamped generated samples to ≤ {clamp_max} mm/day.")
            except Exception as e:
                logger.warning(f"[monitor] Could not run extreme sentinel. Error: {e}")

            # Epoch-level metrics (physical)
            gen_phys = gen_phys.detach().cpu()
            if isinstance(hr_phys, torch.Tensor):
                hr_phys = hr_phys.detach().cpu()
            else:
                hr_phys = torch.tensor(hr_phys)
            if lsm_gen is not None:
                lsm_gen = (lsm_gen.detach().cpu() if isinstance(lsm_gen, torch.Tensor) else torch.tensor(lsm_gen))

            mask = (lsm_gen >= 0.5).float() if (self.eval_land_only and (lsm_gen is not None)) else None

            fss_dict = compute_fss_at_scales(gen_phys, hr_phys, mask=mask,
                                            fss_km=self.fss_scales_km, grid_km_per_px=self.pixel_km,
                                            thr_mm=self.fss_threshold_mm)
            self.fss_hist.append(fss_dict)
            self.epoch_list.append(epoch)
            plot_fss_history(self.fss_hist, epoch_list=self.epoch_list,
                            save_dir=self.paths.metrics_dir, filename="fss_history.png",
                            title="FSS history" + (" (land-only)" if self.eval_land_only else ""), n_samples=len(gen_phys))

            psd_dict = compute_psd_slope(gen_phys, hr_bt=hr_phys if self.psd_compare_to_hr else None, mask=mask)
            self.psd_hist.append(psd_dict)
            plot_psd_slope_history(self.psd_hist, epoch_list=self.epoch_list,
                                save_dir=self.paths.metrics_dir, filename="psd_history.png",
                                title="PSD slope history" + (" (land-only)" if self.eval_land_only else ""), n_samples=len(gen_phys))

            q_dict = compute_p95_p99_and_wet_day(gen_phys, hr_bt=hr_phys if self.quantiles_compare_to_hr else None,
                                                mask=mask, wet_threshold_mm=self.wetday_thresh)
            self.q_hist.append(q_dict)
            plot_quantiles_wetday_history(self.q_hist, epoch_list=self.epoch_list,
                                        save_dir=self.paths.metrics_dir, filename="quantiles_history.png",
                                        title="Quantiles & wet-day history" + (" (land-only)" if self.eval_land_only else ""), n_samples=len(gen_phys))

            # Plot samples (model space) — build a tiny cfg for plotting_utils if it still expects one
            if vis.enabled and ("loss" not in vis.plots_n_epoch):
                cfg_plot = {
                    'visualization': {
                        'transform_back_bf_plot': bool(vis.transform_back_bf_plot),
                        'show_figs': bool(vis.show),
                        'save_figs': bool(vis.save),
                    },
                    'highres': {'variable': self.hr_var},
                    'lowres': {'condition_variables': self.lr_vars},
                }
                fig, _ = plot_samples_and_generated(
                    samples=samples,
                    generated=gen_model,
                    cfg=cfg_plot,
                    transform_back_bf_plot=vis.transform_back_bf_plot,
                    back_transforms=back_transforms,
                    dates=dates,
                )
                if vis.save:
                    out_fig = os.path.join(self.paths.figures_dir, f'epoch_{epoch}_generatedSamples.png')
                    fig.savefig(out_fig, dpi=300, bbox_inches='tight')
                    logger.info(f"→ Figure saved to {out_fig}")
                plt.close(fig)
                break  # one batch per epoch

        # Restore training weights and mode
        self.model.load_state_dict(model_sd_backup)
        self.model.train()


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

    def _plot_live_metrics(self, save_dir: str, n_samples: Optional[int] = None):
        """
        Internal method to plot live training metrics if enabled in the configuration.
        Args:
            save_dir (str): Directory where the metrics plot will be saved.
            n_samples (Optional[int]): Number of samples used for computing metrics, for annotation.
        """
        if len(self.live_metrics['steps']) == 0:
            return

        out = os.path.join(self.paths.metrics_dir, 'inLoop_metrics_timeseries.png')

        try:
            plot_live_training_metrics(
                self.live_metrics['steps'],
                self.live_metrics['edm_cosine'],
                self.live_metrics['hr_lr_corr'],
                save_dir=self.paths.metrics_dir,
                filename='inLoop_metrics_timeseries.png',
                show=self.mon.visualizations.show,
                title="In-loop training metrics (EDM cosine, HR-LR corr)",
                land_only=self.eval_land_only,
                n_samples=n_samples
            )
            logger.info(f"→ Live metrics plot saved to {out}")

        except Exception as e:
            logger.error(f"[Monitor] Could not save live metrics plot to {out}. Error: {e}")

        
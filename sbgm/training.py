import os
import torch
import copy
import math
import pickle
import tqdm
import logging 

import torch.nn as nn
import matplotlib.pyplot as plt

from torch.cuda.amp import autocast, GradScaler

from sbgm.special_transforms import build_back_transforms, build_back_transforms_from_stats
from sbgm.utils import extract_samples, plot_samples_and_generated, report_precip_extremes
from sbgm.data_modules import *
# from sbgm.score_unet import loss_fn, marginal_prob_std_fn, diffusion_coeff_fn
from sbgm.score_sampling import Euler_Maruyama_sampler, pc_sampler, ode_sampler
from sbgm.training_utils import get_model_string, get_cmaps, get_units, get_loss_fn
from sbgm.monitoring import edm_cosine_metric

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
        self.model.debug_pre_sigma_div = cfg['training'].get('debug_pre_sigma_div', True)

        self.marginal_prob_std_fn = marginal_prob_std_fn
        self.diffusion_coeff_fn = diffusion_coeff_fn
        self.optimizer = optimizer
        # self.loss_fn = loss_fn
        self.loss_fn = get_loss_fn(self.cfg, marginal_prob_std_fn=getattr(self, 'marginal_prob_std_fn', None))

        self.lr_scheduler = lr_scheduler

        self.scaling = cfg['transforms']['scaling']
        self.hr_var = cfg['highres']['variable']
        self.hr_scaling_method = cfg['highres']['scaling_method']
        self.full_domain_dims_hr = cfg['highres']['full_domain_dims']
        self.crop_region_hr = cfg['highres']['cutout_domains']
        # self.hr_scaling_params = cfg['highres']['scaling_params']
        self.lr_vars = cfg['lowres']['condition_variables']
        self.lr_scaling_methods = cfg['lowres']['scaling_methods']
        self.full_domain_dims_lr = cfg['lowres']['full_domain_dims']
        self.crop_region_lr = cfg['lowres']['cutout_domains']
        # self.lr_scaling_params = cfg['lowres']['scaling_params']
        
        self.weight_init = cfg['training']['weight_init']
        self.custom_weight_initializer = cfg['training']['custom_weight_initializer']
        self.sdf_weighted_loss = cfg['training']['sdf_weighted_loss']
        self.with_ema = cfg['training']['with_ema']

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

        # Set Exponential Moving Average (EMA) if needed
        if self.with_ema:
            #!!!!! NOTE: EMA is not implemented yet, this is a placeholder for future implementation"
            # Create a copy of the model for EMA
            self.ema_model = copy.deepcopy(self.model)
            # Detach the EMA model parameters to not update them
            for param in self.ema_model.parameters():
                param.detach_()

        # Set up checkpoint directory, name and path
        # ======== !!!!!!!!!!!!! CHANGE CHECKPOINT_NAME TO BE FULL get_model_string !!!!!!!!!!!!! ==========
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
                stats_dir_root=cfg['paths']['stats_load_dir']
            )
        except Exception as e:
            logger.warning(f"[monitor] Could not build back transforms for sentinel; will skip back_transform in training. Error: {e}")
            self.back_transforms_train = None

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

    def load_checkpoint(self,
                        checkpoint_path,
                        load_ema=False,
                        # If load_ema is True, load the EMA model parameters
                        # If load_ema is False, load the model parameters
                        device=None
                        ):
        '''
            Load a checkpoint from the given path.
            Args:
                checkpoint_path: Path to the checkpoint file.
                device: Device to load the checkpoint on. If None, uses the current device.
        '''
        # Check if device is provided, if not, use the current device
        if device is None:
            device = self.device
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)['network_params']
        # Load the state dict into the model
        self.model.load_state_dict(checkpoint)

    def save_model(self,
                   dirname='./model_params',
                   filename='SBGM.pth'
                   ):
        '''
            Save the model parameters.
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

            # # Log the shapes of the extracted samples
            # logger.info(f"▸ Shape of x: {x.shape}")
            # logger.info(f"▸ Shape of seasons: {seasons.shape}")
            # logger.info(f"▸ Shape of cond_images: {cond_images.shape if cond_images is not None else 'None'}")
            # logger.info(f"▸ Shape of lsm: {lsm.shape if lsm is not None else 'None'}")
            # logger.info(f"▸ Shape of topo: {topo.shape if topo is not None else 'None'}")

            # # Apply Classifier Free Guidance conditioning dropout if enabled
            # cfg_guidance = getattr(self, "cfg", {}).get('classifier_free_guidance', None)
            # if cfg_guidance and cfg_guidance.get('enabled', False) and cond_images is not None:
            #     # logger.info("▸ Applying Classifier Free Guidance conditioning dropout...")
            #     drop_prob = cfg_guidance.get('drop_prob', 0.1)
            #     # Make sure the batch size and device keeps consistent
            #     batch_size = cond_images.size(0)
            #     device = cond_images.device

            #     # Create a drop mask, that randomly drops drop_prob% of the cond_images in the batch (B,) for scalar condition
            #     drop_mask = (torch.rand(batch_size, device=device) < drop_prob)

            #     # Expand drop mask for image tensors (B, 1, 1, 1)
            #     drop_mask_img = drop_mask.view(-1, 1, 1, 1)

            #     # Nullify image-like conditions
            #     null_cond = torch.zeros_like(cond_images)
            #     cond_images = torch.where(drop_mask_img, null_cond, cond_images)
            #     if lsm is not None:
            #         null_lsm = torch.zeros_like(lsm)
            #         lsm = torch.where(drop_mask_img, null_lsm, lsm)
            #     if topo is not None:
            #         null_topo = torch.zeros_like(topo)
            #         topo = torch.where(drop_mask_img, null_topo, topo)

            #     # Nullify scalar condition
            #     if seasons is not None:
            #         null_season = torch.zeros_like(seasons)
            #         seasons = torch.where(drop_mask.squeeze(), null_season, seasons)

                # logger.info(f"\n▸ [CFG] Dropped {drop_mask.sum().item()} out of {batch_size} conditions ({drop_prob*100:.1f}%) in the batch.")

            # Zero gradients
            self.optimizer.zero_grad()
            
            # # Use mixed precision training if needed
            # if self.scaler:
            #     with autocast():
            #         # Pass the score model and samples+conditions to the loss_fn
            #         batch_loss = loss_fn(self.model,
            #                              x,
            #                              self.marginal_prob_std_fn,
            #                              y=seasons,
            #                              cond_img=cond_images,
            #                              lsm_cond=lsm,
            #                              topo_cond=topo,
            #                              sdf_cond=sdf)
            #     # Mixed precision: scale loss and update weights
            #     self.scaler.scale(batch_loss).backward()
            #     # Update weights
            #     self.scaler.step(self.optimizer)
            #     # Update scaler
            #     self.scaler.update()
            # else:
            # logger.info("▸ Computing batch loss without mixed precision...")
                # Log the shapes of the inputs for debugging
            for name, tensor in zip(['x', 'seasons', 'cond_images', 'lsm', 'topo'], [x, seasons, cond_images, lsm, topo]):
                if tensor is not None:
                    assert tensor.device == x.device, f"{name} is on device {tensor.device}, expected {x.device}"
            
            if hasattr(self, 'scaler') and self.scaler:
                with autocast():
                    # Pass the score model and samples+conditions to the loss_fn
                    batch_loss = self.loss_fn(self.model,
                                               x,
                                               self.marginal_prob_std_fn,
                                               y=seasons,
                                               cond_img=cond_images,
                                               lsm_cond=lsm,
                                               topo_cond=topo,
                                               sdf_cond=sdf)
            else:
                # No mixed precision, just pass the score model and samples+conditions to the loss_fn
                batch_loss = self.loss_fn(self.model,
                                           x,
                                           self.marginal_prob_std_fn,
                                           y=seasons,
                                           cond_img=cond_images,
                                           lsm_cond=lsm,
                                           topo_cond=topo,
                                           sdf_cond=sdf)


            # === Cosine monitoring (lightweight) ===
            monitor_cfg = self.cfg.get('monitoring', {})
            log_every = monitor_cfg.get('edm_metrics_every', 50)
            global_step = (current_epoch - 1) * len(dataloader) + idx
            edm_on = self.cfg.get('edm', {}).get('enabled', False)

            if edm_on and log_every > 0 and (global_step % log_every == 0):
                cos = edm_cosine_metric(self.model, x, self.marginal_prob_std_fn, y=seasons, cond_img=cond_images, lsm_cond=lsm, topo_cond=topo)
                if cos is not None:
                    pbar.set_postfix(loss=loss_sum / (idx + 1), edm_cosine=cos)
                    if verbose:
                        logger.info(f"→ [monitor][train] Step {idx}: EDM cosine metric: {cos:.4f}")
                    if self.writer is not None:
                        self.writer.add_scalar('monitoring/edm_cosine_metric_train', cos, (current_epoch - 1) * len(dataloader) + idx)

            # === Extreme-prcp sentinel on ground-truth HR (optional; lightweight) ===
            if self.extreme_enabled and (global_step % self.extreme_every_step == 0):
                try:
                    # x is in model space; optionally back-transform to physical mm/day
                    x_for_check = x.detach()
                    if self.extreme_backtransform and self.back_transforms_train is not None:
                        # Expect a callable for HR back-transform under key 'hr'
                        bt = self.back_transforms_train.get('hr', None)
                        if bt is not None:
                            if callable(bt):
                                x_bt = bt(x_for_check.detach().cpu())
                            else:
                                logger.warning(f"[monitor] Back-transform object for HR is not callable and has no 'transform' method.")
                                x_bt = x_for_check.detach().cpu()
                        else:
                            x_bt = x_for_check.detach().cpu()
                    else:
                        x_bt = x_for_check.detach().cpu()
                    # Run helper (accepts torch or numpy)
                    # Ensure x_bt is a torch.Tensor before passing to report_precip_extremes
                    if not isinstance(x_bt, torch.Tensor):
                        x_bt = torch.tensor(x_bt)
                    check = report_precip_extremes(x_bt=x_bt, name="ground_truth_hr", cap_mm_day=self.extreme_threshold_mm)
                    # "check" is boolean: True if any extreme values found
                    has_extreme = check.get('has_extreme', False)
                    n_extreme = check.get('n_extreme', 0)
                    extreme_values = check.get('extreme_values', [])
                    has_below_zero = check.get('has_below_zero', False)
                    n_below_zero = check.get('n_below_zero', 0)
                    below_zero_values = check.get('below_zero_values', [])

                    if has_extreme and self.extreme_log_first_n > 0:
                        # Extract some stats if provided
                        mx = max(extreme_values) if isinstance(extreme_values, list) else None
                        cnt = len(extreme_values) if isinstance(extreme_values, list) else None
                        logger.warning(f"[monitor][train] Extreme precipitation detected at step {idx}:")
                        logger.warning(f"               max={mx:.1f} mm/day, count={cnt}, threshold={self.extreme_threshold_mm} mm/day")
                        self.extreme_log_first_n -= 1  # Decrement counter to log fewer next times

                except Exception as e:
                    logger.warning(f"[monitor] Could not check for extreme precipitation in training step {idx}. Error: {e}")


            # logger.info(f"▸ Batch loss computed: {batch_loss.item():.4f}")
            # Add anomaly detection for loss
            # with torch.autograd.detect_anomaly():
            # Backward pass
            batch_loss.backward()
            # Update weights
            self.optimizer.step()

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
                logger.info(f"▸ Starting epoch {epoch}/{epochs}...")

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

            # If validation loss is lower than best loss, save the model
            if val_loss < best_loss:
                best_loss = val_loss
                # Save the model
                self.save_model(dirname=self.checkpoint_dir, filename=self.checkpoint_name)
                logger.info(f"→ Best model saved with validation loss: {best_loss:.4f} at epoch {epoch}.")
                logger.info(f"→ Checkpoint saved to {os.path.join(self.checkpoint_dir, self.checkpoint_name)}\n\n")


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

            # Generate and save samples, if create_figs is True
            if cfg['visualization']['create_figs'] and cfg['data_handling']['n_gen_samples'] > 0:
                self.generate_and_plot_samples(gen_dataloader,
                                               cfg=cfg,
                                               epoch=epoch)


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
        # Set initial loss to 0
        loss = 0.0
        # Set the progress bar
        pbar = tqdm.tqdm(dataloader, desc=f"Epoch {current_epoch}/{epochs}", unit="batch")

        # Iterate through batches in dataloader (tuple of images and seasons)
        for idx, samples in enumerate(pbar):
            # Samples is a dict with following available keys: 'img', 'classifier', 'img_cond', 'lsm', 'sdf', 'topo', 'points'
            # Extract samples
            x, seasons, cond_images, lsm_hr, lsm, sdf, topo, hr_points, lr_points = extract_samples(samples, self.device)
            # No gradients needed for validation
            with torch.inference_mode(): #torch.no_grad(): # New in PyTorch 1.9, slightly faster than torch.no_grad()
                # Use mixed precision training if needed
                if hasattr(self, 'scaler') and self.scaler:
                    with autocast():
                        # Pass the score model and samples+conditions to the loss_fn
                        batch_loss = self.loss_fn(self.model,
                                             x,
                                             self.marginal_prob_std_fn,
                                             y=seasons,
                                             cond_img=cond_images,
                                             lsm_cond=lsm,
                                             topo_cond=topo,
                                             sdf_cond=sdf)
                else:
                    # No mixed precision, just pass the score model and samples+conditions to the loss_fn
                    batch_loss = self.loss_fn(self.model,
                                         x,
                                         self.marginal_prob_std_fn,
                                         y=seasons,
                                         cond_img=cond_images,
                                         lsm_cond=lsm,
                                         topo_cond=topo,
                                         sdf_cond=sdf)


            # === Extreme-prcp sentinel on ground-truth HR in validation (optional; lightweight) ===
            if self.extreme_enabled and self.extreme_in_validation and (idx % self.extreme_every_step == 0):
                try:
                    # x is in model space; optionally back-transform to physical mm/day
                    x_for_check = x.detach()
                    if self.extreme_backtransform and self.back_transforms_train is not None:
                        # Expect a callable for HR back-transform under key 'hr'
                        hr_back_transform = self.back_transforms_train.get('hr')
                        if hr_back_transform is not None:
                            if callable(hr_back_transform):
                                x_for_check = hr_back_transform(x_for_check)
                            else:
                                logger.warning(f"[monitor] Back-transform object for HR is not callable and has no 'transform' method.")
                    # Run helper (accepts torch or numpy)
                    # Ensure x_for_check is a torch.Tensor before passing to report_precip_extremes
                    if not isinstance(x_for_check, torch.Tensor):
                        x_for_check = torch.tensor(x_for_check)
                    check = report_precip_extremes(x_bt=x_for_check.detach().cpu(), name="ground_truth_hr", cap_mm_day=self.extreme_threshold_mm)
                    # "check" is boolean: True if any extreme values found
                    has_extreme = check.get('has_extreme', False)
                    n_extreme = check.get('n_extreme', 0)
                    extreme_values = check.get('extreme_values', [])
                    has_below_zero = check.get('has_below_zero', False)
                    n_below_zero = check.get('n_below_zero', 0)
                    below_zero_values = check.get('below_zero_values', [])
                    if has_extreme and self.extreme_log_first_n > 0:
                        # Extract some stats if provided
                        mx = max(extreme_values) if isinstance(extreme_values, list) else None
                        cnt = len(extreme_values) if isinstance(extreme_values, list) else None
                        logger.warning(f"[monitor][val] Extreme precipitation detected at step {idx}:")
                        logger.warning(f"               max={mx:.1f} mm/day, count={cnt}, threshold={self.extreme_threshold_mm} mm/day")
                        self.extreme_log_first_n -= 1  # Decrement counter to log fewer next times
                except Exception as e:
                    logger.warning(f"[monitor] Could not check for extreme precipitation in validation step {idx}. Error: {e}")

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
        
        # Load the model from checkpoint_dir with name checkpoint_name
        best_model_state = torch.load(self.checkpoint_path, map_location=self.device)['network_params']
        self.model.load_state_dict(best_model_state)
        # Set model to evaluation mode (set back to training mode after sampling)
        self.model.eval()

        # Set up sampler 
        if cfg['sampler']['sampler_type'] == 'pc_sampler':
            sampler = pc_sampler 
        elif cfg['sampler']['sampler_type'] == 'Euler_Maruyama_sampler':
            sampler = Euler_Maruyama_sampler
        elif cfg['sampler']['sampler_type'] == 'ode_sampler':
            sampler = ode_sampler
        else:
            raise ValueError(f"Sampler type {cfg['sampler']['sampler_type']} not recognized. Please choose from 'pc_sampler', 'Euler_Maruyama_sampler', or 'ode_sampler'.")
        
        # # Set up back transforms for plotting'
        # back_trans = build_back_transforms(
        #             hr_var=cfg['highres']['variable'],
        #             hr_scaling_method=cfg['highres']['scaling_method'],
        #             hr_scaling_params=cfg['highres']['scaling_params'],
        #             lr_vars=cfg['lowres']['condition_variables'],
        #             lr_scaling_methods=cfg['lowres']['scaling_methods'],
        #             lr_scaling_params=cfg['lowres']['scaling_params'],
        #         )
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
                            split               = 'all',
                            stats_dir_root      = cfg['paths']['stats_load_dir']
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

            # Check shapes of x_gen, cond_images_gen, lsm_gen, topo_gen
            # logger.info(f"\nShape of x_gen: {x_gen.shape}")
            # logger.info(f"Shape of seasons_gen: {seasons_gen.shape}")
            # logger.info(f"Shape of cond_images_gen: {cond_images_gen.shape if cond_images_gen is not None else 'None'}")
            # logger.info(f"Shape of lsm_gen: {lsm_gen.shape if lsm_gen is not None else 'None'}")
            # logger.info(f"Shape of topo_gen: {topo_gen.shape if topo_gen is not None else 'None'}")

            generated_samples = sampler(
                score_model = self.model,
                marginal_prob_std = self.marginal_prob_std_fn,
                diffusion_coeff = self.diffusion_coeff_fn,
                batch_size= cfg['data_handling']['n_gen_samples'],
                num_steps = cfg['sampler']['n_timesteps'],
                device = self.device,
                img_size = cfg['highres']['data_size'][0],
                y = seasons_gen,
                cond_img= cond_images_gen,
                lsm_cond = lsm_gen,
                topo_cond = topo_gen,
            )
            generated_samples = generated_samples.squeeze().detach().cpu()



            # === Back-transform generated samples for sentinel (and optional clamp) ===
            try:
                # Reuse back-transforms built earlier
                if isinstance(generated_samples, torch.Tensor):
                    gen_for_check = generated_samples
                else:
                    gen_for_check = torch.as_tensor(generated_samples)

                gen_bt = None

                if cfg.get("monitoring", {}).get("extreme_prcp", {}).get("back_transform", True):
                    bt = back_transforms.get('hr', None)
                    if bt is not None:
                        if callable(bt):
                            gen_bt = bt(gen_for_check)
                        else:
                            logger.warning(f"[monitor] Back-transform object for HR is not callable and has no 'transform' method.")
                    else:
                        gen_bt = gen_for_check
                else:
                    gen_bt = gen_for_check

                # Run sentinel
                mon_cfg = cfg.get('monitoring', {}).get('extreme_prcp', {})
                thr = float(mon_cfg.get('threshold_mm', self.extreme_threshold_mm))
                # Ensure gen_bt is a torch.Tensor before calling detach
                if gen_bt is None:
                    gen_bt = torch.zeros_like(generated_samples)
                if not isinstance(gen_bt, torch.Tensor):
                    gen_bt = torch.as_tensor(gen_bt)
                chk = report_precip_extremes(gen_bt.detach().cpu(), name="generated_hr", cap_mm_day=thr)
                has_extreme = chk.get('has_extreme', False)
                n_extreme = chk.get('n_extreme', 0)
                extreme_values = chk.get('extreme_values', [])
                has_below_zero = chk.get('has_below_zero', False)
                n_below_zero = chk.get('n_below_zero', 0)
                below_zero_values = chk.get('below_zero_values', [])

                if has_extreme:
                    mx = max(extreme_values) if isinstance(extreme_values, list) else None
                    cnt = len(extreme_values) if isinstance(extreme_values, list) else None
                    logger.warning(f"[monitor][gen] Extreme precipitation detected in generated samples:")
                    logger.warning(f"               max={mx:.1f} mm/day, count={cnt}, threshold={thr} mm/day")

                    # Clamp extreme values in generated samples if configured
                    if mon_cfg.get('clamp_in_generation', self.extreme_clamp_in_gen):
                        # Clamp in gen_bt space first
                        clamp_max = float(mon_cfg.get('clamp_max_mm', thr))
                        if not isinstance(gen_bt, torch.Tensor):
                            gen_bt = torch.as_tensor(gen_bt)
                        gen_bt = torch.clamp(gen_bt, min=0.0, max=clamp_max)
                        logger.warning(f"[monitor][gen] Clamped generated samples to max {clamp_max} mm/day.")
                        # Replace array that will be plotted with clamped values
                        generated_samples = gen_bt
            except Exception as e:
                logger.warning(f"[monitor] Could not check for extreme precipitation in generated samples. Error: {e}")



            # Plot generated and original samples
            if cfg['visualization']['create_figs']:
                fig, _ = plot_samples_and_generated(
                    samples=samples,
                    generated=generated_samples,
                    cfg=cfg,
                    transform_back_bf_plot=cfg['visualization']['transform_back_bf_plot'],
                    back_transforms=back_transforms,
                )

                if cfg['visualization']['save_figs']:
                    # Save the figure
                    fig.savefig(os.path.join(self.path_figures, f'epoch_{epoch}_generatedSamples.png'), dpi=300, bbox_inches='tight')
                    logger.info(f"→ Figure saved to {os.path.join(self.path_figures, f'epoch_{epoch}_generatedSamples.png')}")

                if cfg['visualization']['show_figs']:
                    # Show the figure
                    plt.show()
                else:
                    # Close the figure
                    plt.close(fig)

                # Stop after the first batch, as we only want to generate samples once per epoch
                break
                    

        # Set model back to training mode
        self.model.train()

    def plot_losses(self,
                    train_losses,
                    val_losses=None,
                    save_path=None,
                    save_name='losses_plot.png',
                    show_plot=False):
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
        fig, ax = plt.subplots()
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
            logger.info(f"→ Losses plot saved to {os.path.join(save_path, save_name)}")
        
        plt.close(fig)




        
import os
import torch
import copy
import pickle
import tqdm
import logging 

import torch.nn as nn
# import matplotlib.pyplot as plt

from torch.cuda.amp import autocast, GradScaler

from sbgm.special_transforms import build_back_transforms
from sbgm.utils import *
from sbgm.data_modules import *
from sbgm.score_unet import loss_fn, marginal_prob_std_fn, diffusion_coeff_fn
from sbgm.score_sampling import Euler_Maruyama_sampler, pc_sampler, ode_sampler
from sbgm.training_utils import get_model_string, get_cmaps, get_units

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
                 loss_fn,
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

        # Set class variables
        self.model = model
        self.loss_fn = loss_fn
        self.marginal_prob_std_fn = marginal_prob_std_fn
        self.diffusion_coeff_fn = diffusion_coeff_fn
        self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler

        self.scaling = cfg['transforms']['scaling']
        self.hr_var = cfg['highres']['variable']
        self.hr_scaling_method = cfg['highres']['scaling_method']
        self.hr_scaling_params = cfg['highres']['scaling_params']
        self.lr_vars = cfg['lowres']['condition_variables']
        self.lr_scaling_methods = cfg['lowres']['scaling_methods']
        self.lr_scaling_params = cfg['lowres']['scaling_params']
        
        self.weight_init = cfg['training']['weight_init']
        self.custom_weight_initializer = cfg['training']['custom_weight_initializer']
        self.sdf_weighted_loss = cfg['training']['sdf_weighted_loss']
        self.with_ema = cfg['training']['with_ema']
        # Store the full configuration for later use
        self.cfg = cfg

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
        self.checkpoint_dir = cfg['paths']['checkpoint_dir']
        self.checkpoint_name = cfg['paths']['checkpoint_name']
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
            batch_loss = loss_fn(self.model,
                        x,
                        self.marginal_prob_std_fn,
                        y = seasons,
                        cond_img = cond_images,
                        lsm_cond = lsm,
                        topo_cond = topo,
                        sdf_cond = sdf)
            # logger.info(f"▸ Batch loss computed: {batch_loss.item():.4f}")
            # Add anomaly detection for loss
            with torch.autograd.detect_anomaly(True):
                # Backward pass
                batch_loss.backward()
            # Update weights
            self.optimizer.step()

            # Add batch loss to total loss
            loss_sum += batch_loss.item()
            # Update the bar
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
            with torch.no_grad():
                # Use mixed precision training if needed
                if hasattr(self, 'scaler') and self.scaler:
                    with autocast():
                        # Pass the score model and samples+conditions to the loss_fn
                        batch_loss = loss_fn(self.model,
                                             x,
                                             self.marginal_prob_std_fn,
                                             y=seasons,
                                             cond_img=cond_images,
                                             lsm_cond=lsm,
                                             topo_cond=topo,
                                             sdf_cond=sdf)
                else:
                    # No mixed precision, just pass the score model and samples+conditions to the loss_fn
                    batch_loss = loss_fn(self.model,
                                         x,
                                         self.marginal_prob_std_fn,
                                         y=seasons,
                                         cond_img=cond_images,
                                         lsm_cond=lsm,
                                         topo_cond=topo,
                                         sdf_cond=sdf)

            # Add batch loss to total loss
            loss += batch_loss.item()
            # Update the bar

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
        
        # Set up back transforms for plotting'
        back_trans = build_back_transforms(
                    hr_var=cfg['highres']['variable'],
                    hr_scaling_method=cfg['highres']['scaling_method'],
                    hr_scaling_params=cfg['highres']['scaling_params'],
                    lr_vars=cfg['lowres']['condition_variables'],
                    lr_scaling_methods=cfg['lowres']['scaling_methods'],
                    lr_scaling_params=cfg['lowres']['scaling_params'],
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


            # Plot generated and original samples
            if cfg['visualization']['create_figs']:
                fig, axs = plot_samples_and_generated(
                    samples=samples,
                    generated=generated_samples,
                    hr_model=cfg['highres']['model'],
                    hr_units=hr_unit,
                    lr_model=cfg['lowres']['model'],
                    lr_units=lr_units,
                    var=cfg['highres']['variable'],
                    scaling=cfg['transforms']['scaling'],
                    show_ocean=cfg['visualization']['show_ocean'],
                    transform_back_bf_plot=cfg['visualization']['transform_back_bf_plot'],
                    back_transforms=back_trans,
                    hr_cmap=hr_cmap_name,
                    lr_cmap_dict= lr_cmap_dict,
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




        
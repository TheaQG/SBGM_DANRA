import os, torch
import copy
import pickle

import torch.nn as nn
# import matplotlib.pyplot as plt

from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

from sbgm.special_transforms import build_back_transforms
from sbgm.utils import *
from sbgm.data_modules import *
from sbgm.score_unet import loss_fn
from sbgm.score_sampling import *

'''
    ToDo:
        - Add support for mixed precision training
        - Add support for EMA (Exponential Moving Average) of the model
        - Add support for custom weight initialization
'''



class TrainingPipeline_general:
    '''
        Class for building a training pipeline for the SBGM.
        To run through the training batches in one epoch.
    '''

    def __init__(self,
                 model,
                 loss_fn,
                 marginal_prob_std_fn,
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
        self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler

        self.scaling = cfg['transforms']['scaling']
        self.hr_var = cfg['highres']['var']
        self.hr_scaling_method = cfg['highres']['scaling_method']
        self.hr_scaling_params = cfg['highres']['scaling_params']
        self.lr_vars = cfg['lowres']['condition_variables']
        self.lr_scaling_methods = cfg['lowres']['scaling_methods']
        self.lr_scaling_params = cfg['lowres']['scaling_params']
        
        self.weight_init = cfg['training']['weight_init']
        self.custom_weight_initializer = cfg['training']['custom_weight_initializer']
        self.sdf_weighted_loss = cfg['training']['sdf_weighted_loss']
        self.with_ema = cfg['training']['with_ema']

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Initialize weights if needed
        if self.weight_init:
            if self.custom_weight_initializer is not None:
                # Use custom weight initializer if provided
                self.model.apply(self.custom_weight_initializer)
            else:
                self.model.apply(self.xavier_init_weights)
            print(f"→ Model weights initialized with {self.custom_weight_initializer.__name__ if self.custom_weight_initializer else 'Xavier uniform'} initialization.")

        # Set Exponential Moving Average (EMA) if needed
        if self.with_ema:
            #!!!!! NOTE: EMA is not implemented yet, this is a placeholder for future implementation"
            # Create a copy of the model for EMA
            self.ema_model = copy.deepcopy(self.model)
            # Detach the EMA model parameters to not update them
            for param in self.ema_model.parameters():
                param.detach_()

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
            if torch.is_tensor(m.bias):
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
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Use mixed precision training if needed
            if self.scaler:
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
                # Mixed precision: scale loss and update weights
                self.scaler.scale(batch_loss).backward()
                # Update weights
                self.scaler.step(self.optimizer)
                # Update scaler
                self.scaler.update()
            else:
                batch_loss = loss_fn(self.model,
                            x,
                            self.marginal_prob_std_fn,
                            y = seasons,
                            cond_img = cond_images,
                            lsm_cond = lsm,
                            topo_cond = topo,
                            sdf_cond = sdf)
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
            print(f"→ Epoch {getattr(self, 'epoch', '?')} completed: Avg. training Loss: {avg_loss:.4f}")

        return avg_loss
    
    def train(self,
              train_dataloader,
              val_dataloader,
              cfg,
              epochs=1,
              verbose=True,
              PLOT_FIRST=False,
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
        best_loss = float('inf')

        print('\n\n\nStarting training...\n\n\n')

        # Iterate through epochs
        for epoch in range(1, epochs + 1):
            # Set epoch attribute
            self.epoch = epoch 
            # Print epoch number if verbose
            if verbose:
                print(f"▸ Starting epoch {epoch}/{epochs}...")

            # Train on batches
            train_loss = self.train_batches(train_dataloader,
                                            epochs=epochs,
                                            current_epoch=epoch,
                                            verbose=verbose,
                                            PLOT_FIRST=PLOT_FIRST,
                                            SAVE_PATH=SAVE_PATH,
                                            SAVE_NAME=SAVE_NAME,
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
                self.save_model(dirname=checkpoint_dir, filename=checkpoint_name)
                print(f"→ Best model saved with validation loss: {best_loss:.4f} at epoch {epoch}.")
                print(f"→ Checkpoint saved to {os.path.join(checkpoint_dir, checkpoint_name)}\n\n")


            # Pickle dump the losses
            losses = {
                'train_losses': train_losses,
                'val_losses': val_losses if val_dataloader is not None else None
            }
            with open(os.path.join(SAVE_PATH, 'losses.pkl'), 'wb') as f:
                pickle.dump(losses, f)

            # Generate and save samples, if create_figs is True
            if cfg['visualization']['create_figs'] and cfg['data_handling']['n_gen_samples'] > 0:
                self.generate_and_save_samples(cfg, epoch)


        return train_loss, val_loss if val_dataloader is not None else None

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
            print(f'→ Validation Loss: {avg_loss:.4f}')

        return avg_loss
    
    def generate_and_save_samples(self,
                            cfg,
                            epoch,
                            samples
                          ):
        
        # Set path to figures, samples, losses
        path_samples = cfg['paths']['path_save'] + 'samples' + f'/Samples' + '__' + save_str
        path_losses = cfg['paths']['path_save'] + '/losses'
        path_figures = path_samples + '/Figures/'
        
        if not os.path.exists(path_samples):
            os.makedirs(path_samples)
        if not os.path.exists(path_losses):
            os.makedirs(path_losses)
        if not os.path.exists(path_figures):
            os.makedirs(path_figures)

        name_samples = 'Generated_samples' + '__' + 'epoch' + '_'
        name_final_samples = f'Final_generated_sample'
        name_losses = f'Training_losses'

        # Load the model from checkpoint_dir with name checkpoint_name
        best_model_path = os.path.join(checkpoint_dir, checkpoint_name)
        best_model_state = torch.load(best_model_path, map_location=self.device)['network_params']
        self.model.load_state_dict(best_model_state)
        # Set model to evaluation mode (set back to training mode after sampling)
        self.model.eval()

        # Setup progress bar
        pbar = tqdm.tqdm(range(cfg['data_handling']['n_gen_samples']),
                         desc=f"Generating samples for epoch {epoch}",
                         unit="sample")
        
        # Iterate through the number of samples to generate
        for i, samples in enumerate(pbar):
            # Generate samples
            generated_samples = generate_samples(self.model,
                                                 cfg['data_handling']['n_gen_samples'],
                                                 cfg['data_handling']['batch_size'],
                                                 cfg['data_handling']['sample_size'],
                                                 self.marginal_prob_std_fn,
                                                 device=self.device,
                                                 cfg=cfg)

            # Save the generated samples
            save_samples(generated_samples, path_samples, name_samples + str(i + 1) + '.pkl')






        



import os
import torch 
import zarr

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

from sbgm.data_modules import DANRA_Dataset_cutouts_ERA5_Zarr
from sbgm.score_unet import ScoreNet, Encoder, Decoder, DecoderBlock, marginal_prob_std_fn, diffusion_coeff_fn
from sbgm.score_sampling import pc_sampler, Euler_Maruyama_sampler, ode_sampler
from sbgm.training import TrainingPipeline_general
from sbgm.utils import build_data_path
from sbgm.special_transforms import build_back_transforms
# from sbgm.evaluation.evaluation import evaluate_model


def get_dataloader(cfg, verbose=True):
    '''
        Get the dataloader for training and validation datasets based on the configuration.
        Args:
            cfg (dict): Configuration dictionary containing data settings.
            verbose (bool): If True, print detailed information about the data types and sizes.
        Returns:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            gen_loader (DataLoader): DataLoader for the generation dataset.
    '''
    # Print information about data types
    hr_unit, lr_units = get_units(cfg)
    print(f"\nUsing HR data type: {cfg['highres']['model']} {cfg['highres']['variable']} [{hr_unit}]")

    for i, cond in enumerate(cfg['lowres']['condition_variables']):
        print(f"Using LR data type {i+1}: {cfg['lowres']['model']} {cond} [{lr_units[i]}]")

    # Set image dimensions based on config (if None, use default values)
    hr_data_size = tuple(cfg['highres']['data_size']) if cfg['highres']['data_size'] is not None else None
    if hr_data_size is None:
        hr_data_size = (128, 128)

    lr_data_size = tuple(cfg['lowres']['data_size']) if cfg['lowres']['data_size'] is not None else None    
    if lr_data_size is None:
        lr_data_size_use = hr_data_size
    else:
        lr_data_size_use = lr_data_size

    # Check if resize factor is set and print sizes (if verbose)
    if cfg['lowres']['resize_factor'] > 1:
        hr_data_size_use = (hr_data_size[0] // cfg['lowres']['resize_factor'], hr_data_size[1] // cfg['lowres']['resize_factor'])
        lr_data_size_use = (lr_data_size_use[0] // cfg['lowres']['resize_factor'], lr_data_size_use[1] // cfg['lowres']['resize_factor'])
    else:
        hr_data_size_use = hr_data_size
        lr_data_size_use = lr_data_size_use
    if verbose:
        print(f"\n\nHigh-resolution data size: {hr_data_size_use}")
        if cfg['lowres']['resize_factor'] > 1:
            print(f"\tHigh-resolution data size after resize: {hr_data_size_use}")
        print(f"Low-resolution data size: {lr_data_size_use}")
        if cfg['lowres']['resize_factor'] > 1:
            print(f"\tLow-resolution data size after resize: {lr_data_size_use}")

    # Set full domain size 
    full_domain_dims = tuple(cfg['highres']['full_domain_size']) if cfg['highres']['full_domain_size'] is not None else None


    # Use helper functions to create the path for the zarr files
    hr_data_dir_train = build_data_path(cfg['paths']['data_dir'], cfg['highres']['model'], cfg['highres']['variable'], full_domain_dims, 'train')
    hr_data_dir_valid = build_data_path(cfg['paths']['data_dir'], cfg['highres']['model'], cfg['highres']['variable'], full_domain_dims, 'val')
    hr_data_dir_gen = build_data_path(cfg['paths']['data_dir'], cfg['highres']['model'], cfg['highres']['variable'], full_domain_dims, 'gen')
    
    # Loop over lr_vars and create paths for low-resolution data
    lr_cond_dirs_train = {}
    lr_cond_dirs_valid = {}
    lr_cond_dirs_gen = {}

    for i, cond in enumerate(cfg['lowres']['condition_variables']):
        lr_cond_dirs_train[cond] = build_data_path(cfg['paths']['data_dir'], cfg['lowres']['model'], cond, full_domain_dims, 'train')
        lr_cond_dirs_valid[cond] = build_data_path(cfg['paths']['data_dir'], cfg['lowres']['model'], cond, full_domain_dims, 'val')
        lr_cond_dirs_gen[cond] = build_data_path(cfg['paths']['data_dir'], cfg['lowres']['model'], cond, full_domain_dims, 'gen')

    # # Set scaling and matching 
    # scaling = cfg['transforms']['scaling']
    # force_matching_scale = cfg['transforms']['force_matching_scale']
    # show_both_orig_scaled = cfg['visualization']['show_both_orig_scaled']
    # transform_back_bf_plot = cfg['visualization']['transform_back_bf_plot']

    # # Set up scaling methods
    # hr_scaling_method = cfg['highres']['scaling_method']
    # hr_scaling_params = cfg['highres']['scaling_params']
    # lr_scaling_methods = cfg['lowres']['scaling_methods']
    # lr_scaling_params = cfg['lowres']['scaling_params']

    # Set up back transformations (for plotting and visual inspection + later evaluation)
    back_transforms = build_back_transforms(
        hr_var              = cfg['highres']['variable'],
        hr_scaling_method   = cfg['highres']['scaling_method'],
        hr_scaling_params   = cfg['highres']['scaling_params'],
        lr_vars             = cfg['lowres']['condition_variables'],
        lr_scaling_methods  = cfg['lowres']['scaling_methods'],
        lr_scaling_params   = cfg['lowres']['scaling_params']
    )

    if cfg['stationary_conditions']['geographic_conditions']['sample_w_sdf']:
        print('\nSDF weighted loss enabled. Setting lsm and topo to true.\n')
        sample_w_geo = True
    else:
        sample_w_geo = cfg['stationary_conditions']['geographic_conditions']['sample_w_geo']

    if sample_w_geo:
        print('\nUsing geographical features for sampling.\n')
        
        geo_variables = cfg['stationary_conditions']['geographic_conditions']['geo_variables']
        data_dir_lsm = cfg['paths']['lsm_path']
        data_dir_topo = cfg['paths']['topo_path']

        data_lsm = np.flipud(np.load(data_dir_lsm)['data'])
        data_topo = np.flipud(np.load(data_dir_topo)['data'])

        if cfg['transforms']['scaling']:
            if cfg['stationary_conditions']['geographic_conditions']['topo_min'] is None or cfg['stationary_conditions']['geographic_conditions']['topo_max'] is None:
                topo_min, topo_max = np.min(data_topo), np.max(data_topo)
            else:
                topo_min = cfg['stationary_conditions']['geographic_conditions']['topo_min']
                topo_max = cfg['stationary_conditions']['geographic_conditions']['topo_max']
            if cfg['stationary_conditions']['geographic_conditions']['norm_min'] is None or cfg['stationary_conditions']['geographic_conditions']['norm_max'] is None:
                norm_min, norm_max = np.min(data_lsm), np.max(data_lsm)
            else:
                norm_min = cfg['stationary_conditions']['geographic_conditions']['norm_min']
                norm_max = cfg['stationary_conditions']['geographic_conditions']['norm_max']
            OldRange = (topo_max - topo_min)
            NewRange = (norm_max - norm_min)
            data_topo = ((data_topo - topo_min) * NewRange / OldRange) + norm_min
    else: 
        geo_variables = None
        data_lsm = None
        data_topo = None

    # Setup cutouts. If cutout domains None, use default (170, 350, 340, 520) (DK area with room for shuffle)
    cutout_domains = tuple(cfg['highres']['cutout_domains']) if cfg['highres']['cutout_domains'] is not None else (170, 350, 340, 520)
    lr_cutout_domains = tuple(cfg['lowres']['cutout_domains']) if cfg['lowres']['cutout_domains'] is not None else (170, 350, 340, 520)

    # Setup conditional seasons (classification)
    if cfg['stationary_conditions']['seasonal_conditions']['sample_w_cond_season']:
        n_seasons = cfg['stationary_conditions']['seasonal_conditions']['n_seasons']
    else:
        n_seasons = None


    # Make zarr groups
    data_train_zarr = zarr.open_group(hr_data_dir_train, mode='r')
    data_valid_zarr = zarr.open_group(hr_data_dir_valid, mode='r')
    data_gen_zarr = zarr.open_group(hr_data_dir_gen, mode='r')

    n_samples_train = len(list(data_train_zarr.keys()))
    n_samples_valid = len(list(data_valid_zarr.keys()))
    n_samples_gen = len(list(data_gen_zarr.keys()))

    # Setup cache
    if cfg['data_handling']['cache_size'] == 0:
        cache_size_train = n_samples_train//2
        cache_size_valid = n_samples_valid//2
    else:
        cache_size_train = cfg['data_handling']['cache_size']
        cache_size_valid = cfg['data_handling']['cache_size']

    if verbose:
        print(f"\n\n\nNumber of training samples: {n_samples_train}")
        print(f"Number of validation samples: {n_samples_valid}")
        print(f"Cache size for training: {cache_size_train}")
        print(f"Cache size for validation: {cache_size_valid}\n\n\n")


    # Setup datasets

    train_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(
                            hr_variable_dir_zarr=hr_data_dir_train,
                            hr_data_size=hr_data_size_use,
                            n_samples=n_samples_train,
                            cache_size=cache_size_train,
                            hr_variable=cfg['highres']['variable'],
                            hr_model=cfg['highres']['model'],
                            hr_scaling_method=cfg['highres']['scaling_method'],
                            hr_scaling_params=cfg['highres']['scaling_params'],
                            lr_conditions=cfg['lowres']['condition_variables'],
                            lr_model=cfg['lowres']['model'],
                            lr_scaling_methods=cfg['lowres']['scaling_methods'],
                            lr_scaling_params=cfg['lowres']['scaling_params'],
                            lr_cond_dirs_zarr=lr_cond_dirs_train,
                            geo_variables=geo_variables,
                            lsm_full_domain=data_lsm,
                            topo_full_domain=data_topo,
                            shuffle=True,
                            cutouts=cfg['transforms']['sample_w_cutouts'],
                            cutout_domains=list(cutout_domains) if cfg['transforms']['sample_w_cutouts'] else None,
                            n_samples_w_cutouts=n_samples_train,
                            sdf_weighted_loss=cfg['stationary_conditions']['geographic_conditions']['sample_w_sdf'],
                            scale=cfg['transforms']['scaling'],
                            save_original=cfg['visualization']['show_both_orig_scaled'],
                            conditional_seasons=cfg['stationary_conditions']['seasonal_conditions']['sample_w_cond_season'],
                            n_classes=n_seasons,
                            lr_data_size=tuple(lr_data_size_use) if lr_data_size_use is not None else None,
                            lr_cutout_domains=list(lr_cutout_domains) if lr_cutout_domains is not None else None,
                            resize_factor=cfg['lowres']['resize_factor'],
    )

    val_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(
                            hr_variable_dir_zarr=hr_data_dir_valid,
                            hr_data_size=hr_data_size_use,
                            n_samples=n_samples_valid,
                            cache_size=cache_size_valid,
                            hr_variable=cfg['highres']['variable'],
                            hr_model=cfg['highres']['model'],
                            hr_scaling_method=cfg['highres']['scaling_method'],
                            hr_scaling_params=cfg['highres']['scaling_params'],
                            lr_conditions=cfg['lowres']['condition_variables'],
                            lr_model=cfg['lowres']['model'],
                            lr_scaling_methods=cfg['lowres']['scaling_methods'],
                            lr_scaling_params=cfg['lowres']['scaling_params'],
                            lr_cond_dirs_zarr=lr_cond_dirs_valid,
                            geo_variables=geo_variables,
                            lsm_full_domain=data_lsm,
                            topo_full_domain=data_topo,
                            shuffle=True,
                            cutouts=cfg['transforms']['sample_w_cutouts'],
                            cutout_domains=list(cutout_domains) if cfg['transforms']['sample_w_cutouts'] else None,
                            n_samples_w_cutouts=n_samples_valid,
                            sdf_weighted_loss=cfg['stationary_conditions']['geographic_conditions']['sample_w_sdf'],
                            scale=cfg['transforms']['scaling'],
                            save_original=cfg['visualization']['show_both_orig_scaled'],
                            conditional_seasons=cfg['stationary_conditions']['seasonal_conditions']['sample_w_cond_season'],
                            n_classes=n_seasons,
                            lr_data_size=tuple(lr_data_size_use) if lr_data_size_use is not None else None,
                            lr_cutout_domains=list(lr_cutout_domains) if lr_cutout_domains is not None else None,
                            resize_factor=cfg['lowres']['resize_factor'],
    )

    gen_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(
                            hr_variable_dir_zarr=hr_data_dir_gen,
                            hr_data_size=hr_data_size_use,
                            n_samples=n_samples_gen,
                            cache_size=cfg['data_handling']['cache_size'],
                            hr_variable=cfg['highres']['variable'],
                            hr_model=cfg['highres']['model'],
                            hr_scaling_method=cfg['highres']['scaling_method'],
                            hr_scaling_params=cfg['highres']['scaling_params'],
                            lr_conditions=cfg['lowres']['condition_variables'],
                            lr_model=cfg['lowres']['model'],
                            lr_scaling_methods=cfg['lowres']['scaling_methods'],
                            lr_scaling_params=cfg['lowres']['scaling_params'],
                            lr_cond_dirs_zarr=lr_cond_dirs_gen,
                            geo_variables=geo_variables,
                            lsm_full_domain=data_lsm,
                            topo_full_domain=data_topo,
                            shuffle=True,
                            cutouts=cfg['transforms']['sample_w_cutouts'],
                            cutout_domains=list(cutout_domains) if cfg['transforms']['sample_w_cutouts'] else None,
                            n_samples_w_cutouts=n_samples_gen,
                            sdf_weighted_loss=cfg['stationary_conditions']['geographic_conditions']['sample_w_sdf'],
                            scale=cfg['transforms']['scaling'],
                            save_original=cfg['visualization']['show_both_orig_scaled'],
                            conditional_seasons=cfg['stationary_conditions']['seasonal_conditions']['sample_w_cond_season'],
                            n_classes=n_seasons,
                            lr_data_size=tuple(lr_data_size_use) if lr_data_size_use is not None else None,
                            lr_cutout_domains=list(lr_cutout_domains) if lr_cutout_domains is not None else None,
                            resize_factor=cfg['lowres']['resize_factor'],
                            )
    # Setup dataloaders
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg['data_handling']['batch_size'],
                              shuffle=True,
                            # num_workers=cfg['data_handling']['num_workers'],
    )
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg['data_handling']['batch_size'],
                            shuffle=False,
                            # num_workers=cfg['data_handling']['num_workers'],
    )
    gen_loader = DataLoader(gen_dataset,
                            batch_size=cfg['data_handling']['batch_size'],
                            shuffle=False,
                            # num_workers=cfg['data_handling']['num_workers'],
    )

    # Print dataset information
    if verbose:
        print(f"\nTraining dataset: {len(train_dataset)} samples")
        print(f"Validation dataset: {len(val_dataset)} samples")
        print(f"Generation dataset: {len(gen_dataset)} samples\n")
        print(f"Batch size: {cfg['data_handling']['batch_size']}")
        print(f"Number of workers: {cfg['data_handling']['num_workers']}\n")
    
    # Return the dataloaders
    return train_loader, val_loader, gen_loader

def get_model(cfg):
    '''
        Get the model based on the configuration.
        Args:
            cfg (dict): Configuration dictionary containing model settings.
        Returns:
            score_model (ScoreNet): The score model instance.
            checkpoint_path (str): Path to the model checkpoint.
            checkpoint_name (str): Name of the model checkpoint file.
    '''
    
    # Define model parameters
    input_channels = len(cfg['lowres']['condition_variables'])  # 
    output_channels = 1#len(cfg['highres']['variable'])  # Assuming a single output channel for the high-resolution variable
    
    if cfg['lowres']['condition_variables'] is not None:
        sample_w_cond_img = True
    else:
        sample_w_cond_img = False

    # Setup model checkpoint name and path
    save_str = get_model_string(cfg)
    checkpoint_name = save_str + '.pth.tar'

    checkpoint_dir = os.path.join(cfg['paths']['path_save'], cfg['paths']['checkpoint_dir'])

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    # Create the model

    encoder = Encoder(input_channels=input_channels,
                      time_embedding=cfg['sampler']['time_embedding'],
                      cond_on_lsm=cfg['stationary_conditions']['geographic_conditions']['sample_w_geo'],
                      cond_on_topo=cfg['stationary_conditions']['geographic_conditions']['sample_w_geo'],
                      cond_on_img=sample_w_cond_img,
                      block_layers=cfg['sampler']['block_layers'],
                      num_classes=cfg['stationary_conditions']['seasonal_conditions']['n_seasons'] if cfg['stationary_conditions']['seasonal_conditions']['sample_w_cond_season'] else None,
                      n_heads=cfg['sampler']['num_heads'],
                      )
    decoder = Decoder(last_fmap_channels=cfg['sampler']['last_fmap_channels'],
                      output_channels=output_channels,
                      time_embedding=cfg['sampler']['time_embedding'],
                      n_heads=cfg['sampler']['num_heads'],
                      )

    score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn,
                           encoder=encoder,
                           decoder=decoder,
                           )
    
    
    return score_model, checkpoint_path, checkpoint_name

def get_model_string(cfg):
    '''
        Generate a string representation of the model configuration for saving and logging.
        Args:
            cfg (dict): Configuration dictionary containing model settings.
        Returns:
            save_str (str): String representation of the model configuration.
    '''
    # Set image dimensions vased on onfig (if None, use default values)
    hr_data_size = tuple(cfg['highres']['data_size']) if cfg['highres']['data_size'] is not None else None
    if hr_data_size is None:
        hr_data_size = (128, 128)

    lr_data_size = tuple(cfg['lowres']['data_size']) if cfg['lowres']['data_size'] is not None else None    
    if lr_data_size is None:
        lr_data_size_use = hr_data_size
    else:
        lr_data_size_use = lr_data_size

    # Check if resize factor is set and print sizes (if verbose)
    if cfg['lowres']['resize_factor'] > 1:
        hr_data_size_use = (hr_data_size[0] // cfg['lowres']['resize_factor'], hr_data_size[1] // cfg['lowres']['resize_factor'])
        lr_data_size_use = (lr_data_size_use[0] // cfg['lowres']['resize_factor'], lr_data_size_use[1] // cfg['lowres']['resize_factor'])
    else:
        hr_data_size_use = hr_data_size
        lr_data_size_use = lr_data_size_use

    # Setup specific names for saving
    lr_vars_str = '_'.join(cfg['lowres']['condition_variables'])

    save_str = (
        f"{cfg['experiment']['config_name']}__"
        f"HR_{cfg['highres']['variable']}_{cfg['highres']['model']}__"
        f"SIZE_{hr_data_size_use[0]}x{hr_data_size_use[1]}__"
        f"LR_{lr_vars_str}_{cfg['lowres']['model']}__"
        f"LOSS_{cfg['training']['loss_type']}__"
        f"HEADS_{cfg['sampler']['num_heads']}__"
        f"TIMESTEPS_{cfg['sampler']['n_timesteps']}"
    )

    return save_str

def get_optimizer(cfg, model):
    '''
        Get the optimizer based on the configuration.
        Args:
            cfg (dict): Configuration dictionary containing optimizer settings.
            model (torch.nn.Module): The model to optimize.
        Returns:
            optimizer (torch.optim.Optimizer): The optimizer instance.
    '''

    if cfg['training']['optimizer'] == 'adam':
        optimizer = Adam(model.parameters(),
                         lr=cfg['training']['learning_rate'],
                         weight_decay=cfg['training']['weight_decay'])
    elif cfg['training']['optimizer'] == 'sgd':
        optimizer = SGD(model.parameters(),
                        lr=cfg['training']['learning_rate'],
                        momentum=cfg['training']['momentum'],
                        weight_decay=cfg['training']['weight_decay'])
    elif cfg['training']['optimizer'] == 'adamw':
        optimizer = AdamW(model.parameters(),
                          lr=cfg['training']['learning_rate'],
                          weight_decay=cfg['training']['weight_decay'])
    else:
        raise ValueError(f"Optimizer {cfg['training']['optimizer']} not recognized. Use 'adam', 'sgd', or 'adamw'.")
    
    return optimizer

def get_loss(cfg):
    '''
        Get the loss function based on the configuration.
    '''

    
    return

def get_scheduler(cfg, optimizer):
    '''
        Get the learning rate scheduler based on the configuration.
        Args:
            cfg (dict): Configuration dictionary containing scheduler settings.
            optimizer (torch.optim.Optimizer): The optimizer to schedule.
        Returns:
            scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler instance.
    '''
    if cfg['training']['scheduler'] == 'Step':
        scheduler = StepLR(optimizer,
                           step_size=cfg['training']['lr_scheduler_params']['step_size'],
                           gamma=cfg['training']['lr_scheduler_params']['gamma'])
                           
    elif cfg['training']['scheduler'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='min',
                                      factor=cfg['training']['lr_scheduler_params']['factor'],
                                      patience=cfg['training']['lr_scheduler_params']['patience'],
                                      verbose=True)
    elif cfg['training']['scheduler'] == 'CosineAnnealing':
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=cfg['training']['lr_scheduler_params']['T_max'],
                                      eta_min=cfg['training']['lr_scheduler_params']['eta_min'])
    elif cfg['training']['scheduler'] == None:
        scheduler = None
        print("No learning rate scheduler specified. Using the optimizer's default learning rate.")
    else:
        raise ValueError(f"Scheduler {cfg['training']['scheduler']} not recognized. Use 'step', 'reduce_on_plateau', or 'cosine_annealing'.")

    return scheduler



def get_device(verbose=True):
    """
    Get the device to be used for training.
    
    Returns:
        torch.device: The device (CPU or GPU) to be used.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")
    return device
    


def plot_results(train_losses, val_losses, train_scores, val_scores):
    """
    Plot the training and validation losses and scores.
    
    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        train_scores (list): List of training scores.
        val_scores (list): List of validation scores.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot scores
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_scores, label='Train Score')
    plt.plot(epochs, val_scores, label='Validation Score')
    plt.title('Scores')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.show()


def get_units(cfg):
    """
        Get the specifications for plotting samples during training.
        Colors, labels, and other parameters are based on the configuration.
    """

    
    units = {"temp": r"$^\circ$C",
             "prcp": "mm",
             "cape": "J/kg",
             "nwvf": "m/s",
             "ewvf": "m/s",
             "gp200": "hPa",
             "gp500": "hPa",
             "gp850": "hPa",
             "gp1000": "hPa",
             }


    hr_unit = units[cfg['highres']['variable']]
    lr_units = []
    for key in cfg['highres']['variables']:
        if key not in units:
            raise ValueError(f"Variable '{key}' not found in units dictionary.")
        else:
            lr_units.append(units[key])

    return hr_unit, lr_units


def get_cmaps(cfg):
    """
        Get the colormaps for plotting samples during training.
        Colormaps are based on the configuration.
    """
    cmaps = {"temp": "plasma",
             "prcp": "inferno",
             "cape": "viridis",
             "nwvf": "cividis",
             "ewvf": "magma",
             "gp200": "coolwarm",
             "gp500": "coolwarm",
             "gp850": "coolwarm",
             "gp1000": "coolwarm",
             }
    

    hr_cmap = cmaps[cfg['highres']['variable']]
    lr_cmaps = {}
    for key in cfg['highres']['variables']:
        if key not in cmaps:
            raise ValueError(f"Variable '{key}' not found in cmap dictionary.")
        else:
            lr_cmaps[key] = cmaps[key]

    return hr_cmap, lr_cmaps
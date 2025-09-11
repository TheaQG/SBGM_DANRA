import os
import torch
import torch.nn as nn 
import zarr
import logging

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

from sbgm.data_modules import DANRA_Dataset_cutouts_ERA5_Zarr
from sbgm.score_unet import ScoreNet, Encoder, Decoder, EDMPrecondUNet, marginal_prob_std_fn, diffusion_coeff_fn
from sbgm.losses import EDMLoss, DSMLoss
from sbgm.score_sampling import pc_sampler, Euler_Maruyama_sampler, ode_sampler
from sbgm.utils import build_data_path, get_units, get_cmaps, get_model_string
from sbgm.special_transforms import build_back_transforms, build_back_transforms_from_stats
# from sbgm.evaluation.evaluation import evaluate_model

# # Set up logging
logger = logging.getLogger(__name__)

def _get(cfg, path, default=None):
    """
        Safe nested get: path like 'a.b.c'
    """
    node = cfg
    for k in path.split('.'):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node

def get_loss_fn(cfg, marginal_prob_std_fn=None):
    edm_cfg = cfg.get('edm', {})

    if bool(edm_cfg.get('enabled', False)):
        # === EDM branch ===
        P_mean          = float(edm_cfg.get('P_mean', -1.2)) # NVLabs defaults
        P_std           = float(edm_cfg.get('P_std', 1.2))
        sigma_data      = float(edm_cfg.get('sigma_data', 1.0)) # MUST match model preconditioning

        use_sdf         = bool(_get(cfg, 'stationary_conditions.geographic_conditions.sample_w_sdf', False))
        max_land_w      = float(_get(cfg, 'stationary_conditions.geographic_conditions.max_land_weight', 1.0))
        min_sea_w       = float(_get(cfg, 'stationary_conditions.geographic_conditions.min_sea_weight', 0.5))


        return EDMLoss(
                P_mean=P_mean,
                P_std=P_std,
                use_sdf_weight=use_sdf,
                max_land_weight=max_land_w,
                min_sea_weight=min_sea_w)

    # === DSM default branch ===
    ve_cfg = cfg.get('ve_dsm', {})
    t_eps = float(ve_cfg.get('t_eps', 1e-3))
    if marginal_prob_std_fn is None:
        raise ValueError("marginal_prob_std_fn must be provided for VE-DSM loss.")
    
    use_sdf = bool(_get(cfg, 'stationary_conditions.geographic_conditions.sample_w_sdf', True))
    max_land_w = float(_get(cfg, 'stationary_conditions.geographic_conditions.max_land_weight', 1.0))
    min_sea_w = float(_get(cfg, 'stationary_conditions.geographic_conditions.min_sea_weight', 0.5))
    return DSMLoss(
                marginal_prob_std_fn=marginal_prob_std_fn,
                t_eps=t_eps,
                use_sdf_weight=use_sdf,
                max_land_weight=max_land_w,
                min_sea_weight=min_sea_w)

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
    logger.info(f"\nUsing HR data type: {cfg['highres']['model']} {cfg['highres']['variable']} [{hr_unit}]")

    for i, cond in enumerate(cfg['lowres']['condition_variables']):
        logger.info(f"Using LR data type {i+1}: {cfg['lowres']['model']} {cond} [{lr_units[i]}]")

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
        logger.info(f"\n\nHigh-resolution data size: {hr_data_size_use}")
        if cfg['lowres']['resize_factor'] > 1:
            logger.info(f"\tHigh-resolution data size after resize: {hr_data_size_use}")
        logger.info(f"Low-resolution data size: {lr_data_size_use}")
        if cfg['lowres']['resize_factor'] > 1:
            logger.info(f"\tLow-resolution data size after resize: {lr_data_size_use}")

    # Set full domain size 
    full_domain_dims = tuple(cfg['highres']['full_domain_dims']) if cfg['highres']['full_domain_dims'] is not None else None


    # Use helper functions to create the path for the zarr files
    hr_data_dir_train = build_data_path(cfg['paths']['data_dir'], cfg['highres']['model'], cfg['highres']['variable'], full_domain_dims, 'train')
    hr_data_dir_valid = build_data_path(cfg['paths']['data_dir'], cfg['highres']['model'], cfg['highres']['variable'], full_domain_dims, 'valid')
    hr_data_dir_gen = build_data_path(cfg['paths']['data_dir'], cfg['highres']['model'], cfg['highres']['variable'], full_domain_dims, 'test')
    
    # Loop over lr_vars and create paths for low-resolution data
    lr_cond_dirs_train = {}
    lr_cond_dirs_valid = {}
    lr_cond_dirs_gen = {}

    for i, cond in enumerate(cfg['lowres']['condition_variables']):
        lr_cond_dirs_train[cond] = build_data_path(cfg['paths']['data_dir'], cfg['lowres']['model'], cond, full_domain_dims, 'train')
        lr_cond_dirs_valid[cond] = build_data_path(cfg['paths']['data_dir'], cfg['lowres']['model'], cond, full_domain_dims, 'valid')
        lr_cond_dirs_gen[cond] = build_data_path(cfg['paths']['data_dir'], cfg['lowres']['model'], cond, full_domain_dims, 'test')

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
    # back_transforms = build_back_transforms(
    #     hr_var              = cfg['highres']['variable'],
    #     hr_scaling_method   = cfg['highres']['scaling_method'],
    #     hr_scaling_params   = cfg['highres']['scaling_params'],
    #     lr_vars             = cfg['lowres']['condition_variables'],
    #     lr_scaling_methods  = cfg['lowres']['scaling_methods'],
    #     lr_scaling_params   = cfg['lowres']['scaling_params']
    # )
    full_domain_dims_str_hr = f"{full_domain_dims[0]}x{full_domain_dims[1]}" if full_domain_dims is not None else "full_domain"
    full_domain_dims_str_lr = f"{full_domain_dims[0]}x{full_domain_dims[1]}" if full_domain_dims is not None else "full_domain"
    crop_region_hr = cfg['highres']['cutout_domains'] if cfg['highres']['cutout_domains'] is not None else "full_region"
    crop_region_hr_str = '_'.join(map(str, crop_region_hr)) #if isinstance(crop_region_hr, (list, tuple)) else crop_region_hr
    crop_region_lr = cfg['lowres']['cutout_domains'] if cfg['lowres']['cutout_domains'] is not None else "full_region"
    crop_region_lr_str = '_'.join(map(str, crop_region_lr)) #if isinstance(crop_region_lr, (list, tuple)) else crop_region_lr

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

    if cfg['stationary_conditions']['geographic_conditions']['sample_w_sdf']:
        logger.info('SDF weighted loss enabled. Setting lsm and topo to true.\n')
        sample_w_geo = True
    else:
        sample_w_geo = cfg['stationary_conditions']['geographic_conditions']['sample_w_geo']

    if sample_w_geo:
        logger.info('Using geographical features for sampling.\n')
        
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
        logger.info(f"\n\n\nNumber of training samples: {n_samples_train}")
        logger.info(f"Number of validation samples: {n_samples_valid}")
        logger.info(f"Cache size for training: {cache_size_train}")
        logger.info(f"Cache size for validation: {cache_size_valid}\n\n\n")


    # if cfg['data_handling']['cache_size'] == 0:
    #     cache_size_train = n_samples_train//2
    #     cache_size_valid = n_samples_valid//2
    # else:
    #     cache_size_train = cfg['data_handling']['cache_size']
    #     cache_size_valid = cfg['data_handling']['cache_size']

    if verbose:
        logger.info(f"\n\n\nNumber of training samples: {n_samples_train}")
        logger.info(f"Number of validation samples: {n_samples_valid}")
        logger.info(f"Cache size for training: {cache_size_train}")
        logger.info(f"Cache size for validation: {cache_size_valid}\n\n\n")


    # Setup datasets

    train_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(
                            hr_variable_dir_zarr=hr_data_dir_train,
                            hr_data_size=hr_data_size_use,
                            n_samples=n_samples_train,
                            cache_size=cache_size_train,
                            hr_variable=cfg['highres']['variable'],
                            hr_model=cfg['highres']['model'],
                            hr_scaling_method=cfg['highres']['scaling_method'],
                            # hr_scaling_params=cfg['highres']['scaling_params'],
                            lr_conditions=cfg['lowres']['condition_variables'],
                            lr_model=cfg['lowres']['model'],
                            lr_scaling_methods=cfg['lowres']['scaling_methods'],
                            # lr_scaling_params=cfg['lowres']['scaling_params'],
                            lr_cond_dirs_zarr=lr_cond_dirs_train,
                            geo_variables=geo_variables,
                            lsm_full_domain=data_lsm,
                            topo_full_domain=data_topo,
                            cfg = cfg,
                            split = "train",
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
                            # hr_scaling_params=cfg['highres']['scaling_params'],
                            lr_conditions=cfg['lowres']['condition_variables'],
                            lr_model=cfg['lowres']['model'],
                            lr_scaling_methods=cfg['lowres']['scaling_methods'],
                            # lr_scaling_params=cfg['lowres']['scaling_params'],
                            lr_cond_dirs_zarr=lr_cond_dirs_valid,
                            geo_variables=geo_variables,
                            lsm_full_domain=data_lsm,
                            topo_full_domain=data_topo,
                            cfg = cfg,
                            split = "valid",
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
                            # hr_scaling_params=cfg['highres']['scaling_params'],
                            lr_conditions=cfg['lowres']['condition_variables'],
                            lr_model=cfg['lowres']['model'],
                            lr_scaling_methods=cfg['lowres']['scaling_methods'],
                            # lr_scaling_params=cfg['lowres']['scaling_params'],
                            lr_cond_dirs_zarr=lr_cond_dirs_gen,
                            geo_variables=geo_variables,
                            lsm_full_domain=data_lsm,
                            topo_full_domain=data_topo,
                            cfg = cfg,
                            split = "gen",
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
    raw_workers = cfg['data_handling']['num_workers'] 
    try:
        num_workers = int(raw_workers) if raw_workers is not None else 0
    except ValueError:
        # Fallback: treat non-numeric as 0 workers
        num_workers = 0
    logger.info(f"Number of workers set to: {num_workers} (raw input was: {raw_workers})")

    # Check if pin_memory is set in the config, default to False if not
    pin_memory = torch.cuda.is_available() and cfg['data_handling']['pin_memory'] if 'pin_memory' in cfg['data_handling'] else False
    train_loader = DataLoader(
        train_dataset,
        batch_size              = cfg['training']['batch_size'],
        shuffle                 = True,
        num_workers             = int(cfg['data_handling']['num_workers']),#num_workers,
        pin_memory              = torch.cuda.is_available(), #pin_memory,
        persistent_workers      = True, #num_workers > 0, # keeps workers alive between epochs
        prefetch_factor         = 4, # Each worker preloads 4 batches
        drop_last               = True # Better for BatchNorm / GroupNorm

    )
    val_loader = DataLoader(
        val_dataset,
        batch_size              = cfg['training']['batch_size'],
        shuffle                 = False,
        num_workers             = int(cfg['data_handling']['num_workers']),#max(2, num_workers // 4),
        pin_memory              = torch.cuda.is_available(), #pin_memory,
        persistent_workers      = True, #num_workers > 0, # keeps workers alive between epochs
        prefetch_factor         = 2, # Each worker preloads 2 batches
        drop_last               = (len(val_dataset) % cfg['training']['batch_size']) != 0
    )
    gen_loader = DataLoader(
        gen_dataset,
        batch_size              = cfg['data_handling']['n_gen_samples'], # Generation dataset uses a fixed batch size based on n samples to generate
        shuffle                 = False,
        num_workers             = 0, #max(2, num_workers // 4),
        # pin_memory              = pin_memory,
        # persistent_workers      = num_workers > 0,
        drop_last               = (len(gen_dataset) % cfg['training']['batch_size']) != 0,
    )

    # Print dataset information
    # if verbose:
    logger.info(f"\nTraining dataset: {len(train_dataset)} samples")
    logger.info(f"Validation dataset: {len(val_dataset)} samples")
    logger.info(f"Generation dataset: {len(gen_dataset)} samples\n")
    logger.info(f"Batch size: {cfg['training']['batch_size']}")
    logger.info(f"Number of workers: {cfg['data_handling']['num_workers']}\n")
    
    # Return the dataloaders
    return train_loader, val_loader, gen_loader


def get_gen_dataloader(cfg, verbose=True):
    '''
        Get the dataloader for training and validation datasets based on the configuration.
        Args:
            cfg (dict): Configuration dictionary containing data settings.
            verbose (bool): If True, print detailed information about the data types and sizes.
        Returns:
            gen_loader (DataLoader): DataLoader for the generation dataset.
    '''
    # Print information about data types
    hr_unit, lr_units = get_units(cfg)
    logger.info(f"\nUsing HR data type: {cfg['highres']['model']} {cfg['highres']['variable']} [{hr_unit}]")

    for i, cond in enumerate(cfg['lowres']['condition_variables']):
        logger.info(f"Using LR data type {i+1}: {cfg['lowres']['model']} {cond} [{lr_units[i]}]")

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
        logger.info(f"\n\nHigh-resolution data size: {hr_data_size_use}")
        if cfg['lowres']['resize_factor'] > 1:
            logger.info(f"\tHigh-resolution data size after resize: {hr_data_size_use}")
        logger.info(f"Low-resolution data size: {lr_data_size_use}")
        if cfg['lowres']['resize_factor'] > 1:
            logger.info(f"\tLow-resolution data size after resize: {lr_data_size_use}")

    # Set full domain size 
    full_domain_dims = tuple(cfg['highres']['full_domain_dims']) if cfg['highres']['full_domain_dims'] is not None else None


    # Use helper functions to create the path for the zarr files
    hr_data_dir_train = build_data_path(cfg['paths']['data_dir'], cfg['highres']['model'], cfg['highres']['variable'], full_domain_dims, 'train')
    hr_data_dir_valid = build_data_path(cfg['paths']['data_dir'], cfg['highres']['model'], cfg['highres']['variable'], full_domain_dims, 'valid')
    hr_data_dir_gen = build_data_path(cfg['paths']['data_dir'], cfg['highres']['model'], cfg['highres']['variable'], full_domain_dims, 'test')
    
    # Loop over lr_vars and create paths for low-resolution data
    lr_cond_dirs_train = {}
    lr_cond_dirs_valid = {}
    lr_cond_dirs_gen = {}

    for i, cond in enumerate(cfg['lowres']['condition_variables']):
        lr_cond_dirs_train[cond] = build_data_path(cfg['paths']['data_dir'], cfg['lowres']['model'], cond, full_domain_dims, 'train')
        lr_cond_dirs_valid[cond] = build_data_path(cfg['paths']['data_dir'], cfg['lowres']['model'], cond, full_domain_dims, 'valid')
        lr_cond_dirs_gen[cond] = build_data_path(cfg['paths']['data_dir'], cfg['lowres']['model'], cond, full_domain_dims, 'test')

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
    # back_transforms = build_back_transforms(
    #     hr_var              = cfg['highres']['variable'],
    #     hr_scaling_method   = cfg['highres']['scaling_method'],
    #     hr_scaling_params   = cfg['highres']['scaling_params'],
    #     lr_vars             = cfg['lowres']['condition_variables'],
    #     lr_scaling_methods  = cfg['lowres']['scaling_methods'],
    #     lr_scaling_params   = cfg['lowres']['scaling_params']
    # )
    full_domain_dims_str_hr = f"{full_domain_dims[0]}x{full_domain_dims[1]}" if full_domain_dims is not None else "full_domain"
    full_domain_dims_str_lr = f"{full_domain_dims[0]}x{full_domain_dims[1]}" if full_domain_dims is not None else "full_domain"
    crop_region_hr = cfg['highres']['cutout_domains'] if cfg['highres']['cutout_domains'] is not None else "full_region"
    crop_region_hr_str = '_'.join(map(str, crop_region_hr)) #if isinstance(crop_region_hr, (list, tuple)) else crop_region_hr
    crop_region_lr = cfg['lowres']['cutout_domains'] if cfg['lowres']['cutout_domains'] is not None else "full_region"
    crop_region_lr_str = '_'.join(map(str, crop_region_lr)) #if isinstance(crop_region_lr, (list, tuple)) else crop_region_lr

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

    if cfg['stationary_conditions']['geographic_conditions']['sample_w_sdf']:
        logger.info('SDF weighted loss enabled. Setting lsm and topo to true.\n')
        sample_w_geo = True
    else:
        sample_w_geo = cfg['stationary_conditions']['geographic_conditions']['sample_w_geo']

    if sample_w_geo:
        logger.info('Using geographical features for sampling.\n')
        
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
    data_gen_zarr = zarr.open_group(hr_data_dir_gen, mode='r')

    n_samples_gen = len(list(data_gen_zarr.keys()))

    # Setup dataset
    gen_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(
                            hr_variable_dir_zarr=hr_data_dir_gen,
                            hr_data_size=hr_data_size_use,
                            n_samples=n_samples_gen,
                            cache_size=cfg['data_handling']['cache_size'],
                            hr_variable=cfg['highres']['variable'],
                            hr_model=cfg['highres']['model'],
                            hr_scaling_method=cfg['highres']['scaling_method'],
                            # hr_scaling_params=cfg['highres']['scaling_params'],
                            lr_conditions=cfg['lowres']['condition_variables'],
                            lr_model=cfg['lowres']['model'],
                            lr_scaling_methods=cfg['lowres']['scaling_methods'],
                            # lr_scaling_params=cfg['lowres']['scaling_params'],
                            lr_cond_dirs_zarr=lr_cond_dirs_gen,
                            geo_variables=geo_variables,
                            lsm_full_domain=data_lsm,
                            topo_full_domain=data_topo,
                            cfg = cfg,
                            split = "gen",
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
    raw_workers = cfg['data_handling']['num_workers'] 
    try:
        num_workers = int(raw_workers) if raw_workers is not None else 0
    except ValueError:
        # Fallback: treat non-numeric as 0 workers
        num_workers = 0

    gen_loader = DataLoader(
        gen_dataset,
        batch_size              = cfg['evaluation']['batch_size'],
        shuffle                 = False,
        num_workers             = 0, #max(2, num_workers // 4),
        # pin_memory              = pin_memory,
        # persistent_workers      = num_workers > 0,
        drop_last               = (len(gen_dataset) % cfg['training']['batch_size']) != 0,
    )

    # Print dataset information
    # if verbose:
    logger.info(f"Generation dataset: {len(gen_dataset)} samples\n")
    
    # Return the dataloaders
    return gen_loader


def infer_in_channels(cfg: dict) -> int:
    # low-res conditions
    n_lr = len(cfg['lowres']['condition_variables']) if cfg['lowres']['condition_variables'] is not None else 0
    # geo maps (value + mask)
    n_geo = 0
    if cfg['stationary_conditions']['geographic_conditions']['sample_w_geo']:
        n_geo = 2 * len(cfg["stationary_conditions"]["geographic_conditions"]["geo_variables"])
    return n_lr + n_geo

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
    input_channels = infer_in_channels(cfg)
    output_channels = 1#len(cfg['highres']['variable'])  # Assuming a single output channel for the high-resolution variable
    
    # Log the number of channels
    logger.info(f"Input channels: {input_channels}")
    logger.info(f"Output channels: {output_channels}")

    device = get_device()

    # === Model architecture knobs (decoder upsampling/norm/activation) ===
    model_cfg = cfg.get('model', {})
    use_resize_conv = bool(model_cfg.get('use_resize_conv', True))
    decoder_norm = model_cfg.get('decoder_norm', 'group')  # Options: 'group', 'instance', None
    decoder_gn_groups = int(model_cfg.get('decoder_gn_groups', 8))  # Number of groups for GroupNorm
    decoder_activation_name = model_cfg.get('decoder_activation', 'SiLU')  # Options: 'relu', 'sily', 'gelu', etc.
    decoder_activation_name_lower = decoder_activation_name.lower()

    _act_map = {'relu': nn.ReLU,
                'silu': nn.SiLU,
                'gelu': nn.GELU,}
    decoder_activation = _act_map.get(decoder_activation_name_lower, nn.ReLU)  # Default to SiLU if not found
    logger.info(f"[MODEL] use_resize_conv: {use_resize_conv}, decoder_norm: {decoder_norm}, decoder_gn_groups: {decoder_gn_groups}, decoder_activation: {decoder_activation_name}")

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
                      cond_on_img=sample_w_cond_img,
                      block_layers=cfg['sampler']['block_layers'],
                      num_classes=cfg['stationary_conditions']['seasonal_conditions']['n_seasons'] if cfg['stationary_conditions']['seasonal_conditions']['sample_w_cond_season'] else None,
                      n_heads=cfg['sampler']['num_heads'],
                      )
    decoder = Decoder(last_fmap_channels=cfg['sampler']['last_fmap_channels'],
                      output_channels=output_channels,
                      time_embedding=cfg['sampler']['time_embedding'],
                      n_heads=cfg['sampler']['num_heads'],
                      use_resize_conv=use_resize_conv,
                      norm=decoder_norm,
                      gn_groups=decoder_gn_groups,
                      activation=decoder_activation,
                      )
    
    edm_cfg = cfg.get('edm', {})
    edm_enabled = bool(edm_cfg.get('enabled', False))

    if edm_enabled:
        sigma_data = float(edm_cfg.get('sigma_data', 1.0))
        predict_residual = bool(edm_cfg.get('predict_residual', False)) # NOTE: Start with False, when EDM is stable, try True
        score_model = EDMPrecondUNet(encoder=encoder,
                                     decoder=decoder,
                                     sigma_data=sigma_data,
                                     predict_residual=predict_residual).to(device)
        
    else:
        score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn,
                            encoder=encoder,
                            decoder=decoder,
                            debug_pre_sigma_div=False
                            )

    if hasattr(score_model, "debug_pre_sigma_div"):
        object.__setattr__(score_model, "debug_pre_sigma_div", False)

    return score_model, checkpoint_dir, checkpoint_name


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
    lr_scheduler_type = cfg['training'].get('lr_scheduler', None)
    if lr_scheduler_type == 'Step':
        scheduler = StepLR(optimizer,
                           step_size=cfg['training']['lr_scheduler_params']['step_size'],
                           gamma=cfg['training']['lr_scheduler_params']['gamma'])
                           
    elif lr_scheduler_type == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='min',
                                      factor=cfg['training']['lr_scheduler_params']['factor'],
                                      patience=cfg['training']['lr_scheduler_params']['patience'],
                                      verbose=True)
    elif lr_scheduler_type == 'CosineAnnealing':
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=cfg['training']['lr_scheduler_params']['T_max'],
                                      eta_min=cfg['training']['lr_scheduler_params']['eta_min'])
    elif lr_scheduler_type == None:
        scheduler = None
        logger.warning("No learning rate scheduler specified. Using the optimizer's default learning rate.")
    else:
        raise ValueError(f"Scheduler {lr_scheduler_type} not recognized. Use 'step', 'reduce_on_plateau', or 'cosine_annealing'.")

    return scheduler



def get_device(verbose=True):
    """
    Get the device to be used for training.
    
    Returns:
        torch.device: The device (CPU or GPU) to be used.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        logger.info(f"Using device: {device}")
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


def setup_logger(log_dir, name="train_log", log_to_stdout=True):
    # Set up the path for the log directory
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")

    # Set up a logger, with level set to INFO which means it will log INFO, WARNING, ERROR, and CRITICAL messages
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers (we remove all handlers to avoid duplicates)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler to write logs to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    # Set the format for the log messages
    file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    # Apply the formatter to the file handler
    logger.addHandler(file_handler)

    # Optional: also print to terminal
    if log_to_stdout:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(file_formatter)
        logger.addHandler(stream_handler)

    logger.info(f"Logging to {log_path}")
    return logger

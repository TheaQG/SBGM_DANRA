

import os
import torch
import pickle
import zarr

import numpy as np


from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Import objects from other files in this repository
from sbgm.data_modules import DANRA_Dataset_cutouts_ERA5_Zarr
from sbgm.special_transforms import build_back_transforms
from sbgm.score_unet import Encoder, Decoder, ScoreNet, marginal_prob_std_fn, diffusion_coeff_fn
from sbgm.score_sampling import pc_sampler, Euler_Maruyama_sampler, ode_sampler
from sbgm.training import *
from sbgm.utils import *


'''
    .args to be passed to the main function:
    - hr_var
    - lr_conditions
    - hr_data_size
    - lr_data_size
    - cutout_domains
    - scaling
    - show_both_orig_scaled
    - force_matching_scale
    - transform_back_bf_plot
    - path_data
    - hr_model
    - lr_model
    - full_domain_dims



'''

import os


def launch_sbgm_from_args():
    '''
        Launch the training from the command line arguments
    '''
    

    parser = argparse.ArgumentParser(description='Train a model for the downscaling of climate data')
    parser.add_argument('--hr_model', type=str, default='DANRA', help='The HR model to use')
    parser.add_argument('--hr_var', type=str, default='prcp', help='The HR variable to use')
    parser.add_argument('--hr_data_size', type=str2list, default=[128,128], help='The HR image dimension as list, e.g. [128, 128]')
    parser.add_argument('--hr_scaling_method', type=str, default='log_minus1_1', help='The scaling method for the HR variable (zscore, log, log_minus1_1)')
    # Scaling params are provided as JSON-like strings
    parser.add_argument('--hr_scaling_params', type=str, default='{"glob_min": 0, "glob_max": 160, "glob_min_log": -20, "glob_max_log": 10, "glob_mean_log": -25.0, "glob_std_log": 10.0, "buffer_frac": 0.5}',# '{"glob_mean": 8.69251, "glob_std": 6.192434}', # 
                        help='The scaling parameters for the HR variable, in JSON-like string format dict') #
    parser.add_argument('--lr_model', type=str, default='ERA5', help='The LR model to use')
    parser.add_argument('--lr_conditions', type=str2list, default=["prcp",#],
                                                                   "temp"],#,
                                                                #    "ewvf",#],
                                                                #    "nwvf"],
                        help='List of LR condition variables')
    parser.add_argument('--lr_scaling_methods', type=str2list, default=["log_minus1_1",#],
                                                                        "zscore"],#],
                                                                        # "zscore",#],
                                                                        # "zscore"],
                        help='List of scaling methods for LR conditions')
    # Scaling params are provided as JSON-like strings
    parser.add_argument('--lr_scaling_params', type=str2list, default=['{"glob_min": 0, "glob_max": 70, "glob_min_log": -10, "glob_max_log": 5, "glob_mean_log": -25.0, "glob_std_log": 10.0, "buffer_frac": 0.5}',#],
                                                                       '{"glob_mean": 8.69251, "glob_std": 6.192434}'],#'{"glob_min": 0, "glob_max": 70, "glob_min_log": -10, "glob_max_log": 5, "glob_mean_log": -25.0, "glob_std_log": 10.0, "buffer_frac": 0.5}'],#,
                                                                    #    '{"glob_mean": 0.0, "glob_std": 500.0}',#],
                                                                    #    '{"glob_mean": 0.0, "glob_std": 500.0}'],
                        help='List of dicts of scaling parameters for LR conditions, in JSON-like string format dict')
    parser.add_argument('--lr_data_size', type=str2list, default=None, help='The LR image dimension as list, e.g. [128, 128]')
    parser.add_argument('--lr_cutout_domains', type=str2list, default=None, help='Cutout domain for LR conditioning and geo variables as [x1, x2, y1, y2]. If not provided, HR cutout is used.')#parser.add_argument('--lr_cutout_domains', nargs=4, type=int, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'), help='Cutout domains for LR conditioning area. If omitted, defaults to HR cutout domains.')
    parser.add_argument('--resize_factor', type=int, default=4, help='Resize factor to reduce input data size. Mainly used for testing on smaller data.')
    parser.add_argument('--force_matching_scale', type=str2bool, default=True, help='If True, force HR and LR images with the same variable to share the same color scale')
    parser.add_argument('--transform_back_bf_plot', type=str2bool, default=True, help='Whether to transform back before plotting')
    parser.add_argument('--sample_w_geo', type=str2bool, default=True, help='Whether to sample with lsm and topo')
    parser.add_argument('--sample_w_cutouts', type=str2bool, default=True, help='Whether to sample with cutouts')
    parser.add_argument('--sample_w_cond_season', type=str2bool, default=True, help='Whether to sample with conditional seasons')
    parser.add_argument('--sample_w_sdf', type=str2bool, default=True, help='Whether to sample with sdf')
    parser.add_argument('--scaling', type=str2bool, default=True, help='Whether to scale the data')
    parser.add_argument('--path_data', type=str, default='/Users/au728490/Library/CloudStorage/OneDrive-Aarhusuniversitet/PhD_AU/Python_Scripts/Data/Data_DiffMod/', help='The path to the data')
    parser.add_argument('--full_domain_dims', type=str2list, default=[589, 789], help='The full domain dimensions for the data')
    parser.add_argument('--save_figs', type=str2bool, default=False, help='Whether to save the figures')
    parser.add_argument('--specific_fig_name', type=str, default=None, help='If not None, saves figure with this name')
    parser.add_argument('--show_figs', type=str2bool, default=True, help='Whether to show the figures')
    parser.add_argument('--show_both_orig_scaled', type=str2bool, default=False, help='Whether to show both the original and scaled data in the same figure')
    parser.add_argument('--show_geo', type=str2bool, default=True, help='Whether to show the geo variables when plotting')
    parser.add_argument('--show_ocean', type=str2bool, default=False, help='Whether to show the ocean')
    parser.add_argument('--path_save', type=str, default='/Users/au728490/Library/CloudStorage/OneDrive-Aarhusuniversitet/PhD_AU/Python_Scripts/DiffusionModels/SBGM_modular/', help='The path to save the figures')
    parser.add_argument('--cutout_domains', type=str2list, default='170, 350, 340, 520', help='The cutout domains')
    parser.add_argument('--topo_min', type=int, default=-12, help='The minimum value of the topological data')
    parser.add_argument('--topo_max', type=int, default=330, help='The maximum value of the topological data')
    parser.add_argument('--norm_min', type=int, default=0, help='The minimum value of the normalized topological data')
    parser.add_argument('--norm_max', type=int, default=1, help='The maximum value of the normalized topological data')
    parser.add_argument('--n_seasons', type=int, default=4, help='The number of seasons')
    parser.add_argument('--n_gen_samples', type=int, default=3, help='The number of generated samples')
    parser.add_argument('--num_workers', type=int, default=4, help='The number of workers')
    parser.add_argument('--n_timesteps', type=int, default=1000, help='The number of timesteps in the diffusion process')
    parser.add_argument('--sampler', type=str, default='pc_sampler', help='The sampler to use for the langevin dynamics sampling')
    parser.add_argument('--num_heads', type=int, default=4, help='The number of heads in the attention mechanism')
    parser.add_argument('--last_fmap_channels', type=int, default=512, help='The number of channels in the last feature map')
    parser.add_argument('--time_embedding', type=int, default=256, help='The size of the time embedding')
    parser.add_argument('--check_transforms', type=str2bool, default=True, help='Whether to check the transforms by plotting')

    parser.add_argument('--cache_size', type=int, default=0, help='The cache size')
    parser.add_argument('--batch_size', type=int, default=16, help='The batch size')
    parser.add_argument('--device', type=str, default=None, help='The device to use for training')

    parser.add_argument('--learning_rate', type=float, default=1e-4, help='The learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='The minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='The weight decay')
    parser.add_argument('--epochs', type=int, default=500, help='The number of epochs')
    parser.add_argument('--lr_scheduler', type=str, default='ReduceLROnPlateau', help='The learning rate scheduler')
    parser.add_argument('--lr_scheduler_params', type=str2dict, default='{"factor": 0.5, "patience": 5, "threshold": 0.01, "min_lr": 1e-6}', help='The learning rate scheduler parameters')
    parser.add_argument('--early_stopping', type=str2bool, default=True, help='Whether to use early stopping')
    parser.add_argument('--early_stopping_params', type=str2dict, default='{"patience": 50, "min_delta": 0.0001}', help='The early stopping parameters')

    parser.add_argument('--optimizer', type=str, default='adam', help='The optimizer to use')
    parser.add_argument('--loss_type', type=str, default='sdfweighted', help='The type of loss function')

    parser.add_argument('--path_checkpoint', type=str, default='model_checkpoints/', help='The path to the checkpoints')
    parser.add_argument('--config_name', type=str, default='sbgm', help='The name of the configuration file')
    parser.add_argument('--create_figs', type=str2bool, default=True, help='Whether to create figures')
    # parser.add_argument('--show_figs', type=str2bool, default=False, help='Whether to show the figures')
    # parser.add_argument('--save_figs', type=str2bool, default=True, help='Whether to save the figures')
    parser.add_argument('--plot_interval', type=int, default=1, help='Number of epochs between plots')

    
    args = parser.parse_args()

    
    # Launch the training
    main_sbgm(args)


def main_sbgm(args):

    print('\n\n')
    print('#'*50)
    print('Running ddpm')
    print('#'*50)
    print('\n\n')

    # Define the device to use
    if args.device is not None:
        device = args.device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nUsing device: {device}')
    
    # Set HR and LR variables for use
    hr_var = args.hr_var
    lr_vars = args.lr_conditions
   

    
    # Define default LR colormaps and extra colormaps for geo variables
    cmap_prcp = 'inferno'
    cmap_temp = 'plasma'
    lr_cmap_dict = {"prcp": cmap_prcp, "temp": cmap_temp}
    extra_cmap_dict = {"topo": "terrain", "lsm": "binary", "sdf": "coolwarm"}

    if hr_var == 'temp':
        cmap_name = 'plasma'
        hr_units = r'$^\circ$C'
    elif hr_var == 'prcp':
        cmap_name = 'inferno'
        hr_units = 'mm'
    else:
        hr_units = 'Unknown'
    
    # Set units for LR conditions
    prcp_units = 'mm'
    temp_units = r'$^\circ$C'
    lr_units = []
    for cond in lr_vars:
        if cond == 'prcp':
            lr_units.append(prcp_units)
        elif cond == 'temp':
            lr_units.append(temp_units)
        else:
            lr_units.append('Unknown')

    # Set image dimensions (if None, use default 128x128)
    hr_data_size = tuple(args.hr_data_size) if args.hr_data_size is not None else None
    if hr_data_size is None:
        hr_data_size = (128, 128)
    lr_data_size = tuple(args.lr_data_size) if args.lr_data_size is not None else None

    if lr_data_size == None:
        lr_data_size_use = hr_data_size
    else:
        lr_data_size_use = lr_data_size

    # Check if resize factor is set and print sizes    
    if args.resize_factor > 1:
        hr_data_size_use = (hr_data_size[0]//args.resize_factor, hr_data_size[1]//args.resize_factor)
        lr_data_size_use = (lr_data_size_use[0]//args.resize_factor, lr_data_size_use[1]//args.resize_factor)
    else:
        hr_data_size_use = hr_data_size
        lr_data_size_use = lr_data_size_use

    print(f'\n\nHR data size OG: {hr_data_size}')
    print(f'\tHR data size reduced: ({hr_data_size_use[0]}, {hr_data_size_use[1]})')
    print(f'LR data size OG: {lr_data_size}')
    print(f'\tLR data size reduced: ({lr_data_size_use[0]}, {lr_data_size_use[1]})')

    # if lr_data_size is None:
    #     lr_data_size = (128, 128)
    
    # Set full domain size
    full_domain_dims = tuple(args.full_domain_dims) if args.full_domain_dims is not None else None

    # Use helper function to create the path for the zarr files
    print(f'\nUsing HR data type: {args.hr_model} {hr_var} [{hr_units}]')
    hr_data_dir_train = build_data_path(args.path_data, args.hr_model, hr_var, full_domain_dims, 'train')
    hr_data_dir_valid = build_data_path(args.path_data, args.hr_model, hr_var, full_domain_dims, 'valid')
    hr_data_dir_test = build_data_path(args.path_data, args.hr_model, hr_var, full_domain_dims, 'test')

    print(hr_data_dir_train)
    # Loop over lr_vars and create the path for the zarr files
    lr_cond_dirs_train = {}
    lr_cond_dirs_valid = {}
    lr_cond_dirs_test = {}
    for i, cond in enumerate(lr_vars):
        print(f'Using LR data type: {args.lr_model} {cond} [{lr_units[lr_vars.index(cond)]}]')
        # Check if cond is in extra_cmap_dict
        lr_cond_dirs_train[cond] = build_data_path(args.path_data, args.lr_model, cond, full_domain_dims, 'train')
        lr_cond_dirs_valid[cond] = build_data_path(args.path_data, args.lr_model, cond, full_domain_dims, 'valid')
        lr_cond_dirs_test[cond] = build_data_path(args.path_data, args.lr_model, cond, full_domain_dims, 'test')

    # Set scaling and matching options
    scaling = args.scaling
    show_both_orig_scaled = args.show_both_orig_scaled
    force_matching_scale = args.force_matching_scale
    transform_back_bf_plot = args.transform_back_bf_plot

    # Set up scaling methods
    hr_scaling_method = args.hr_scaling_method
    hr_scaling_params = ast.literal_eval(args.hr_scaling_params)
    lr_scaling_methods = args.lr_scaling_methods
    lr_scaling_params = [ast.literal_eval(param) for param in args.lr_scaling_params]

    # Set up back transformations (for plotting and visual inspection)
    back_transforms = build_back_transforms(
        hr_var               = hr_var,
        hr_scaling_method    = hr_scaling_method,
        hr_scaling_params    = hr_scaling_params,
        lr_vars              = lr_vars,
        lr_scaling_methods   = lr_scaling_methods,
        lr_scaling_params    = lr_scaling_params,
    )

    # Setup geo variables
    sample_w_sdf = args.sample_w_sdf
    if sample_w_sdf:
        print('\nSDF weighted loss enabled. Setting lsm and topo to True.\n')
        sample_w_geo = True
    else:
        sample_w_geo = args.sample_w_geo

    if sample_w_geo:
        geo_variables = ['lsm', 'topo']
        data_dir_lsm = args.path_data + 'data_lsm/truth_fullDomain/lsm_full.npz'
        data_dir_topo = args.path_data + 'data_topo/truth_fullDomain/topo_full.npz'
        data_lsm = np.flipud(np.load(data_dir_lsm)['data'])
        data_topo = np.flipud(np.load(data_dir_topo)['data'])
        if scaling:
            if args.topo_min is None or args.topo_max is None:
                topo_min, topo_max = np.min(data_topo), np.max(data_topo)
            else:
                topo_min, topo_max = args.topo_min, args.topo_max
            if args.norm_min is None or args.norm_max is None:
                norm_min, norm_max = np.min(data_lsm), np.max(data_lsm)
            else:
                norm_min, norm_max = args.norm_min, args.norm_max
            OldRange = (topo_max - topo_min)
            NewRange = (norm_max - norm_min)
            data_topo = ((data_topo - topo_min) * NewRange / OldRange) + norm_min
    else:
        geo_variables = None
        data_lsm = None
        data_topo = None



    # Setup cutouts
    sample_w_cutouts = args.sample_w_cutouts
    # Set cutout domains, if None, use default (170, 350, 340, 520) (DK area with room for shuffle) 
    cutout_domains = tuple(args.cutout_domains) if args.cutout_domains is not None else None
    if cutout_domains is None:
        cutout_domains = (170, 350, 340, 520)
    lr_cutout_domains = tuple(args.lr_cutout_domains) if args.lr_cutout_domains is not None else None
    if lr_cutout_domains is None:
        lr_cutout_domains = (170, 350, 340, 520)
    # Setup conditional seasons (classification)
    sample_w_cond_season = args.sample_w_cond_season
    if sample_w_cond_season:
        n_seasons = args.n_seasons
    else:
        n_seasons = None

    # Make zarr groups
    data_danra_train_zarr = zarr.open_group(hr_data_dir_train, mode='r')
    data_danra_valid_zarr = zarr.open_group(hr_data_dir_valid, mode='r')
    data_danra_test_zarr = zarr.open_group(hr_data_dir_test, mode='r')

    # /scratch/project_465000956/quistgaa/Data/Data_DiffMod/
    n_files_train = len(list(data_danra_train_zarr.keys()))
    n_files_valid = len(list(data_danra_valid_zarr.keys()))
    n_files_test = len(list(data_danra_test_zarr.keys()))


    n_samples_train = n_files_train
    n_samples_valid = n_files_valid
    n_samples_test = n_files_test
    
    cache_size = args.cache_size
    if cache_size == 0:
        cache_size_train = n_samples_train//2
        cache_size_valid = n_samples_valid//2
        cache_size_test = n_samples_test//2
    else:
        cache_size_train = cache_size
        cache_size_valid = cache_size
        cache_size_test = cache_size
    print(f'\n\n\nNumber of training samples: {n_samples_train}')
    print(f'Number of validation samples: {n_samples_valid}')
    print(f'Number of test samples: {n_samples_test}\n')
    print(f'Total number of samples: {n_samples_train + n_samples_valid + n_samples_test}\n\n\n')

    print(f'\n\n\nCache size for training: {cache_size_train}')
    print(f'Cache size for validation: {cache_size_valid}')
    print(f'Cache size for test: {cache_size_test}\n')
    print(f'Total cache size: {cache_size_train + cache_size_valid + cache_size_test}\n\n\n')




    # Define model parameters
    input_channels = len(lr_vars) # equal to number of LR variables
    output_channels = 1
    last_fmap_channels = args.last_fmap_channels
    time_embedding = args.time_embedding
    num_heads = args.num_heads

    if lr_vars is not None:
        sample_w_cond_img = True
    else:
        sample_w_cond_img = False


    # Define hyper parameters
    learning_rate = args.learning_rate
    min_lr = args.min_lr
    weight_decay = args.weight_decay
    epochs = args.epochs




    # Setup specific names for saving
    lr_vars_str = '_'.join(lr_vars)
    save_str = (
        f"{args.config_name}__"
        f"HR_{hr_var}_{args.hr_model}__"
        f"SIZE_{hr_data_size_use[0]}x{hr_data_size_use[1]}__"
        f"LR_{lr_vars_str}_{args.lr_model}__"
        f"LOSS_{args.loss_type}__"
        f"HEADS_{num_heads}__"
        f"TIMESTEPS_{args.n_timesteps}"
    )

    #Setup checkpoint path
    checkpoint_dir = os.path.join(args.path_save, args.path_checkpoint)

    checkpoint_name = save_str + '.pth.tar'

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)


    # Set path to figures, samples, losses
    path_samples = args.path_save + 'samples' + f'/Samples' + '__' + save_str
    path_losses = args.path_save + '/losses'
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



    train_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(
                                hr_variable_dir_zarr=hr_data_dir_train,
                                hr_data_size=hr_data_size,
                                n_samples=n_samples_train,
                                cache_size=cache_size_train,
                                hr_variable=hr_var,
                                hr_model=args.hr_model,
                                hr_scaling_method=hr_scaling_method,
                                hr_scaling_params=hr_scaling_params,
                                lr_conditions=lr_vars,
                                lr_model=args.lr_model,
                                lr_scaling_methods=lr_scaling_methods,
                                lr_scaling_params=lr_scaling_params,
                                lr_cond_dirs_zarr=lr_cond_dirs_train,
                                geo_variables=geo_variables,
                                lsm_full_domain=data_lsm,
                                topo_full_domain=data_topo,
                                shuffle=True,
                                cutouts=sample_w_cutouts,
                                cutout_domains=cutout_domains if sample_w_cutouts else None,
                                n_samples_w_cutouts=n_samples_train,
                                sdf_weighted_loss=sample_w_sdf,
                                scale=scaling,
                                save_original=args.show_both_orig_scaled,
                                conditional_seasons=sample_w_cond_season,
                                n_classes=n_seasons,
                                lr_data_size=tuple(lr_data_size) if lr_data_size is not None else None,
                                lr_cutout_domains=tuple(lr_cutout_domains) if lr_cutout_domains is not None else None,
                                resize_factor=args.resize_factor,
                                )
    
    valid_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(
                                hr_variable_dir_zarr=hr_data_dir_valid,
                                hr_data_size=hr_data_size,
                                n_samples=n_samples_valid,
                                cache_size=cache_size_valid,
                                hr_variable=hr_var,
                                hr_model=args.hr_model,
                                hr_scaling_method=hr_scaling_method,
                                hr_scaling_params=hr_scaling_params,
                                lr_conditions=lr_vars,
                                lr_model=args.lr_model,
                                lr_scaling_methods=lr_scaling_methods,
                                lr_scaling_params=lr_scaling_params,
                                lr_cond_dirs_zarr=lr_cond_dirs_valid,
                                geo_variables=geo_variables,
                                lsm_full_domain=data_lsm,
                                topo_full_domain=data_topo,
                                shuffle=False,
                                cutouts=sample_w_cutouts,
                                cutout_domains=cutout_domains if sample_w_cutouts else None,
                                n_samples_w_cutouts=n_samples_valid,
                                sdf_weighted_loss=sample_w_sdf,
                                scale=scaling,
                                save_original=args.show_both_orig_scaled,
                                conditional_seasons=sample_w_cond_season,
                                n_classes=n_seasons,
                                lr_data_size=tuple(lr_data_size) if lr_data_size is not None else None,
                                lr_cutout_domains=tuple(lr_cutout_domains) if lr_cutout_domains is not None else None,
                                resize_factor=args.resize_factor,
                                )

    gen_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(
                                hr_variable_dir_zarr=hr_data_dir_test,
                                hr_data_size=hr_data_size,
                                n_samples=n_samples_test,
                                cache_size=cache_size_test,
                                hr_variable=hr_var,
                                hr_model=args.hr_model,
                                hr_scaling_method=hr_scaling_method,
                                hr_scaling_params=hr_scaling_params,
                                lr_conditions=lr_vars,
                                lr_model=args.lr_model,
                                lr_scaling_methods=lr_scaling_methods,
                                lr_scaling_params=lr_scaling_params,
                                lr_cond_dirs_zarr=lr_cond_dirs_test,
                                geo_variables=geo_variables,
                                lsm_full_domain=data_lsm,
                                topo_full_domain=data_topo,
                                shuffle=True,
                                cutouts=sample_w_cutouts,
                                cutout_domains=cutout_domains if sample_w_cutouts else None,
                                n_samples_w_cutouts=n_samples_test,
                                sdf_weighted_loss=sample_w_sdf,
                                scale=scaling,
                                save_original=args.show_both_orig_scaled,
                                conditional_seasons=sample_w_cond_season,
                                n_classes=n_seasons,
                                lr_data_size=tuple(lr_data_size) if lr_data_size is not None else None,
                                lr_cutout_domains=tuple(lr_cutout_domains) if lr_cutout_domains is not None else None,
                                resize_factor=args.resize_factor,
                                )    

    # Define batch size
    batch_size = args.batch_size

    # Define the torch dataloaders for train and validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)#, num_workers=args.num_workers)

    n_gen_samples = args.n_gen_samples
    gen_dataloader = DataLoader(gen_dataset, batch_size=n_gen_samples, shuffle=False)#, num_workers=args.num_workers)


    # Examine sample from train dataloader (sample is full batch)
    print('\n')
    sample = train_dataloader.dataset[0]
    for key, value in sample.items():
        try:
            print(f'{key}: {value.shape}')
        except AttributeError:
            print(f'{key}: {value}')
    print('\n\n')
            

    
    # Get first batch of samples from dataloader
    # samples = next(iter(train_dataloader))
    
    fig, axs = plot_sample(sample,
                    hr_model = args.hr_model,
                    hr_units = hr_units,
                    lr_model = args.lr_model,
                    lr_units = lr_units,
                    var = hr_var,
                    show_ocean=args.show_ocean,
                    hr_cmap = cmap_name,
                    lr_cmap_dict = lr_cmap_dict,
                    extra_keys = ['topo', 'lsm', 'sdf'],
                    extra_cmap_dict = extra_cmap_dict
                    )
    # Save the figure
    SAVE_NAME = 'Initial_sample_plot.png'
    fig.savefig(os.path.join(path_samples, SAVE_NAME), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"▸ Saved initial sample plot to {path_samples}/{SAVE_NAME}")
    # Set PLOT_FIRST to False to avoid plotting every batch
    




    # Define the seed for reproducibility, and set seed for torch, numpy and random
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Set torch to deterministic mode, meaning that the same input will always produce the same output
    torch.backends.cudnn.deterministic = False
    # Set torch to benchmark mode, meaning that the best algorithm will be chosen for the input
    torch.backends.cudnn.benchmark = True


    print(f'\n\n\nInput channels: {input_channels}')
    print(f'Output channels: {output_channels}')
    # Define the encoder and decoder from modules_DANRA_downscaling.py
    encoder = Encoder(input_channels, 
                        time_embedding,
                        cond_on_lsm=sample_w_geo,
                        cond_on_topo=sample_w_geo,
                        cond_on_img=sample_w_cond_img, 
                        cond_img_dim=(len(lr_vars), lr_data_size_use[0], lr_data_size_use[1]),
                        block_layers=[2, 2, 2, 2], 
                        num_classes=n_seasons,
                        n_heads=num_heads
                        )
    decoder = Decoder(last_fmap_channels, 
                        output_channels, 
                        time_embedding, 
                        n_heads=num_heads
                        )
    score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn, encoder=encoder, decoder=decoder)
    score_model = score_model.to(device)


    # Define the optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.AdamW(score_model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(score_model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(score_model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay)


    # Define the training pipeline
    pipeline = TrainingPipeline_general(score_model,
                                        loss_fn,
                                        marginal_prob_std_fn,
                                        optimizer,
                                        device,
                                        scaling=scaling,
                                        hr_var=hr_var,
                                        hr_scaling_method=hr_scaling_method,
                                        hr_scaling_params=hr_scaling_params,
                                        lr_vars=lr_vars,
                                        lr_scaling_methods=lr_scaling_methods,
                                        lr_scaling_params=lr_scaling_params,
                                        weight_init=True,
                                        sdf_weighted_loss=sample_w_sdf,
                                        check_transforms=args.check_transforms,
                                        )
    
    # Check if path to pretrained model exists
    if os.path.isfile(checkpoint_path):
        print('\n\nLoading pretrained weights from checkpoint:')
        print(checkpoint_path)

        checkpoint_state = torch.load(checkpoint_path, map_location=device)['network_params']
        pipeline.model.load_state_dict(checkpoint_state)
    else:
        print('\n\nNo pretrained weights found at checkpoint:')
        print(checkpoint_dir)
        print(f'Under name: {checkpoint_name}')
        print('Starting training from scratch...\n\n')

    # Define learning rate scheduler
    if args.lr_scheduler is not None:
        lr_scheduler_params = args.lr_scheduler_params
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(pipeline.optimizer,
                                                                  'min',
                                                                  factor = lr_scheduler_params['factor'],
                                                                  patience = lr_scheduler_params['patience'],
                                                                  threshold = lr_scheduler_params['threshold'],
                                                                  min_lr = min_lr
                                                                  )
        


    # Check if device is cude, if so print information and empty cache
    if torch.cuda.is_available():
        print(f'\nModel is training on {torch.cuda.get_device_name()}')
        print(f'Model is using {torch.cuda.memory_allocated()} bytes of memory\n\n')
        torch.cuda.empty_cache()
        
    
    # Set empty lists for storing losses
    train_losses = []
    valid_losses = []

    # Set best loss to infinity
    best_loss = np.inf

    

    print('\n\n\nStarting training...\n\n\n')

    
    # Loop over epochs
    for epoch in range(epochs):

        print(f'\n\n\nEpoch {epoch+1} of {epochs}\n\n\n')
        if epoch == 0:
            PLOT_FIRST_IMG = True
        else:
            PLOT_FIRST_IMG = False

        # Calculate the training loss
        train_loss = pipeline.train(train_dataloader,
                                    verbose=False,
                                    PLOT_FIRST=PLOT_FIRST_IMG,
                                    SAVE_PATH=path_samples,
                                    SAVE_NAME='data_sample.png'
                                    )
        train_losses.append(train_loss)

        # Calculate the validation loss
        valid_loss = pipeline.validate(valid_dataloader,
                                        verbose=False,
                                        )
        valid_losses.append(valid_loss)

        # Print the training and validation losses
        print(f'\n\n\nTraining loss: {train_loss:.6f}')
        print(f'Validation loss: {valid_loss:.6f}\n\n\n')
        
        with open(path_losses + '/' + name_losses + '_train', 'wb') as fp:
            pickle.dump(train_losses, fp)
        with open(path_losses + '/' + name_losses + '_valid', 'wb') as fp:
            pickle.dump(valid_losses, fp)

        if args.create_figs:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.plot(train_losses, label='Train')
            ax.plot(valid_losses, label='Validation')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss')
            ax.legend(loc='upper right')
            fig.tight_layout()
            if args.show_figs:
                plt.show()
                
            if args.save_figs:
                fig.savefig(path_figures + name_losses + '.png', dpi=600, bbox_inches='tight', pad_inches=0.1)


        # If validation loss is lower than best loss, save the model. With possibility of early stopping
        if valid_loss < best_loss:
            best_loss = valid_loss
            pipeline.save_model(checkpoint_dir, checkpoint_name)
            print(f'Model saved at epoch {epoch+1} with validation loss: {valid_loss:.6f}')
            print(f'Saved to: {checkpoint_dir} with name {checkpoint_name}\n\n')

            # If create figures is enabled, create figures
            if args.create_figs and n_gen_samples > 0: # and epoch % args.plot_interval == 0:
                if hr_var == 'temp':
                    cmap_var_name = 'plasma'
                elif hr_var == 'prcp':
                    cmap_var_name = 'inferno'

                if epoch == 0:
                    print('First epoch, generating samples...')
                else:
                    print('Valid loss is better than best loss, and epoch is a multiple of plot interval, generating samples...')

                # Load the model
                best_model_path = checkpoint_path
                best_model_state = torch.load(best_model_path)['network_params']

                # Load the model state
                pipeline.model.load_state_dict(best_model_state)
                # Set model to evaluation mode (remember to set back to training mode after generating samples)
                pipeline.model.eval()

                # Set topography, lsm and sdf vmin and vmax for colorbars 
                lsm_vmin = 0
                lsm_vmax = 1
                sdf_vmin = 0
                sdf_vmax = 1
                # Set topo based on scaling or not
                if scaling:
                    topo_vmin = 1
                    topo_vmax = 0
                else:
                    topo_vmin = -12
                    topo_vmax = 300
                
                # Setup dictionary with plotting specifics for each variable
                plot_settings = {
                    'Truth': {'cmap': cmap_var_name, 'use_local_scale': True, 'vmin': None, 'vmax': None},
                    'Condition': {'cmap': cmap_var_name, 'use_local_scale': True, 'vmin': None, 'vmax': None},
                    'Generated': {'cmap': cmap_var_name, 'use_local_scale': True, 'vmin': None, 'vmax': None},
                    'Topography': {'cmap': 'terrain', 'use_local_scale': False, 'vmin': topo_vmin, 'vmax': topo_vmax},
                    'LSM': {'cmap': 'binary', 'use_local_scale': False, 'vmin': lsm_vmin, 'vmax': lsm_vmax},
                    'SDF': {'cmap': 'coolwarm', 'use_local_scale': False, 'vmin': sdf_vmin, 'vmax': sdf_vmax},
                }

                for idx, samples in tqdm.tqdm(enumerate(gen_dataloader), total=len(gen_dataloader)):
                    sample_batch_size = n_gen_samples
                    
                    # Define the sampler to use
                    sampler_name = args.sampler # ['pc_sampler', 'Euler_Maruyama_sampler', 'ode_sampler']

                    if sampler_name == 'pc_sampler':
                        sampler = pc_sampler
                    elif sampler_name == 'Euler_Maruyama_sampler':
                        sampler = Euler_Maruyama_sampler
                    elif sampler_name == 'ode_sampler':
                        sampler = ode_sampler
                    else:
                        raise ValueError('Sampler not recognized. Please choose between: pc_sampler, Euler_Maruyama_sampler, ode_sampler')

                    test_images, test_seasons, test_cond, test_lsm_hr, test_lsm, test_sdf, test_topo, _, _ = extract_samples(samples, device=device)
                    data_plot = [test_images, test_cond, test_lsm, test_sdf, test_topo]
                    data_names = ['Truth', 'Condition', 'LSM', 'SDF', 'Topography']
                    # Filter out None samples
                    data_plot = [sample for sample in data_plot if sample is not None]
                    data_names = [name for name, sample in zip(data_names, data_plot) if sample is not None]

                    # Count length of data_plot
                    n_axs = len(data_plot)

                    generated_samples = sampler(score_model=score_model,
                                                marginal_prob_std=marginal_prob_std_fn,
                                                diffusion_coeff=diffusion_coeff_fn,
                                                batch_size=sample_batch_size,
                                                num_steps=args.n_timesteps,
                                                device=device,
                                                img_size=hr_data_size[0],
                                                y=test_seasons,
                                                cond_img=test_cond,
                                                lsm_cond=test_lsm,
                                                topo_cond=test_topo,
                                                ).squeeze()
                    generated_samples = generated_samples.detach().cpu()

                    # Append generated samples to data_plot and data_names
                    data_plot.append(generated_samples)
                    data_names.append('Generated')

                    if args.check_transforms and scaling:
                        # Get a sample and its back transformed version
                        sample = samples['hr']
                        

                    # Use plot_samples_and_generated to plot 
                    fig, _ = plot_samples_and_generated(
                        samples,                  # the original sample dictionary
                        generated_samples,              # tensor (B,1,H,W)
                        hr_model=args.hr_model,
                        hr_units=hr_units,
                        lr_model=args.lr_model,
                        lr_units=lr_units,
                        var=hr_var,
                        scaling=scaling,
                        show_ocean=args.show_ocean,
                        transform_back_bf_plot=transform_back_bf_plot,
                        back_transforms=back_transforms,
                        hr_cmap = cmap_name,
                        lr_cmap_dict = lr_cmap_dict,
                        # extra_keys = ['topo', 'lsm', 'sdf'],
                        # extra_cmap_dict = extra_cmap_dict
                    )

                    # Show figure if requested
                    if args.show_figs:
                        plt.show()
                    
                    
                    # Save figure
                    if args.save_figs:
                        if epoch == (epochs - 1):
                            fig.savefig(path_figures + name_final_samples + '.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
                            print(f'Saving final generated sample in {path_samples} as {name_final_samples}.png')
                        else:
                            fig.savefig(path_figures + name_samples + str(epoch+1) + '.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
                            print(f'Saving generated samples in {path_figures} as {name_samples}{epoch+1}.png')
                    
                    # Close figure
                    plt.close(fig)
                                    

                    break
                
                # Set model back to train mode
                pipeline.model.train()
            


            # Early stopping
            PATIENCE = args.early_stopping_params['patience']
        else:
            PATIENCE -= 1
            if PATIENCE == 0:
                print(f'Early stopping at epoch {epoch+1}')
                break

        # Close any and all figures
        plt.close('all')
        
        # If learning rate scheduler is not None, step the scheduler
        if args.lr_scheduler is not None:
            lr_scheduler.step(valid_loss)                    

if __name__ == '__main__':

    launch_sbgm_from_args()

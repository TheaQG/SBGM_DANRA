"""
    CLI WRAPPER FOR GENERATING SAMPLES
    SHOULD BE COMBINED WITH EVALUATION AS WELL
"""


import argparse

from generation import GenerationSBGM

from utils import *

def generate_from_args():
    '''
        Launch the generation from the command line arguments
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
    parser.add_argument('--check_transforms', type=str2bool, default=False, help='Whether to check the transforms applied to the data')
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
    parser.add_argument('--plot_interval', type=int, default=5, help='Number of epochs between plots')

    args = parser.parse_args()

    # Use the GenerationSBGM class to launch the generation
    generation = GenerationSBGM(args)

    # Launch the multiple sample generation
    generation.generate_multiple_samples(8,
                                         plot_samples=True,
                                         show_plots=False,
                                         save_sample_plots=True,
                                         )

    # Launch the single sample generation
    generation.generate_single_sample(plot_samples=True,
                                      show_plots=False,
                                      save_sample_plots=True,
                                      )

    # Launch the repeated single sample generation
    generation.generate_repeated_single_sample(8,
                                               plot_samples=True,
                                               show_plots=False,
                                               save_sample_plots=True,
                                               )

    # # Launch the generation
    # generation_sbgm(args)


if __name__ == '__main__':
    generate_from_args()
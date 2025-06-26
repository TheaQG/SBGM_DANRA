'''
    Script to run evaluations of trained ddpm models on test DANRA dataset.
    Default evalutes on evaluation set of size equal to two years of data (730 samples), 2001-2002.
    The default size is 64x64.
    Script only creates samples, does not evaluate or plot.

    !!! MISSING: No shift in image, steady over specific area of DK
    


'''

import torch
import os

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from sbgm.data_modules import *
from sbgm.score_unet import *
from sbgm.score_sampling import *
from sbgm.training import *
from sbgm.utils import *

def generation_main(cfg):
    """
        Main function to run generation.
    """
    # # Set up the generation class with the provided configuration
    # generation = GenerationSBGM(cfg)

    # # Set up data folder for generation
    # generation.setup_data_folder(n_gen_samples=cfg['generation']['n_gen_samples'])

    # # Generate multiple samples
    # generation.generate_multiple_samples(
    #     n_gen_samples=cfg['generation']['n_gen_samples'],
    #     plot_samples=cfg['visualization']['plot_samples'],
    #     show_plots=cfg['visualization']['show_plots'],
    #     save_sample_plots=cfg['visualization']['save_sample_plots']
    # )

    # # Generate single sample
    # generation.generate_single_sample(
    #     plot_samples=cfg['visualization']['plot_samples'],
    #     show_plots=cfg['visualization']['show_plots'],
    #     save_sample_plots=cfg['visualization']['save_sample_plots']
    # )

    # # Generate repeated single sample
    # generation.generate_repeated_single_sample(
    #     n_repeats=4,
    #     plot_samples=cfg['visualization']['plot_samples'],
    #     show_plots=cfg['visualization']['show_plots'],
    #     save_sample_plots=cfg['visualization']['save_sample_plots']
    # )


# def generate_from_args():
#     '''
#         Launch the generation from the command line arguments
#     '''
    

#     parser = argparse.ArgumentParser(description='Train a model for the downscaling of climate data')
#     parser.add_argument('--hr_model', type=str, default='DANRA', help='The HR model to use')
#     parser.add_argument('--hr_var', type=str, default='prcp', help='The HR variable to use')
#     parser.add_argument('--hr_data_size', type=str2list, default=[128,128], help='The HR image dimension as list, e.g. [128, 128]')
#     parser.add_argument('--hr_scaling_method', type=str, default='log_minus1_1', help='The scaling method for the HR variable (zscore, log, log_minus1_1)')
#     # Scaling params are provided as JSON-like strings
#     parser.add_argument('--hr_scaling_params', type=str, default='{"glob_min": 0, "glob_max": 160, "glob_min_log": -20, "glob_max_log": 10, "glob_mean_log": -25.0, "glob_std_log": 10.0, "buffer_frac": 0.5}',# '{"glob_mean": 8.69251, "glob_std": 6.192434}', # 
#                         help='The scaling parameters for the HR variable, in JSON-like string format dict') #
#     parser.add_argument('--lr_model', type=str, default='ERA5', help='The LR model to use')
#     parser.add_argument('--lr_conditions', type=str2list, default=["prcp",#],
#                                                                    "temp"],#,
#                                                                 #    "ewvf",#],
#                                                                 #    "nwvf"],
#                         help='List of LR condition variables')
#     parser.add_argument('--lr_scaling_methods', type=str2list, default=["log_minus1_1",#],
#                                                                         "zscore"],#],
#                                                                         # "zscore",#],
#                                                                         # "zscore"],
#                         help='List of scaling methods for LR conditions')
#     # Scaling params are provided as JSON-like strings
#     parser.add_argument('--lr_scaling_params', type=str2list, default=['{"glob_min": 0, "glob_max": 70, "glob_min_log": -10, "glob_max_log": 5, "glob_mean_log": -25.0, "glob_std_log": 10.0, "buffer_frac": 0.5}',#],
#                                                                        '{"glob_mean": 8.69251, "glob_std": 6.192434}'],#'{"glob_min": 0, "glob_max": 70, "glob_min_log": -10, "glob_max_log": 5, "glob_mean_log": -25.0, "glob_std_log": 10.0, "buffer_frac": 0.5}'],#,
#                                                                     #    '{"glob_mean": 0.0, "glob_std": 500.0}',#],
#                                                                     #    '{"glob_mean": 0.0, "glob_std": 500.0}'],
#                         help='List of dicts of scaling parameters for LR conditions, in JSON-like string format dict')
#     parser.add_argument('--lr_data_size', type=str2list, default=None, help='The LR image dimension as list, e.g. [128, 128]')
#     parser.add_argument('--lr_cutout_domains', type=str2list, default=None, help='Cutout domain for LR conditioning and geo variables as [x1, x2, y1, y2]. If not provided, HR cutout is used.')#parser.add_argument('--lr_cutout_domains', nargs=4, type=int, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'), help='Cutout domains for LR conditioning area. If omitted, defaults to HR cutout domains.')
#     parser.add_argument('--resize_factor', type=int, default=4, help='Resize factor to reduce input data size. Mainly used for testing on smaller data.')
#     parser.add_argument('--check_transforms', type=str2bool, default=False, help='Whether to check the transforms on the data')
#     parser.add_argument('--force_matching_scale', type=str2bool, default=True, help='If True, force HR and LR images with the same variable to share the same color scale')
#     parser.add_argument('--transform_back_bf_plot', type=str2bool, default=True, help='Whether to transform back before plotting')
#     parser.add_argument('--sample_w_geo', type=str2bool, default=True, help='Whether to sample with lsm and topo')
#     parser.add_argument('--sample_w_cutouts', type=str2bool, default=True, help='Whether to sample with cutouts')
#     parser.add_argument('--sample_w_cond_season', type=str2bool, default=True, help='Whether to sample with conditional seasons')
#     parser.add_argument('--sample_w_sdf', type=str2bool, default=True, help='Whether to sample with sdf')
#     parser.add_argument('--scaling', type=str2bool, default=True, help='Whether to scale the data')
#     parser.add_argument('--path_data', type=str, default='/Users/au728490/Library/CloudStorage/OneDrive-Aarhusuniversitet/PhD_AU/Python_Scripts/Data/Data_DiffMod/', help='The path to the data')
#     parser.add_argument('--full_domain_dims', type=str2list, default=[589, 789], help='The full domain dimensions for the data')
#     parser.add_argument('--save_figs', type=str2bool, default=False, help='Whether to save the figures')
#     parser.add_argument('--specific_fig_name', type=str, default=None, help='If not None, saves figure with this name')
#     parser.add_argument('--show_figs', type=str2bool, default=True, help='Whether to show the figures')
#     parser.add_argument('--show_both_orig_scaled', type=str2bool, default=True, help='Whether to show both the original and scaled data in the same figure')
#     parser.add_argument('--show_geo', type=str2bool, default=True, help='Whether to show the geo variables when plotting')
#     parser.add_argument('--show_ocean', type=str2bool, default=False, help='Whether to show the ocean')
#     parser.add_argument('--path_save', type=str, default='/Users/au728490/Library/CloudStorage/OneDrive-Aarhusuniversitet/PhD_AU/Python_Scripts/DiffusionModels/SBGM_modular/', help='The path to save the figures')
#     parser.add_argument('--cutout_domains', type=str2list, default='170, 350, 340, 520', help='The cutout domains')
#     parser.add_argument('--topo_min', type=int, default=-12, help='The minimum value of the topological data')
#     parser.add_argument('--topo_max', type=int, default=330, help='The maximum value of the topological data')
#     parser.add_argument('--norm_min', type=int, default=0, help='The minimum value of the normalized topological data')
#     parser.add_argument('--norm_max', type=int, default=1, help='The maximum value of the normalized topological data')
#     parser.add_argument('--n_seasons', type=int, default=4, help='The number of seasons')
#     parser.add_argument('--n_gen_samples', type=int, default=3, help='The number of generated samples')
#     parser.add_argument('--num_workers', type=int, default=4, help='The number of workers')
#     parser.add_argument('--n_timesteps', type=int, default=1000, help='The number of timesteps in the diffusion process')
#     parser.add_argument('--sampler', type=str, default='pc_sampler', help='The sampler to use for the langevin dynamics sampling')
#     parser.add_argument('--num_heads', type=int, default=4, help='The number of heads in the attention mechanism')
#     parser.add_argument('--last_fmap_channels', type=int, default=512, help='The number of channels in the last feature map')
#     parser.add_argument('--time_embedding', type=int, default=256, help='The size of the time embedding')
#     parser.add_argument('--cache_size', type=int, default=0, help='The cache size')
#     parser.add_argument('--batch_size', type=int, default=16, help='The batch size')
#     parser.add_argument('--device', type=str, default=None, help='The device to use for training')

#     parser.add_argument('--learning_rate', type=float, default=1e-4, help='The learning rate')
#     parser.add_argument('--min_lr', type=float, default=1e-6, help='The minimum learning rate')
#     parser.add_argument('--weight_decay', type=float, default=1e-6, help='The weight decay')
#     parser.add_argument('--epochs', type=int, default=500, help='The number of epochs')
#     parser.add_argument('--lr_scheduler', type=str, default='ReduceLROnPlateau', help='The learning rate scheduler')
#     parser.add_argument('--lr_scheduler_params', type=str2dict, default='{"factor": 0.5, "patience": 5, "threshold": 0.01, "min_lr": 1e-6}', help='The learning rate scheduler parameters')
#     parser.add_argument('--early_stopping', type=str2bool, default=True, help='Whether to use early stopping')
#     parser.add_argument('--early_stopping_params', type=str2dict, default='{"patience": 50, "min_delta": 0.0001}', help='The early stopping parameters')

#     parser.add_argument('--optimizer', type=str, default='adam', help='The optimizer to use')
#     parser.add_argument('--loss_type', type=str, default='sdfweighted', help='The type of loss function')

#     parser.add_argument('--path_checkpoint', type=str, default='model_checkpoints/', help='The path to the checkpoints')
#     parser.add_argument('--config_name', type=str, default='sbgm', help='The name of the configuration file')
#     parser.add_argument('--create_figs', type=str2bool, default=True, help='Whether to create figures')
#     # parser.add_argument('--show_figs', type=str2bool, default=False, help='Whether to show the figures')
#     # parser.add_argument('--save_figs', type=str2bool, default=True, help='Whether to save the figures')
#     parser.add_argument('--plot_interval', type=int, default=5, help='Number of epochs between plots')
#     parser.add_argument('--gen_years', type=str2list, default=[2018, 2020], help='The years to generate samples for')

#     args = parser.parse_args()

#     # Use the GenerationSBGM class to launch the generation
#     generation = GenerationSBGM(args)

#     # Launch the multiple sample generation
#     generation.generate_multiple_samples(n_gen_samples=args.n_gen_samples,
#                                          plot_samples=True,
#                                          show_plots=False,
#                                          save_sample_plots=True,
#                                          )

#     # Launch the single sample generation
#     generation.generate_single_sample(plot_samples=True,
#                                       show_plots=False,
#                                       save_sample_plots=True,
#                                       )

#     ''' 
#     !!!!!!!!!!!!!! PLOTTING DOES NOT WORK YET !!!!!!!!!!!!!!
#     '''
#     # Launch the repeated single sample generation
#     generation.generate_repeated_single_sample(4,
#                                                plot_samples=True,
#                                                show_plots=False,
#                                                save_sample_plots=True,
#                                                )

#     # # Launch the generation
#     # generation_sbgm(args)

# # def args_from_checkpoint_filename(filename: str) -> Namespace:
# #         """
# #         Extract key argument settings from a checkpoint filename.
# #         Assumes filename of the form:
# #         sbgm_PrcpStripped__HR_prcp_DANRA__SIZE_128x128__LR_prcp_temp_ERA5__LOSS_sdfweighted__HEADS_4__TIMESTEPS_1000.pth.tar
# #         """
# #         # Remove file extension
# #         basename = os.path.basename(filename).replace('.pth.tar', '')
# #         parts = basename.split('__')

# #         args_dict = {}

# #         for part in parts:
# #             if part.startswith("HR_"):
# #                 _, hr_var, hr_model = part.split('_')
# #                 args_dict["hr_var"] = hr_var
# #                 args_dict["hr_model"] = hr_model
# #             elif part.startswith("SIZE_"):
# #                 size_str = part.replace("SIZE_", "")
# #                 args_dict["hr_data_size"] = [int(x) for x in size_str.split('x')]
# #             elif part.startswith("LR_"):
# #                 _, *lr_vars, lr_model = part.split('_')
# #                 args_dict["lr_conditions"] = lr_vars
# #                 args_dict["lr_model"] = lr_model
# #             elif part.startswith("LOSS_"):
# #                 args_dict["loss_type"] = part.replace("LOSS_", "")
# #             elif part.startswith("HEADS_"):
# #                 args_dict["num_heads"] = int(part.replace("HEADS_", ""))
# #             elif part.startswith("TIMESTEPS_"):
# #                 args_dict["n_timesteps"] = int(part.replace("TIMESTEPS_", ""))

# #         return Namespace(**args_dict)

# class GenerationSBGM():
#     '''
#         Class to generate samples from trained SBGM model.
#         Can generate samples as:
#         - Single samples
#         - Multiple samples
#         - Repeated samples of single sample

#     '''
#     def __init__(self, args):
#         '''
#             Constructor for the GenerationSBGM class.
#             Args:
#                 args: Namespace, arguments from the command line.
#         '''
        
        
        
        
#         ##################################
#         #                                #
#         # SETUP WITH NECESSARY ARGUMENTS #
#         #                                #
#         ##################################

#         self.hr_var = args.hr_var
#         self.lr_vars = args.lr_conditions

#         self.hr_model = args.hr_model
#         self.lr_model = args.lr_model

#         # Define default LR colormaps and extra colormaps for geo variables
#         cmap_prcp = 'inferno'
#         cmap_temp = 'plasma'
#         self.lr_cmap_dict = {"prcp": cmap_prcp, "temp": cmap_temp}
#         self.extra_cmap_dict = {"topo": "terrain", "lsm": "binary", "sdf": "coolwarm"}

#         if self.hr_var == 'temp':
#             self.cmap_name = 'plasma'
#             self.hr_units = r'$^\circ$C'
#         elif self.hr_var == 'prcp':
#             self.cmap_name = 'inferno'
#             self.hr_units = 'mm'
#         else:
#             self.hr_units = 'Unknown'
        
#         # Set units for LR conditions
#         prcp_units = 'mm'
#         temp_units = r'$^\circ$C'
#         self.lr_units = []
#         for cond in self.lr_vars:
#             if cond == 'prcp':
#                 self.lr_units.append(prcp_units)
#             elif cond == 'temp':
#                 self.lr_units.append(temp_units)
#             else:
#                 self.lr_units.append('Unknown')

#         # Set seed for reproducibility
#         seed = 100
#         torch.manual_seed(seed)
#         np.random.seed(seed)

#         # Set torch to deterministic mode, meaning that the same input will always produce the same output
#         torch.backends.cudnn.deterministic = False
#         # Set torch to benchmark mode, meaning that the best algorithm will be chosen for the input
#         torch.backends.cudnn.benchmark = True

#         # Define the device to use
#         if args.device is not None:
#             self.device = args.device
#         else:
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         print(f'\nUsing device: {self.device}')



#         ##############################
#         # SETUP DATA HYPERPARAMETERS #
#         ##############################

#         # Path to data
#         self.path_data = args.path_data

#         # Set image dimensions (if None, use default 128x128)
#         self.hr_data_size = tuple(args.hr_data_size) if args.hr_data_size is not None else None
#         if self.hr_data_size is None:
#             self.hr_data_size = (128, 128)
#         self.lr_data_size = tuple(args.lr_data_size) if args.lr_data_size is not None else None

#         if self.lr_data_size == None:
#             self.lr_data_size_use = self.hr_data_size
#         else:
#             self.lr_data_size_use = self.lr_data_size

#         # Check if resize factor is set and print sizes    
#         if args.resize_factor > 1:
#             self.hr_data_size_use = (self.hr_data_size[0]//args.resize_factor, self.hr_data_size[1]//args.resize_factor)
#             self.lr_data_size_use = (self.lr_data_size_use[0]//args.resize_factor, self.lr_data_size_use[1]//args.resize_factor)
#         else:
#             self.hr_data_size_use = self.hr_data_size
#             self.lr_data_size_use = self.lr_data_size_use

#         print(f'\n\nHR data size OG: {self.hr_data_size}')
#         print(f'\tHR data size reduced: ({self.hr_data_size_use[0]}, {self.hr_data_size_use[1]})')
#         print(f'LR data size OG: {self.lr_data_size}')
#         print(f'\tLR data size reduced: ({self.lr_data_size_use[0]}, {self.lr_data_size_use[1]})')

#         # Set full domain size
#         self.full_domain_dims = tuple(args.full_domain_dims) if args.full_domain_dims is not None else None

#         # Resize factor (for training on smaller images)
#         if args.resize_factor is not None:
#             self.resize_factor = args.resize_factor
#         else:
#             self.resize_factor = 1

#         # Set scaling and matching options
#         self.scaling = args.scaling
#         self.show_both_orig_scaled = args.show_both_orig_scaled
#         self.force_matching_scale = args.force_matching_scale
#         self.transform_back_bf_plot = args.transform_back_bf_plot

#         # Set up scaling methods
#         self.hr_scaling_method = args.hr_scaling_method
#         self.hr_scaling_params = ast.literal_eval(args.hr_scaling_params)
#         self.lr_scaling_methods = args.lr_scaling_methods
#         self.lr_scaling_params = [ast.literal_eval(param) for param in args.lr_scaling_params]

#         # Set up back transforms for visual inspection
#         self.back_transforms = build_back_transforms(
#             hr_var               = self.hr_var,
#             hr_scaling_method    = self.hr_scaling_method,
#             hr_scaling_params    = self.hr_scaling_params,
#             lr_vars              = self.lr_vars,
#             lr_scaling_methods   = self.lr_scaling_methods,
#             lr_scaling_params    = self.lr_scaling_params,
#         )

#         # Setup geo variables
#         self.sample_w_sdf = args.sample_w_sdf
#         if self.sample_w_sdf:
#             print('\nSDF weighted loss enabled. Setting lsm and topo to True.\n')
#             self.sample_w_geo = True
#         else:
#             self.sample_w_geo = args.sample_w_geo

#         if self.sample_w_geo:
#             self.geo_variables = ['lsm', 'topo']
#             data_dir_lsm = args.path_data + 'data_lsm/truth_fullDomain/lsm_full.npz'
#             data_dir_topo = args.path_data + 'data_topo/truth_fullDomain/topo_full.npz'
#             self.data_lsm = np.flipud(np.load(data_dir_lsm)['data'])
#             self.data_topo = np.flipud(np.load(data_dir_topo)['data'])
#             if self.scaling:
#                 if args.topo_min is None or args.topo_max is None:
#                     topo_min, topo_max = np.min(self.data_topo), np.max(self.data_topo)
#                 else:
#                     topo_min, topo_max = args.topo_min, args.topo_max
#                 if args.norm_min is None or args.norm_max is None:
#                     norm_min, norm_max = np.min(self.data_lsm), np.max(self.data_lsm)
#                 else:
#                     norm_min, norm_max = args.norm_min, args.norm_max
#                 OldRange = (topo_max - topo_min)
#                 NewRange = (norm_max - norm_min)
#                 self.data_topo = ((self.data_topo - topo_min) * NewRange / OldRange) + norm_min
#         else:
#             self.geo_variables = None
#             self.data_lsm = None
#             self.data_topo = None

#         # Setup cutouts
#         self.sample_w_cutouts = args.sample_w_cutouts
#         # Set cutout domains, if None, use default (170, 350, 340, 520) (DK area with room for shuffle) 
#         self.cutout_domains = tuple(args.cutout_domains) if args.cutout_domains is not None else None
#         if self.cutout_domains is None:
#             self.cutout_domains = (170, 350, 340, 520)
#         self.lr_cutout_domains = tuple(args.lr_cutout_domains) if args.lr_cutout_domains is not None else None
#         if self.lr_cutout_domains is None:
#             self.lr_cutout_domains = (170, 350, 340, 520)
#         # Setup conditional seasons (classification)
#         self.sample_w_cond_season = args.sample_w_cond_season
#         if self.sample_w_cond_season:
#             self.n_seasons = args.n_seasons
#         else:
#             self.n_seasons = None



#         # Define model parameters
#         self.input_channels = len(self.lr_vars) # equal to number of LR variables
#         self.output_channels = 1 # equal to number of HR variables
#         self.last_fmap_channels = args.last_fmap_channels
#         self.time_embedding = args.time_embedding
#         self.num_heads = args.num_heads

#         if self.lr_vars is not None:
#             self.sample_w_cond_img = True
#         else:
#             self.sample_w_cond_img = False


#         # Define hyper parameters
#         self.learning_rate = args.learning_rate
#         self.min_lr = args.min_lr
#         self.weight_decay = args.weight_decay
#         self.epochs = args.epochs



#         self.config_name = args.config_name
#         lr_vars_str = '_'.join(self.lr_vars)
#         self.save_str = (
#             f"{args.config_name}__"
#             f"HR_{self.hr_var}_{args.hr_model}__"
#             f"SIZE_{self.hr_data_size_use[0]}x{self.hr_data_size_use[1]}__"
#             f"LR_{lr_vars_str}_{args.lr_model}__"
#             f"LOSS_{args.loss_type}__"
#             f"HEADS_{self.num_heads}__"
#             f"TIMESTEPS_{args.n_timesteps}"
#         )

#         self.PATH_SAVE = args.path_save
#         self.PATH_GENERATED_SAMPLES = self.PATH_SAVE + 'evaluation/generated_samples/' + self.save_str + '/'

#         # Check if the directory exists, if not create it
#         if not os.path.exists(self.PATH_GENERATED_SAMPLES):
#             os.makedirs(self.PATH_GENERATED_SAMPLES)
#             print(f'\n\nCreated directory: {self.PATH_GENERATED_SAMPLES}')

#         # Setup specific names for saving
#         self.lr_vars_str = '_'.join(self.lr_vars)
#         self.save_str = (
#             f"{args.config_name}__"
#             f"HR_{self.hr_var}_{args.hr_model}__"
#             f"SIZE_{self.hr_data_size_use[0]}x{self.hr_data_size_use[1]}__"
#             f"LR_{self.lr_vars_str}_{args.lr_model}__"
#             f"LOSS_{args.loss_type}__"
#             f"HEADS_{self.num_heads}__"
#             f"TIMESTEPS_{args.n_timesteps}"
#         )

#         # Set the year range for generation
#         self.gen_years = args.gen_years
#         self.year_start = self.gen_years[0]
#         self.year_end = self.gen_years[1]
        
#         ###############################
#         # SETUP MODEL HYPERPARAMETERS #
#         ###############################

#         self.loss_type = args.loss_type
#         if self.loss_type == 'sdfweighted':
#             self.sdf_weighted_loss = True
#         else:
#             self.sdf_weighted_loss = False



#         self.optimizer = args.optimizer
        
#         # if args.optimizer == 'adam':
#         #     self.optimizer = torch.optim.AdamW(score_model.parameters(),
#         #                                 lr=learning_rate,
#         #                                 weight_decay=weight_decay)
#         # elif args.optimizer == 'adamw':
#         #     self.optimizer = torch.optim.AdamW(score_model.parameters(),
#         #                                 lr=learning_rate,
#         #                                 weight_decay=weight_decay)
#         # elif args.optimizer == 'sgd':
#         #     self.optimizer = torch.optim.SGD(score_model.parameters(),
#         #                                 lr=learning_rate,
#         #                                 weight_decay=weight_decay)

#         self.num_workers = args.num_workers
        
#         # # Define learning rate scheduler
#         # if args.lr_scheduler is not None:
#         #     self.lr_scheduler_params = args.lr_scheduler_params
#         #     self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
#         #                                                             'min',
#         #                                                             factor = self.lr_scheduler_params['factor'],
#         #                                                             patience = self.lr_scheduler_params['patience'],
#         #                                                             threshold = self.lr_scheduler_params['threshold'],
#         #                                                             min_lr = self.min_lr
#         #                                                             )
                

#         # Define diffusion hyperparameters
#         self.n_timesteps = args.n_timesteps
#         # noise_variance = args.noise_variance # Defined above

#         # Set the path to the model checkpoint 
#         self.path_checkpoint = args.path_checkpoint

#         # Select folder with .nc/.npz files for generation
#         self.gen_dir_hr = build_data_path(self.path_data, self.hr_model, self.hr_var, self.full_domain_dims, 'eval', zarr_file=False)
#         print(f'HR {self.hr_var} data path: {self.gen_dir_hr}')
        
#         # Loop over lr_vars and create the path for the zarr files
#         self.gen_dir_lrs = {}
#         for i, cond in enumerate(self.lr_vars):
#             print(f'Using LR data type: {self.lr_model} {cond} [{self.lr_units[self.lr_vars.index(cond)]}]')
#             # Check if cond is in extra_cmap_dict
#             self.gen_dir_lrs[cond] = build_data_path(self.path_data, self.lr_model, cond, self.full_domain_dims, 'eval', zarr_file=False)
#             print(f'LR {cond} data path: {self.gen_dir_lrs[cond]}')
            
#         # self.gen_dir_lr = build_data_path(self.path_data, self.lr_model, self.lr_vars, self.full_domain_dims, 'eval')


#     def setup_data_folder(self, n_gen_samples):
#         """
#             Method to set up temporary eval directories and create zarr files for n_gen_samples in the specified years.

#             1. Lists files in eval folder and filters them based on the specified years.
#             2. Finds common dates across all conditions - if not enough common dates, reduces n_gen_samples.
#             3. Samples n_gen_samples dates and selects files. Then checks seasonal distribution.
#             4. Cretes (or empties) temporary eval directories for each condition.
#             5. Copies selected files to temporary eval directories.
#             6. Defines zarr path 
#             7. Converts npz files in temporary eval directories to zarr format.
#         """

#         self.n_gen_samples = n_gen_samples
#         self.cache_size = self.n_gen_samples
#         self.year_start = self.gen_years[0]
#         self.year_end = self.gen_years[1]


#         # --- 1a List and filter HR files ---
#         gen_files_hr = os.listdir(self.gen_dir_hr) # gen_dir_hr points to the full eval folder
#         n_gen_files_hr_all = len(gen_files_hr)
#         gen_files_hr = [f for f in gen_files_hr if f != '.DS_Store' and self.year_start <= int(f[-12:-8]) <= self.year_end]
#         n_gen_files_hr_filtered = len(gen_files_hr)
#         print(f'\n\nNumber of files in HR generation dataset (full -> in years): {n_gen_files_hr_all} -> {n_gen_files_hr_filtered}')

#         if n_gen_files_hr_filtered == 0:
#             print(f'\n\nNo HR files found for the years {self.year_start}-{self.year_end}. Exiting...')
#             exit(1)

#         # --- 1b List and filter LR files ---

#         gen_files_lrs = {}

#         for cond in self.lr_vars:
#             files = os.listdir(self.gen_dir_lrs[cond])
#             n_gen_files_lr_all = len(files)
#             gen_files_lrs[cond] = [f for f in files if f != '.DS_Store' and self.year_start <= int(f[-12:-8]) <= self.year_end]
#             n_gen_files_lr_filtered = len(gen_files_lrs[cond])
#             print(f'{cond}: {n_gen_files_lr_all} -> {n_gen_files_lr_filtered} files in selected years')
#             if n_gen_files_lr_filtered == 0:
#                 print(f'\n\nNo LR files found for the years {self.year_start}-{self.year_end}. Exiting...')
#                 exit(1)

#         # --- 2. Find common dates (HR and all LR conds) ---
#         hr_dates = set(int(f[-12:-4]) for f in gen_files_hr)
#         common_dates = hr_dates.copy()
#         for cond in self.lr_vars:
#             cond_dates = set(int(f[-12:-4]) for f in gen_files_lrs[cond])
#             common_dates &= cond_dates
#         common_dates = sorted(list(common_dates))
        
#         if len(common_dates) == 0:
#             print(f'\n\nNo common dates found for the years {self.year_start}-{self.year_end}. Exiting...')
#             exit(1)
#         if self.n_gen_samples > len(common_dates):
#             print(f'Not enough common dates. Reducing from {self.n_gen_samples} to {len(common_dates)}.')
#             self.n_gen_samples = len(common_dates)

#         # --- 3a Sample dates and select files ---
#         gen_dates = sorted(np.random.choice(common_dates, size=self.n_gen_samples, replace=False))
#         print(f'\nSelected dates: {gen_dates}')

#         gen_files_hr = sorted([f for f in gen_files_hr if int(f[-12:-4]) in gen_dates], key=lambda x: int(x[-12:-4]))
#         gen_files_lrs = {cond: sorted([f for f in gen_files_lrs[cond] if int(f[-12:-4]) in gen_dates], key=lambda x: int(x[-12:-4]))
#                         for cond in self.lr_vars}

#         # --- 3b Check seasonal distribution ---
#         print('\nChecking seasonal distribution...')
#         seasons = {'Winter': ['12', '01', '02'], 'Spring': ['03', '04', '05'], 'Summer': ['06', '07', '08'], 'Autumn': ['09', '10', '11']}
#         for season, months in seasons.items():
#             count = sum(str(date)[-4:-2] in months for date in gen_dates)
#             print(f'{season}: {count}')

#         # --- 4a Setup tmp eval dirs ---
#         self.tmp_eval_hr = build_data_path(self.path_data, self.hr_model, self.hr_var, self.full_domain_dims, 'tmp_eval', zarr_file=False)
#         self.tmp_eval_lrs = {cond: build_data_path(self.path_data, self.lr_model, cond, self.full_domain_dims, 'tmp_eval', zarr_file=False)
#                             for cond in self.lr_vars}
        
#         # --- 4b Create or empty tmp eval dirs ---
#         os.makedirs(self.tmp_eval_hr, exist_ok=True)
#         for f in os.listdir(self.tmp_eval_hr):
#             os.remove(os.path.join(self.tmp_eval_hr, f))

#         for cond in self.lr_vars:
#             os.makedirs(self.tmp_eval_lrs[cond], exist_ok=True)
#             for f in os.listdir(self.tmp_eval_lrs[cond]):
#                 os.remove(os.path.join(self.tmp_eval_lrs[cond], f))

#         # --- 5. Copy selected files ---
#         print('\nCopying selected files to tmp eval directories...')

#         for i in range(self.n_gen_samples):
#             # Copy HR files
#             src = os.path.join(self.gen_dir_hr, gen_files_hr[i])
#             dst = os.path.join(self.tmp_eval_hr, gen_files_hr[i])
#             os.system(f'cp {src} {dst}')
#             # Copy LR files
#             for cond in self.lr_vars:
#                 src = os.path.join(self.gen_dir_lrs[cond], gen_files_lrs[cond][i])
#                 dst = os.path.join(self.tmp_eval_lrs[cond], gen_files_lrs[cond][i])
#                 os.system(f'cp {src} {dst}')
#         print(f'\nCopied {self.n_gen_samples} files to tmp eval directories to:')
#         print(f'\tHR tmp eval dir: {self.tmp_eval_hr}')
#         for cond in self.lr_vars:
#             print(f'\tLR tmp eval dir: {self.tmp_eval_lrs[cond]}')

#         # --- 6. Define zarr paths ---
#         self.eval_hr_zarr = build_data_path(self.path_data, self.hr_model, self.hr_var, self.full_domain_dims, 'eval', zarr_file=True)
#         self.eval_lrs_zarr = {cond: build_data_path(self.path_data, self.lr_model, cond, self.full_domain_dims, 'eval', zarr_file=True)
#                             for cond in self.lr_vars}

#         # --- 7. Convert npz files to zarr ---
#         print('\nConverting npz files to zarr format...')
#         # Convert HR files with convert_npz_to_zarr
#         convert_npz_to_zarr(self.tmp_eval_hr, self.eval_hr_zarr)
#         print(f'\nConverted {self.n_gen_samples} HR files to zarr format in {self.eval_hr_zarr}')
#         # Convert LR files with convert_npz_to_zarr
#         for cond in self.lr_vars:
#             convert_npz_to_zarr(self.tmp_eval_lrs[cond], self.eval_lrs_zarr[cond])
#             print(f'\nConverted {self.n_gen_samples} LR files to zarr format in {self.eval_lrs_zarr[cond]}')

#         # Point to the variable names used later on
#         self.hr_variable_dir_zarr = self.eval_hr_zarr
#         self.lr_cond_dirs_zarr = {cond: self.eval_lrs_zarr[cond] for cond in self.lr_vars}
        

#         print(f'\n\nFinished setting up data folder for generation.\n\n')


#     def setup_data_loader(self, n_gen_samples, save_sample_fig=True):
#         '''
#             Method to setup the data loader for generation.
#         '''
#         # Call the setup_data_folder method to setup the data
#         self.setup_data_folder(n_gen_samples)

#         print(f'\n\nSetting up data loader for {n_gen_samples} samples...')
#         print(f'HR variable: {self.hr_var} from zarr file: {self.hr_variable_dir_zarr}')
#         print(f'LR variables: {self.lr_vars} from zarr files: {self.lr_cond_dirs_zarr}')
#         # Create the dataset
#         eval_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(
#                                 hr_variable_dir_zarr=self.hr_variable_dir_zarr,
#                                 hr_data_size=self.hr_data_size_use,
#                                 n_samples=n_gen_samples,
#                                 cache_size=self.cache_size,
#                                 hr_variable=self.hr_var,
#                                 hr_model=self.hr_model,
#                                 hr_scaling_method=self.hr_scaling_method,
#                                 hr_scaling_params=self.hr_scaling_params,
#                                 lr_conditions=self.lr_vars,
#                                 lr_model=self.lr_model,
#                                 lr_scaling_methods=self.lr_scaling_methods,
#                                 lr_scaling_params=self.lr_scaling_params,
#                                 lr_cond_dirs_zarr=self.lr_cond_dirs_zarr,
#                                 geo_variables=self.geo_variables,
#                                 lsm_full_domain=self.data_lsm,
#                                 topo_full_domain=self.data_topo,
#                                 shuffle=True,
#                                 cutouts=self.sample_w_cutouts,
#                                 cutout_domains=self.cutout_domains if self.sample_w_cutouts else None, 
#                                 n_samples_w_cutouts=n_gen_samples,
#                                 sdf_weighted_loss=self.sample_w_sdf,
#                                 scale=self.scaling,
#                                 save_original=self.show_both_orig_scaled,
#                                 conditional_seasons=self.sample_w_cond_season,
#                                 n_classes=self.n_seasons,
#                                 lr_data_size=tuple(self.lr_data_size_use) if self.lr_data_size_use is not None else None,
#                                 lr_cutout_domains=tuple(self.lr_cutout_domains) if self.lr_cutout_domains is not None else None,
#                                 resize_factor=self.resize_factor,
#                                 )    
        
#         # Make a dataloader with batch size equal to n
#         self.eval_dataloader = DataLoader(eval_dataset,
#                                           batch_size=n_gen_samples,
#                                           shuffle=False,
#                                           num_workers=self.num_workers
#                                           )

#         # Check the data by plotting the first sample
#         samples = next(iter(self.eval_dataloader))
#         if save_sample_fig:
#             # Make an empty dict to hold the btransformed samples
#             samples_bt = {}
#             # Loop over the samples and back transform them
#             for key in samples.keys():
#                 if key not in samples or samples[key] is None:
#                     continue
#                 # Print key and shape for debugging
#                 print(f"Key: {key}")
#                 try:
#                     print(f"Shape: {samples[key].shape}")
#                 except:
#                     print(f"Shape: {len(samples[key])}")
#                 if torch.is_tensor(samples[key]):
#                     img = samples[key].detach().cpu().numpy()
#                 if key in self.back_transforms:
#                     img = self.back_transforms[key](img)
#                 else:
#                     img = samples[key]
#                 samples_bt[key] = img
#             # Plot the back transformed samples
#             fig, ax = plot_samples(samples_bt,
#                                 hr_model=self.hr_model,
#                                 hr_units=self.hr_units,
#                                 lr_model=self.lr_model,
#                                 lr_units=self.lr_units,
#                                 var=self.hr_var,
#                                 extra_keys=['lsm', 'topo', 'sdf'],
#                                 hr_cmap=self.cmap_name,
#                                 lr_cmap_dict=self.lr_cmap_dict,
#                                 extra_cmap_dict=self.extra_cmap_dict
#                                 )
#             # Save figure
#             fig.savefig(os.path.join(self.PATH_GENERATED_SAMPLES, 'sample_plot_backtransformed.png'), dpi=300)
#             # Use plot_sample() to plot the first sample
#             fig, ax = plot_samples(samples,
#                                 hr_model=self.hr_model,
#                                 hr_units=self.hr_units,
#                                 lr_model=self.lr_model,
#                                 lr_units=self.lr_units,
#                                 var=self.hr_var,
#                                 extra_keys=['lsm', 'topo', 'sdf'],
#                                 hr_cmap=self.cmap_name,
#                                 lr_cmap_dict=self.lr_cmap_dict,
#                                 extra_cmap_dict=self.extra_cmap_dict
#                                 )
#             #

#             # Save figure
#             fig.savefig(os.path.join(self.PATH_GENERATED_SAMPLES, 'sample_plot.png'), dpi=300)
#             print(f'\nSaved sample plot to {os.path.join(self.PATH_GENERATED_SAMPLES, "sample_plot.png")}')
#         print(f'\n\nFinished setting up data loader for {n_gen_samples} samples...\n\n')


#     def setup_model(self):
#         '''
#             Method to setup the SBGM model for generation.
#         '''
#         ###################################
#         #                                 #
#         # SETTING UP MODEL FOR GENERATION #
#         #                                 #   
#         ###################################        


#         # Define the encoder and decoder from modules_DANRA_downscaling.py
#         encoder = Encoder(self.input_channels, 
#                             self.time_embedding,
#                             cond_on_lsm=self.sample_w_geo,
#                             cond_on_topo=self.sample_w_geo,
#                             cond_on_img=self.sample_w_cond_img, 
#                             cond_img_dim=(len(self.lr_vars), self.lr_data_size_use[0], self.lr_data_size_use[1]), 
#                             block_layers=[2, 2, 2, 2], 
#                             num_classes=self.n_seasons,
#                             n_heads=self.num_heads
#                             )
#         decoder = Decoder(self.last_fmap_channels, 
#                             self.output_channels, 
#                             self.time_embedding, 
#                             n_heads=self.num_heads
#                             )
#         self.score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn, encoder=encoder, decoder=decoder)
#         self.score_model = self.score_model.to(self.device)


#         # Define the optimizer
#         if self.optimizer == 'adam':
#             optimizer = torch.optim.AdamW(self.score_model.parameters(),
#                                         lr=self.learning_rate,
#                                         weight_decay=self.weight_decay)
#         elif self.optimizer == 'adamw':
#             optimizer = torch.optim.AdamW(self.score_model.parameters(),
#                                         lr=self.learning_rate,
#                                         weight_decay=self.weight_decay)
#         elif self.optimizer == 'sgd':
#             optimizer = torch.optim.SGD(self.score_model.parameters(),
#                                         lr=self.learning_rate,
#                                         weight_decay=self.weight_decay)

#         # Define training pipeline
#         self.pipeline = TrainingPipeline_general(self.score_model,
#                                             loss_fn,
#                                             marginal_prob_std_fn,
#                                             optimizer,
#                                             self.device,
#                                             weight_init=True,
#                                             sdf_weighted_loss=self.sdf_weighted_loss
#                                             )

        
#     def load_checkpoint(self):
#         '''
#             Method to load the checkpoint of the trained SBGM model.
#         '''

#         # Call the setup_model method to setup the model
#         self.setup_model()

#         # Define path to checkpoint
#         checkpoint_dir = self.path_checkpoint

#         name_checkpoint = self.save_str + '.pth.tar'

#         checkpoint_path = os.path.join(checkpoint_dir, name_checkpoint)

#         # Check if checkpoint exists
#         if not os.path.exists(checkpoint_path):
#             print(f'\n\nCheckpoint {os.path.join(checkpoint_dir, name_checkpoint)} does not exist, exiting...')
#             exit(1)

#         # Load model checkpoint
#         map_location=torch.device('cpu')
#         best_model_state = torch.load(checkpoint_path, map_location=map_location)['network_params']

#         # Load best state model into model
#         print('\nLoading best model state from checkpoint: ')
#         print(checkpoint_path)
#         print('\n\n')

#         # Load the model state and set the model to evaluation mode
#         self.pipeline.model.load_state_dict(best_model_state)
#         self.pipeline.model.eval()





#     def generate_multiple_samples(self,
#                                   n_gen_samples,
#                                   sampler='pc_sampler',
#                                   plot_samples=False,
#                                   show_plots=False,
#                                   save_sample_plots=False,
#                                   save_sample_plots_dir=None,
#                                   ):
#         '''
#             Method to generate multiple samples from the trained SBGM model.
#         '''
        
#         # Call the setup_data_loader method to setup the data loader
#         self.setup_data_loader(n_gen_samples)

#         # Call the load_checkpoint method to load the model checkpoint
#         self.load_checkpoint()

        
#         print(f"Generating {n_gen_samples} samples from the trained SBGM model...")


#         for idx, samples in tqdm.tqdm(enumerate(self.eval_dataloader), total=len(self.eval_dataloader), position=0):
#             sample_batch_size = n_gen_samples

#             # Define sampler to use
#             sampler_name = sampler # ['pc_sampler', 'Euler_Maruyama', 'ode_sampler']

#             if sampler_name == 'pc_sampler':
#                 sampler = pc_sampler
#             elif sampler_name == 'Euler_Maruyama':
#                 sampler = Euler_Maruyama_sampler
#             elif sampler_name == 'ode_sampler':
#                 sampler = ode_sampler
#             else:
#                 raise ValueError(f'Invalid sampler name: {sampler_name}. Please choose from: ["pc_sampler", "Euler_Maruyama", "ode_sampler"]')
            
#             # Extract the samples
#             eval_img, eval_season, eval_cond, eval_lsm_hr, eval_lsm, eval_topo, eval_sdf, eval_point_hr, eval_point_lr = extract_samples(samples, self.device)
            
            
#             # Generate images from model
#             generated_samples = sampler(score_model=self.score_model,
#                                         marginal_prob_std=marginal_prob_std_fn,
#                                         diffusion_coeff=diffusion_coeff_fn,
#                                         batch_size=sample_batch_size,
#                                         num_steps=self.n_timesteps,
#                                         device=self.device,
#                                         img_size=self.hr_data_size_use[0],
#                                         y=eval_season,
#                                         cond_img=eval_cond,
#                                         lsm_cond=eval_lsm,
#                                         topo_cond=eval_topo,
#                                         ).squeeze()
#             generated_samples = generated_samples.detach().cpu()

#             # Stop after first iteration, all samples are generated
#             break


#         # Plot the samples and conditionals, if requested
#         if plot_samples:

#             # Use plot_samples_and_generated to plot
#             fig, ax = plot_samples_and_generated(samples,
#                                                  generated_samples,
#                                                  hr_model=self.hr_model,
#                                                  hr_units=self.hr_units,
#                                                  lr_model=self.lr_model,
#                                                  lr_units=self.lr_units,
#                                                  var=self.hr_var,
#                                                  show_ocean=True,
#                                                  transform_back_bf_plot=self.transform_back_bf_plot,
#                                                  back_transforms=self.back_transforms,
#                                                  n_samples_threshold=n_gen_samples
#             )
#             fig.tight_layout()
#             if show_plots:
#                 plt.show()
#             if save_sample_plots:
#                 if save_sample_plots_dir is None:
#                     save_sample_plots_dir = self.PATH_GENERATED_SAMPLES
#                 print(f'\n\nSaving generated samples to {save_sample_plots_dir}...')
#                 fig.savefig(save_sample_plots_dir + f'generated_samples_{n_gen_samples}.png', dpi=300, bbox_inches='tight')
#                 print(f'\n\nGenerated samples saved to {save_sample_plots_dir}!')
                

#         # Save the generated and corresponding eval images
#         print(f'\n\nSaving generated images to {self.PATH_GENERATED_SAMPLES}...')
#         # Check if files are None - if so, do not save that file
#         if eval_img is not None:
#             print(f'Saving generated and evaluation images...')
#             # If transform_back_bf_plot is True, use back transforms to transform the images back
#             if self.transform_back_bf_plot:
#                 print(f'Using back transforms to transform generated and evaluation images back...')
#                 generated_samples = self.back_transforms['hr'](generated_samples)
#                 eval_img = self.back_transforms['hr'](eval_img)
#             print(f'Saving generated samples to {self.PATH_GENERATED_SAMPLES} with name gen_samples_{n_gen_samples}')
#             np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'gen_samples_' + str(n_gen_samples), generated_samples.cpu().numpy())
#             np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'eval_samples_' + str(n_gen_samples), eval_img.cpu().numpy())
#         if eval_lsm is not None:
#             print(f'Saving LSM samples...')
#             np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'lsm_samples_' + str(n_gen_samples), eval_lsm.cpu().numpy())
#         if eval_lsm_hr is not None:
#             print(f'Saving LSM HR samples...')
#             np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'lsm_hr_samples_' + str(n_gen_samples), eval_lsm_hr.cpu().numpy())
#         if eval_topo is not None:
#             print(f'Saving Topo samples...')
#             np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'topo_samples_' + str(n_gen_samples), eval_topo.cpu().numpy())
#         if eval_sdf is not None:
#             print(f'Saving SDF samples...')
#             np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'sdf_samples_' + str(n_gen_samples), eval_sdf.cpu().numpy())
#         if eval_cond is not None:
#             print(f'Saving conditional samples...')
#             # If transform_back_bf_plot is True, use back transforms to transform the images back
#             if self.transform_back_bf_plot:
#                 print(f'Using back transforms to transform conditional samples back...')
#                 for key in eval_cond.keys():
#                     eval_cond[key] = self.back_transforms[key](eval_cond[key])
#             np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'cond_samples_' + str(n_gen_samples), eval_cond.cpu().numpy())
#         if eval_season is not None:
#             print(f'Saving seasonal samples...')
#             np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'season_samples_' + str(n_gen_samples), eval_season.cpu().numpy())
#         # if eval_point_hr is not None:
#         #     np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'point_hr_samples_' + str(n_gen_samples), eval_point_hr.cpu().numpy())
#         # if eval_point_lr is not None:
#         #     np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'point_lr_samples_' + str(n_gen_samples), eval_point_lr.cpu().numpy())


#         print(f'\n\n{n_gen_samples} generated samples saved to {self.PATH_GENERATED_SAMPLES}!')


#     def generate_single_sample(self,
#                                sampler='pc_sampler',
#                                idx=0,
#                                date=None,
#                                plot_samples=False,
#                                show_plots=False,
#                                save_sample_plots=False,
#                                save_sample_plots_dir=None,
#                                ):
#         '''
#             Method to generate a single sample from the trained SBGM model.
#             So far, only generates a single sample from the first batch of the data loader.
#             DEVELOPMENT NEEDED: Generate a single sample from a specific date/index in the gen_years list.
#                 Not straightforward: setup_data_folder needs to be modified to be able to select specific dates.
#         '''

#         # Call the setup_data_loader method to setup the data loader
#         self.setup_data_loader(1)

#         # Call the load_checkpoint method to load the model checkpoint
#         self.load_checkpoint()

#         print(f"Generating single (random) sample from the trained SBGM model...")

#         for idx, samples in tqdm.tqdm(enumerate(self.eval_dataloader), total=len(self.eval_dataloader), position=0):
#             sample_batch_size = 1

#             # Define sampler to use
#             sampler_name = sampler

#             if sampler_name == 'pc_sampler':
#                 sampler = pc_sampler
#             elif sampler_name == 'Euler_Maruyama':
#                 sampler = Euler_Maruyama_sampler
#             elif sampler_name == 'ode_sampler':
#                 sampler = ode_sampler
#             else:
#                 raise ValueError(f'Invalid sampler name: {sampler_name}. Please choose from: ["pc_sampler", "Euler_Maruyama", "ode_sampler"]')
            
#             # Extract the sample
#             eval_img, eval_season, eval_cond, eval_lsm_hr, eval_lsm, eval_topo, eval_sdf, eval_point_hr, eval_point_lr = extract_samples(samples, self.device)
            
#             # Generate images from model
#             generated_samples = sampler(score_model=self.score_model,
#                                         marginal_prob_std=marginal_prob_std_fn,
#                                         diffusion_coeff=diffusion_coeff_fn,
#                                         batch_size=sample_batch_size,
#                                         num_steps=self.n_timesteps,
#                                         device=self.device,
#                                         img_size=self.hr_data_size_use[0],
#                                         y=eval_season,
#                                         cond_img=eval_cond,
#                                         lsm_cond=eval_lsm,
#                                         topo_cond=eval_topo,
#                                         ).squeeze()
#             generated_samples = generated_samples.detach().cpu()

#             # Stop after first iteration, all samples are generated
#             break

#                 # Plot the samples and conditionals, if requested
#         if plot_samples:
#             # Print keys and shapes of the samples
#             print(f'Generated samples:')
#             for key, value in samples.items():
#                 try:
#                     print(f'{key}: {value.shape}')
#                 except:
#                     print(f'{key}: {value}')
#             print(f'Generated samples shape: {generated_samples.shape}')
#             # Use plot_samples_and_generated to plot
#             fig, ax = plot_samples_and_generated(samples,
#                                                  generated_samples,
#                                                  hr_model=self.hr_model,
#                                                  hr_units=self.hr_units,
#                                                  lr_model=self.lr_model,
#                                                  lr_units=self.lr_units,
#                                                  var=self.hr_var,
#                                                  show_ocean=True,
#                                                  hr_cmap = self.cmap_name,
#                                                  lr_cmap_dict = self.lr_cmap_dict,
#                                                  transform_back_bf_plot=self.transform_back_bf_plot,
#                                                  back_transforms=self.back_transforms,
#                                                  n_samples_threshold=1,
#                                                 #  extra_keys = ['topo', 'lsm', 'sdf'],
#                                                 #  extra_cmap_dict = extra_cmap_dict
#             )
#             fig.tight_layout()
#             if show_plots:
#                 plt.show()
#             if save_sample_plots:
#                 if save_sample_plots_dir is None:
#                     save_sample_plots_dir = self.PATH_GENERATED_SAMPLES
#                 print(f'\n\nSaving generated samples to {save_sample_plots_dir}...')
#                 fig.savefig(save_sample_plots_dir + 'generated_single_sample.png', dpi=300, bbox_inches='tight')
#                 print(f'\n\nGenerated samples saved to {save_sample_plots_dir}!')


#         # Save the generated and corresponding eval images
#         print(f'\n\nSaving generated images to {self.PATH_GENERATED_SAMPLES}...')
#         # Check if files are None - if so, do not save that file
#         if eval_img is not None:
#             print(f'Saving generated and evaluation images...')
#             # If transform_back_bf_plot is True, use back transforms to transform the images back
#             if self.transform_back_bf_plot:
#                 print(f'Back transforming generated and evaluation images...')
#                 generated_samples = self.back_transforms['hr'](generated_samples)
#                 eval_img = self.back_transforms['hr'](eval_img)

#             np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'gen_singleSample', generated_samples.cpu().numpy())
#             np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'eval_singleSample', eval_img.cpu().numpy())
#         if eval_lsm is not None:
#             print(f'Saving LSM samples...')
#             np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'lsm_singleSample', eval_lsm.cpu().numpy())
#         if eval_lsm_hr is not None:
#             print(f'Saving LSM HR samples...')
#             np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'lsm_hr_singleSample', eval_lsm_hr.cpu().numpy())
#         if eval_topo is not None:
#             print(f'Saving Topo samples...')
#             np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'topo_singleSample', eval_topo.cpu().numpy())
#         if eval_sdf is not None:
#             print(f'Saving SDF samples...')
#             np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'sdf_singleSample', eval_sdf.cpu().numpy())
#         if eval_cond is not None:
#             print(f'Saving conditional samples...')
#             # If transform_back_bf_plot is True, use back transforms to transform the images back
#             if self.transform_back_bf_plot:
#                 print(f'Back transforming conditional samples...')
#                 for key in eval_cond.keys():
#                     eval_cond[key] = self.back_transforms[key](eval_cond[key])
#             np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'cond_singleSample', eval_cond.cpu().numpy())
#         if eval_season is not None:
#             print(f'Saving seasonal samples...')
#             np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'season_singleSample', eval_season.cpu().numpy())
#         # if eval_point_hr is not None:
#         #     np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'point_hr_singleSample', eval_point_hr.cpu().numpy())
#         # if eval_point_lr is not None:
#         #     np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'point_lr_singleSample', eval_point_lr.cpu().numpy())

#         print(f'\n\nSingle generated sample saved to {self.PATH_GENERATED_SAMPLES}!')



    

#     def generate_repeated_single_sample(self,
#                                         n_repeats,
#                                         sampler='pc_sampler',
#                                         idx=0,
#                                         date=None,
#                                         plot_samples=False,
#                                         show_plots=False,
#                                         save_sample_plots=False,
#                                         save_sample_plots_dir=None,
#                                         ):
#         '''
#             Method to generate multiple samples from the trained SBGM model based on the same conditionals.
#             Generates n_repeats samples from the first batch of the data loader, or from a specific date/index if implemented.
#         '''

#         # Call the setup_data_loader method to setup the data loader
#         self.setup_data_loader(1)

#         # Call the load_checkpoint method to load the model checkpoint
#         self.load_checkpoint()

#         # Get the only sample from the data loader
#         samples = next(iter(self.eval_dataloader))
#         sample_batch_size = 1

#         print(f"Generating {n_repeats} samples from the trained SBGM model with the same conditionals...")

#         if sampler == 'pc_sampler':
#             sampler_fn = pc_sampler
#         elif sampler == 'Euler_Maruyama':
#             sampler_fn = Euler_Maruyama_sampler
#         elif sampler == 'ode_sampler':
#             sampler_fn = ode_sampler
#         else:
#             raise ValueError(f'Invalid sampler name: {sampler_name}. Please choose from: ["pc_sampler", "Euler_Maruyama", "ode_sampler"]')

#         # Extract the sample
#         eval_img, eval_season, eval_cond, eval_lsm_hr, eval_lsm, eval_topo, eval_sdf, eval_point_hr, eval_point_lr = extract_samples(samples, self.device)

#         # Initialize a list to store generated samples
#         generated_samples_list = []

#         # Generate multiple samples
#         for _ in range(n_repeats):
#             # Generate images from model
#             generated_sample = sampler_fn(score_model=self.score_model,
#                                     marginal_prob_std=marginal_prob_std_fn,
#                                     diffusion_coeff=diffusion_coeff_fn,
#                                     batch_size=sample_batch_size,
#                                     num_steps=self.n_timesteps,
#                                     device=self.device,
#                                     img_size=self.hr_data_size_use[0],
#                                     y=eval_season,
#                                     cond_img=eval_cond,
#                                     lsm_cond=eval_lsm,
#                                     topo_cond=eval_topo,
#                                     ).squeeze()
#             generated_sample = generated_sample.detach().cpu()
#             generated_samples_list.append(generated_sample)

#         # Convert list of generated samples to a tensor
#         generated_samples = torch.stack(generated_samples_list)

#         # Repeat the input sample dictionary
#         repeated_dict = {}

#         # Loop through keys and values 
#         for key, val in samples.items():
#             if isinstance(val, torch.Tensor):
#                 if val.shape[0] == 1:
#                     # Repeat the tensor along the first dimension
#                     repeated_val = val.repeat(n_repeats, *[1 for _ in val.shape[1:]])
#                 else:
#                     raise ValueError(f"Expected batch size 1 for key {key}, but got shape {val.shape[0]}")
#                 repeated_dict[key] = repeated_val
#             else:
#                 # For lists (e.g. points), repeat the first element n_repeats times
#                 if isinstance(val, list) and len(val) == 4 and all(isinstance(v, torch.Tensor) and v.numel() == 1 for v in val):
#                     repeated_dict[key] = [val for _ in range(n_repeats)]
#                 else:
#                     repeated_dict[key] = val

#         # Plot the samples and conditionals, if requested
#         if plot_samples:

#             # Use plot_samples_and_generated to plot
#             fig, ax = plot_samples_and_generated(repeated_dict,
#                                                  generated_samples,
#                                                  hr_model=self.hr_model,
#                                                  hr_units=self.hr_units,
#                                                  lr_model=self.lr_model,
#                                                  lr_units=self.lr_units,
#                                                  var=self.hr_var,
#                                                  show_ocean=True,
#                                                  transform_back_bf_plot=self.transform_back_bf_plot,
#                                                  back_transforms=self.back_transforms,
#                                                  n_samples_threshold=n_repeats
#             )
#             fig.tight_layout()

#             if show_plots:
#                 plt.show()
#             if save_sample_plots:
#                 if save_sample_plots_dir is None:
#                     save_sample_plots_dir = self.PATH_GENERATED_SAMPLES
#                 print(f'\n\nSaving generated samples to {save_sample_plots_dir}...')
#                 fig.savefig(save_sample_plots_dir + f'generated_repeated_sample_{n_repeats}.png', dpi=300, bbox_inches='tight')
#                 print(f'\n\nGenerated samples saved to {save_sample_plots_dir}!')

#         # Save the generated and corresponding eval images
#         print(f'\n\nSaving {n_repeats} generated images to {self.PATH_GENERATED_SAMPLES}...')
#         # Check if files are None - if so, do not save that file
#         if eval_img is not None:
#             print(f'Saving generated and evaluation images...')
#             # If transform_back_bf_plot is True, use back transforms to transform the images back
#             if self.transform_back_bf_plot:
#                 print(f'Using back transforms to transform generated and evaluation images back...')
#                 generated_samples = self.back_transforms['hr'](generated_samples)
#                 eval_img = self.back_transforms['hr'](eval_img)
#             np.savez_compressed(self.PATH_GENERATED_SAMPLES + f'gen_repeatedSamples_{n_repeats}', generated_samples.cpu().numpy())
#             np.savez_compressed(self.PATH_GENERATED_SAMPLES + f'eval_repeatedSamples_{n_repeats}', eval_img.cpu().numpy())
#         if eval_lsm is not None:
#             print(f'Saving LSM samples...')
#             np.savez_compressed(self.PATH_GENERATED_SAMPLES + f'lsm_repeatedSamples_{n_repeats}', eval_lsm.cpu().numpy())
#         if eval_lsm_hr is not None:
#             print(f'Saving LSM HR samples...')
#             np.savez_compressed(self.PATH_GENERATED_SAMPLES + f'lsm_hr_repeatedSamples_{n_repeats}', eval_lsm_hr.cpu().numpy())
#         if eval_topo is not None:
#             print(f'Saving Topo samples...')
#             np.savez_compressed(self.PATH_GENERATED_SAMPLES + f'topo_repeatedSamples_{n_repeats}', eval_topo.cpu().numpy())
#         if eval_sdf is not None:
#             print(f'Saving SDF samples...')
#             np.savez_compressed(self.PATH_GENERATED_SAMPLES + f'sdf_repeatedSamples_{n_repeats}', eval_sdf.cpu().numpy())
#         if eval_cond is not None:
#             print(f'Saving conditional samples...')
#             # If transform_back_bf_plot is True, use back transforms to transform the images back
#             if self.transform_back_bf_plot:
#                 print(f'Using back transforms to transform conditional samples back...')
#                 for key in eval_cond.keys():
#                     eval_cond[key] = self.back_transforms[key](eval_cond[key])
#             np.savez_compressed(self.PATH_GENERATED_SAMPLES + f'cond_repeatedSamples_{n_repeats}', eval_cond.cpu().numpy())
#         if eval_season is not None:
#             print(f'Saving seasonal samples...')
#             np.savez_compressed(self.PATH_GENERATED_SAMPLES + f'season_repeatedSamples_{n_repeats}', eval_season.cpu().numpy())
#         # if eval_point_hr is not None:
#         #     np.savez_compressed(self.PATH_GENERATED_SAMPLES + f'point_hr_repeatedSamples_{n_repeats}', eval_point_hr.cpu().numpy())
#         # if eval_point_lr is not None:
#         #     np.savez_compressed(self.PATH_GENERATED_SAMPLES + f'point_lr_repeatedSamples_{n_repeats}', eval_point_lr.cpu().numpy())

#         print(f'\n\n{n_repeats} generated samples saved to {self.PATH_GENERATED_SAMPLES}!')


# if __name__ == '__main__':

#     generate_from_args()

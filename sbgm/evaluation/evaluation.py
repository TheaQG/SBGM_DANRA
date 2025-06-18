import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import *


def evaluate_from_args():
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

    evaluation_multiple = Evaluation(args, generated_sample_type='multiple', n_samples=args.n_gen_samples)

    # fig, axs = evaluation_multiple.plot_example_images(masked=False, plot_with_cond=True, plot_with_lsm=False, show_figs=False, n_samples=100, same_cbar=False, save_figs=True)
    
    evaluation_multiple.full_pixel_statistics(show_figs=False, save_figs=True, save_stats=False, save_path=None, n_samples=args.n_gen_samples)
    evaluation_multiple.spatial_statistics(show_figs=False, save_figs=True, save_stats=False, save_path=None, save_plot_path=args.path_save, n_samples=args.n_gen_samples)


    # evaluation_single = Evaluation(args, generated_sample_type='single')

    # fig, axs = evaluation_single.plot_example_images(masked=False, plot_with_cond=True, plot_with_lsm=False, show_figs=False,save_figs=True, n_samples=4, same_cbar=False)    

    # evaluation_repeated = Evaluation(args, generated_sample_type='repeated')

    # fig, axs = evaluation_repeated.plot_example_images(masked=False, plot_with_cond=True, plot_with_lsm=False, show_figs=False, n_samples=4, save_figs=True)




    print('Evaluation completed successfully!')

class Evaluation:
    '''
        Class to evaluate generated samples (saved in npz files) from the SBGM model.

        Evaluates the generated samples using the following metrics:
        - All pixel values distribution (across space and time), generated and eval images
        - RMSE and MAE for all pixels in all samples
    '''
    def __init__(self, args, generated_sample_type='repeated', n_samples=4):
        '''
            Setup the evaluation class.
            Get configuration parameters from args.
            Load samples from config path.
            
            - sample_type: What type of generated samples to evaluate. ['repeated', 'single', 'multiple'] 
        '''


        self.hr_var = args.hr_var
        self.hr_model = args.hr_model

        self.lr_model = args.lr_model
        self.lr_vars = args.lr_conditions
        self.lr_vars_str = '_'.join(self.lr_vars)

        self.hr_data_size = tuple(args.hr_data_size) if args.hr_data_size is not None else (128, 128)
        if self.hr_data_size is None:
            self.hr_data_size = (128, 128)
        
        self.loss_type = args.loss_type
        
        self.save_str = (
        f"{args.config_name}__"
        f"HR_{self.hr_var}_{self.hr_model}__"
        f"SIZE_{self.hr_data_size[0]}x{self.hr_data_size[1]}__"
        f"LR_{self.lr_vars_str}_{self.lr_model}__"
        f"LOSS_{self.loss_type}__"
        f"HEADS_{args.num_heads}__"
        f"TIMESTEPS_{args.n_timesteps}"
        )

        self.PATH_SAVE = args.path_save
        self.PATH_GENERATED_SAMPLES = self.PATH_SAVE + 'evaluation/generated_samples/' + self.save_str + '/'

        print(f'\nLoading generated samples from {self.PATH_GENERATED_SAMPLES}')

        self.generated_sample_type = generated_sample_type
        print(f'Type of generated samples: {self.generated_sample_type}\n')


        self.FIG_PATH = args.path_save + 'evaluation/plot_samples/' + self.save_str + '/' 
        

        if not os.path.exists(self.FIG_PATH):
            print(f'Creating directory {self.FIG_PATH}')
            os.makedirs(self.FIG_PATH)

        if self.generated_sample_type == 'repeated':
            load_str = '_repeatedSamples_' + str(n_samples) + '.npz'
        elif self.generated_sample_type == 'single':
            load_str = '_singleSample.npz'
        elif self.generated_sample_type == 'multiple':
            load_str ='_samples_' + str(n_samples) + '.npz'

        # Load generated images, truth evaluation images and lsm mask for each image
        self.gen_imgs = np.load(self.PATH_GENERATED_SAMPLES + 'gen' + load_str)['arr_0']
        self.eval_imgs = np.load(self.PATH_GENERATED_SAMPLES + 'eval' + load_str)['arr_0']
        try:
            self.lsm_imgs = np.load(self.PATH_GENERATED_SAMPLES + 'lsm' + load_str)['arr_0']
        except FileNotFoundError:
            print(f'LSM mask not found in {self.PATH_GENERATED_SAMPLES}. Setting to None.')
            print(f'Also setting masked to False in plot_example_images()')
            self.lsm_imgs = None
        self.cond_imgs = np.load(self.PATH_GENERATED_SAMPLES + 'cond' + load_str)['arr_0']

        # Convert to torch tensors
        self.gen_imgs = torch.from_numpy(self.gen_imgs).squeeze()
        self.eval_imgs = torch.from_numpy(self.eval_imgs).squeeze()
        if self.lsm_imgs is not None:
            self.lsm_imgs = torch.from_numpy(self.lsm_imgs).squeeze()
        self.cond_imgs = torch.from_numpy(self.cond_imgs).squeeze()


        print(self.gen_imgs.shape)
        print(self.eval_imgs.shape)
        if self.lsm_imgs is not None:
            print(self.lsm_imgs.shape)
        print(self.cond_imgs.shape)


    def plot_example_images(self,
                            masked=False,
                            plot_with_cond=False,
                            plot_with_lsm=False,
                            n_samples=0,
                            same_cbar=True,
                            show_figs=False,
                            save_figs=False
                            ):
        '''
            Plot example of generated and eval images w or w/o masking
        '''

        # If masked, check if lsm_imgs is None
        if masked and self.lsm_imgs is None:
            print(f'LSM mask not found. Setting masked to False in plot_example_images()')
            masked = False


        # Set number of samples to plot
        if self.generated_sample_type == 'single':
            n_samples = 1
        else:
            if n_samples == 0 and self.gen_imgs.shape[0] < 8:
                n_samples = self.gen_imgs.shape[0]
            elif n_samples == 0 and self.gen_imgs.shape[0] >= 8:
                n_samples = 8

        print(f'Plotting {n_samples} samples')
        # If only one sample is plotted, unsqueeze to add batch dimension
        if n_samples == 1:
            gen_imgs = self.gen_imgs.unsqueeze(0)
            eval_imgs = self.eval_imgs.unsqueeze(0)
        else:
            gen_imgs = self.gen_imgs[:n_samples]
            if self.generated_sample_type == 'repeated':
                # Repeate the eval image n_samples times
                eval_imgs = self.eval_imgs.repeat(n_samples, 1, 1)
            else:
                eval_imgs = self.eval_imgs[:n_samples]
            

        # Define lists of images to plot
        plot_list = [gen_imgs, eval_imgs]
        plot_titles = ['Generated image', 'Evaluation image']


        # Add conditional images and LSM mask to plot list if specified
        if plot_with_cond:
            if self.generated_sample_type == 'repeated':
                cond_imgs = self.cond_imgs.repeat(n_samples, 1, 1)
            else:
                cond_imgs = self.cond_imgs

            if n_samples == 1:
                cond_imgs = self.cond_imgs.unsqueeze(0)
            plot_list.append(cond_imgs)
            plot_titles.append('Conditional image')

        if self.lsm_imgs is not None:
            if self.generated_sample_type == 'repeated':
                lsm_imgs = self.lsm_imgs.repeat(n_samples, 1, 1)
            else:
                lsm_imgs = self.lsm_imgs
            if n_samples == 1:
                lsm_imgs = self.lsm_imgs.unsqueeze(0)
            
            if plot_with_lsm:
                plot_list.append(lsm_imgs)
                plot_titles.append('LSM mask')


        # Set number of axes and figure/axis object
        n_axs = len(plot_list)
        fig, axs = plt.subplots(n_axs, n_samples, figsize=(n_samples*2, n_axs*2))

        # If only one sample is plotted, unsqueeze axis to be able to iterate over it
        if n_axs == 1 and n_samples == 1:
            axs = np.array([[axs]])
        elif n_axs == 1:
            axs = np.expand_dims(axs, axis=0)
        if n_samples == 1:
            axs = np.expand_dims(axs, axis=1)


        # Plot on same colorbar, if same_cbar is True
        if same_cbar:
            vmin = np.nanmin([plot_list[i].min() for i in range(n_axs)])
            vmax = np.nanmax([plot_list[i].max() for i in range(n_axs)])
        else:
            vmin = None
            vmax = None
        print(f'Members to plot: {plot_titles}')
        print(f'Number of samples: {n_samples}')
        
        # Loop over samples and plot (n_axs x n_samples) images
        for i in range(n_samples):
            for j in range(n_axs):
                # If masked, set ocean pixels to nan
                if masked and plot_titles[j] != 'LSM mask':
                    plot_list[j][i][lsm_imgs[i]==0] = np.nan

                # Add plot_title to first image in each row (as y-label)
                if i == 0:
                    axs[j, i].set_ylabel(plot_titles[j], fontsize=14)
                # If lsm, set vmin and vmax to 0 and 0.1
                if plot_titles[j] == 'LSM mask':
                    im = axs[j, i].imshow(plot_list[j][i], vmin=0, vmax=0.1)
                else:
                    im = axs[j, i].imshow(plot_list[j][i], vmin=vmin, vmax=vmax)
                axs[j, i].set_ylim([0,plot_list[j][i].shape[0]])
                axs[j, i].set_xticks([])
                axs[j, i].set_yticks([])
                
                # If colorbar is the same for all images, only add it to the last image in each row
                if same_cbar and i == n_samples-1:
                    fig.colorbar(im, ax=axs[j, i], fraction=0.046, pad=0.04)
                elif not same_cbar:
                    fig.colorbar(im, ax=axs[j, i], fraction=0.046, pad=0.04)

        fig.tight_layout()

        # Save figure if specified
        if save_figs:
            if masked:
                print(f'\nSaving figure to {self.FIG_PATH + self.save_str + "__example_eval_gen_images_masked.png"}')
                fig.savefig(self.FIG_PATH + self.save_str + '__example_eval_gen_images_masked.png', dpi=600, bbox_inches='tight')
            else:
                print(f'\nSaving figure to {self.FIG_PATH + self.save_str + "__example_eval_gen_images.png"}')
                fig.savefig(self.FIG_PATH + self.save_str + '__example_eval_gen_images.png', dpi=600, bbox_inches='tight')

        # Show figure if specified
        if show_figs:
            plt.show()
        else:
            plt.close(fig)

        return fig, axs



    def full_pixel_statistics(self,
                              show_figs=False,
                              save_figs=False,
                              save_stats=False,
                              save_path=None,
                              n_samples=1
                              ):
        '''
            Calculate pixel-wise statistics for the generated samples.
            - RMSE and MAE for all pixels in all samples
            - Pixel value distribution for all samples
        '''
        
        
        #########################
        #                       #
        # FULL PIXEL STATISTICS #
        #                       #
        #########################
        
        # Calculate total single pixel-wise MAE and RMSE for all samples, no averaging
        # Flatten and concatenate the generated and eval images
        gen_imgs_flat = self.gen_imgs.flatten()
        eval_imgs_flat = self.eval_imgs.flatten()

        # Plot the pixel-wise distribution of the generated and eval images
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(gen_imgs_flat, bins=50, alpha=0.5, label='Generated')
        ax.hist(eval_imgs_flat, bins=50, alpha=0.5, color='r', label='Eval')
        ax.axvline(x=np.nanmean(eval_imgs_flat), color='r', alpha=0.5, linestyle='--', label=f'Eval mean, {np.nanmean(eval_imgs_flat):.2f}')
        ax.axvline(x=np.nanmean(gen_imgs_flat), color='b', alpha=0.5, linestyle='--', label=f'Generated mean, {np.nanmean(gen_imgs_flat):.2f}')
        ax.set_title(f'Distribution of generated and eval images, bias: {np.nanmean(gen_imgs_flat)-np.nanmean(eval_imgs_flat):.2f}', fontsize=14)
        ax.set_xlabel(f'Pixel value', fontsize=14)
        ax.set_ylabel(f'Count', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        # Set the x-axis limits to 4 sigma around the mean of the eval images
        x_min = np.nanmin([np.nanmin(eval_imgs_flat), np.nanmin(gen_imgs_flat)])
        x_max = np.nanmax([np.nanmax(eval_imgs_flat), np.nanmax(gen_imgs_flat)])
        ax.set_xlim([x_min, x_max])
        ax.legend()

        fig.tight_layout()

        # Save figure if specified
        if save_figs:
            save_str_loc = self.save_str + '_' + str(n_samples) + '_samples__pixel_distribution.png'
            print(f'\nSaving figure to {self.FIG_PATH + save_str_loc}')
            fig.savefig(self.FIG_PATH + save_str_loc, dpi=600, bbox_inches='tight')

        # Show figure if specified
        if show_figs:
            plt.show()
        else:
            plt.close(fig)

        # Save statistics if specified
        if save_stats:
            save_str_loc = self.save_str + '_' + str(n_samples) + '_samples__pixel_statistics.npz'
            np.savez(save_path + self.save_str + '_' + str(n_samples) + '_samples__pixel_statistics.npz', gen_imgs_flat, eval_imgs_flat)
        


        ###############################
        #                             #
        # RMSE AND MAE FOR ALL PIXELS #
        #                             # 
        ###############################

        # Calculate MAE and RMSE for all samples ignoring nans
        mae_all = torch.abs(gen_imgs_flat - eval_imgs_flat)
        rmse_all = torch.sqrt(torch.square(gen_imgs_flat - eval_imgs_flat))

        # Make figure with two plots: RMSE and MAE for all pixels
        fig, axs = plt.subplots(2, 1, figsize=(12,6))#, sharex='col')

        axs[0].hist(rmse_all, bins=150, alpha=0.7, label='RMSE all pixels', edgecolor='k')
        axs[0].set_title(f'RMSE for all pixels', fontsize=16)
        axs[0].tick_params(axis='y', which='major', labelsize=14)
        axs[0].set_ylabel(f'Count', fontsize=16)

        axs[1].hist(mae_all, bins=70, alpha=0.7, label='MAE all pixels', edgecolor='k')
        axs[1].set_title(f'MAE for all pixels', fontsize=16)
        axs[1].tick_params(axis='both', which='major', labelsize=14)
        axs[1].set_xlabel(f'RMSE', fontsize=16)
        axs[1].set_ylabel(f'Count', fontsize=16)

        fig.tight_layout()

        # Save figure if specified
        if save_figs:
            save_str_loc = self.save_str + '_' + str(n_samples) + '_samples__RMSE_MAE_histograms.png'
            print(f'Saving figure to {self.FIG_PATH + save_str_loc}')
            fig.savefig(self.FIG_PATH + save_str_loc, dpi=600, bbox_inches='tight')
        
        # Show figure if specified
        if show_figs:
            plt.show()
        else:
            plt.close(fig)

        # Save statistics if specified
        if save_stats:
            save_str_loc = self.save_str + '_' + str(n_samples) + '_samples__RMSE_MAE_statistics.npz'
            np.savez(save_path + save_str_loc, mae_all=mae_all.numpy(), rmse_all=rmse_all.numpy())





    def daily_statistics(self,
                        plot_stats=False,
                        save_plots=False,
                        save_plot_path=None,
                        save_stats=False,
                        save_path=None
                        ):
        '''
            Calculate daily average MAE and RMSE for all samples (average over spatial dimensions) ignoring nans
        '''

        # Calculate daily average MAE and RMSE for all samples (average over spatial dimensions) ignoring nans
        mae_daily = torch.abs(self.gen_imgs - self.eval_imgs).nanmean(dim=(1,2))
        rmse_daily = torch.sqrt(torch.square(self.gen_imgs - self.eval_imgs).nanmean(dim=(1,2)))


                                

    def spatial_statistics(self,
                           show_figs=False,
                           save_figs=False,
                           save_plot_path=None,
                           save_stats=False,
                           save_path=None,
                           n_samples=1
                           ):
        '''
            Calculate spatial statistics for the generated samples.
            - Moran's I
            - Bias per pixel
            - Bias per image
            - Bias per pixel per image
        '''
        
        # Calculate rmse per pixel
        rmse_per_pixel = torch.sqrt(torch.square(self.gen_imgs - self.eval_imgs).nanmean(dim=0))

        # Calculate mae per pixel
        mae_per_pixel = torch.abs(self.gen_imgs - self.eval_imgs).nanmean(dim=0)

        # Calculate bias per pixel
        bias_per_pixel = self.gen_imgs.nanmean(dim=0) - self.eval_imgs.nanmean(dim=0)

        # Plot the spatial statistics
        fig, axs = plt.subplots(2, 2, figsize=(12,12))

        im = axs[0,0].imshow(rmse_per_pixel)
        axs[0,0].set_title(f'RMSE per pixel', fontsize=16)
        fig.colorbar(im, ax=axs[0,0])

        im = axs[0,1].imshow(mae_per_pixel)
        axs[0,1].set_title(f'MAE per pixel', fontsize=16)
        fig.colorbar(im, ax=axs[0,1])

        im = axs[1,0].imshow(bias_per_pixel)
        axs[1,0].set_title(f'Bias per pixel', fontsize=16)
        fig.colorbar(im, ax=axs[1,0])

        fig.tight_layout()

        # Save figure if specified
        if save_figs:
            save_str_loc = self.save_str + '_' + str(n_samples) + '_samples__spatial_statistics.png'
            print(f'Saving figure to {self.FIG_PATH + save_str_loc}')
            fig.savefig(self.FIG_PATH + save_str_loc, dpi=600, bbox_inches='tight')

        # Show figure if specified
        if show_figs:
            plt.show()
        else:
            plt.close(fig)





if __name__ == '__main__':
    evaluate_from_args()
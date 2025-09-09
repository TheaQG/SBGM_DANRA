
import torch
import os
import logging

import numpy as np
from matplotlib import pyplot as plt

from sbgm.score_sampling import pc_sampler
from sbgm.score_unet import marginal_prob_std_fn, diffusion_coeff_fn
from sbgm.utils import plot_samples_and_generated, extract_samples, get_model_string, get_first_sample_dict

logger = logging.getLogger(__name__)


class Evaluation:
    '''
        Class to evaluate generated samples (saved in npz files) from the SBGM model.

        Evaluates the generated samples using the following metrics:
        - All pixel values distribution (across space and time), generated and eval images
        - RMSE and MAE for all pixels in all samples
    '''
    def __init__(self, cfg, generated_sample_type='repeated', n_samples=4):
        '''
            Setup the evaluation class.
            Get configuration from parsed cfg
            Load samples from config path.
            
            - sample_type: What type of generated samples to evaluate. ['repeated', 'single', 'multiple'] 
        '''


        self.hr_var = cfg.highres.variable
        self.hr_model = cfg.highres.model

        self.lr_model = cfg.lowres.model
        self.lr_vars = cfg.lowres.condition_variables
        self.lr_vars_str = '_'.join(self.lr_vars)

        self.hr_data_size = cfg.highres.data_size
        if self.hr_data_size is None:
            self.hr_data_size = [128, 128]
        
        self.loss_type = cfg.training.loss_type
        
        self.model_name_str = get_model_string(cfg)
        self.output_dir = os.path.join(cfg.paths.sample_dir, 'generation', self.model_name_str)
        self.generated_sample_path = os.path.join(self.output_dir, 'generated_samples')
        self.evaluation_path = os.path.join(cfg.paths.evaluation_dir, self.model_name_str)
        self.evaluation_fig_path = os.path.join(self.evaluation_path, 'figures')
        self.evaluation_stats_path = os.path.join(self.evaluation_path, 'statistics')
        # Make sure evaluation path exists, if not create it
        os.makedirs(self.evaluation_path, exist_ok=True)
        os.makedirs(self.evaluation_fig_path, exist_ok=True)
        os.makedirs(self.evaluation_stats_path, exist_ok=True)

        logger.info(f'\nLoading generated samples from {self.generated_sample_path}')

        self.generated_sample_type = generated_sample_type
        logger.info(f'Type of generated samples: {self.generated_sample_type}\n')


        # Load correct .npz !!! NEEDS TO BE FIXED! NEED CORRECT PATH TO DATA !!!
        if self.generated_sample_type == 'repeated':
            load_str = '_repeated_' + str(n_samples) + '.npz'
        elif self.generated_sample_type == 'single':
            load_str = '_single.npz'
        elif self.generated_sample_type == 'multiple':
            load_str ='_multi_n_' + str(n_samples) + '.npz'
        else:
            raise ValueError(f'Unknown generated sample type: {self.generated_sample_type}. Choose from: ["repeated", "single", "multiple"].')

        
        # Load generated samples (called 'gen_samples_')
        gen_im_path = os.path.join(self.generated_sample_path, 'gen_samples' + load_str)
        self.gen_imgs = np.load(gen_im_path)['arr_0']

        eval_im_path = os.path.join(self.generated_sample_path, 'eval_samples' + load_str)
        self.eval_imgs = np.load(eval_im_path)['arr_0']

        # Load conditional images
        self.cond_imgs = []
        for cond_var in self.lr_vars:
            cond_path = os.path.join(self.generated_sample_path, f'cond_samples_{cond_var}{load_str}')
            if not os.path.exists(cond_path):
                raise FileNotFoundError(f"Missing conditional file: {cond_path}")
            cond_arr = np.load(cond_path)['arr_0']
            cond_tensor = torch.from_numpy(cond_arr).squeeze()
            self.cond_imgs.append(cond_tensor)

        # Stack into shape: [B, C, H, W]
        # First ensure shape compatibility
        shapes = [c.shape for c in self.cond_imgs]
        if len(set(shapes)) != 1:
            raise ValueError(f"Inconsistent conditional image shapes: {shapes}")
        self.cond_imgs = torch.stack(self.cond_imgs, dim=1)

        try:
            lsm_im_path = os.path.join(self.generated_sample_path, 'lsm_samples' + load_str)
            self.lsm_imgs = np.load(lsm_im_path)['arr_0']
        except FileNotFoundError:
            print(f'LSM mask not found in {self.generated_sample_path}. Setting to None.')
            print(f'Also setting masked to False in plot_example_images()')
            self.lsm_imgs = None

        # Convert to torch tensors
        self.gen_imgs = torch.from_numpy(self.gen_imgs).squeeze()
        self.eval_imgs = torch.from_numpy(self.eval_imgs).squeeze()
        if self.lsm_imgs is not None:
            self.lsm_imgs = torch.from_numpy(self.lsm_imgs).squeeze()


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
            cond_imgs = self.cond_imgs[:n_samples]
            for i, cond_name in enumerate(self.lr_vars):
                cond_channel = cond_imgs[:, i, :, :]
                plot_list.append(cond_channel)
                plot_titles.append(f'Condition: {cond_name}')

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
        else:
            print(f'LSM mask not found in {self.generated_sample_path}. Setting to None.')
            lsm_imgs = None

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
                # If LSM exists, mask ocean pixels
                if lsm_imgs is not None:
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
                save_path = os.path.join(self.evaluation_fig_path, f'example_n_samples_{n_samples}_eval_gen_images_masked.png')
                print(f'\nSaving figure to {save_path}')
                fig.savefig(save_path, dpi=600, bbox_inches='tight')

            else:
                save_path = os.path.join(self.evaluation_fig_path, f'example_n_samples_{n_samples}_eval_gen_images.png')
                print(f'\nSaving figure to {save_path}')
                fig.savefig(save_path, dpi=600, bbox_inches='tight')

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
        ax.axvline(x=float(np.nanmean(eval_imgs_flat)), color='r', alpha=0.5, linestyle='--', label=f'Eval mean, {np.nanmean(eval_imgs_flat):.2f}')
        ax.axvline(x=float(np.nanmean(gen_imgs_flat)), color='b', alpha=0.5, linestyle='--', label=f'Generated mean, {np.nanmean(gen_imgs_flat):.2f}')
        ax.set_title(f'Distribution of generated and eval images, bias: {np.nanmean(gen_imgs_flat)-np.nanmean(eval_imgs_flat):.2f}', fontsize=14)
        ax.set_xlabel(f'Pixel value', fontsize=14)
        ax.set_ylabel(f'Count', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        # Set the x-axis limits to 4 sigma around the mean of the eval images
        x_min = np.nanmin([np.nanmin(eval_imgs_flat), np.nanmin(gen_imgs_flat)])
        x_max = np.nanmax([np.nanmax(eval_imgs_flat), np.nanmax(gen_imgs_flat)])
        ax.set_xlim(x_min, x_max)
        ax.legend()

        fig.tight_layout()

        # Save figure if specified
        if save_figs:
            save_path = os.path.join(self.evaluation_fig_path, f'example_n_samples_{n_samples}_eval_gen_images.png')
            print(f'\nSaving figure to {save_path}')
            fig.savefig(save_path, dpi=600, bbox_inches='tight')

        # Show figure if specified
        if show_figs:
            plt.show()
        else:
            plt.close(fig)

        # Save statistics if specified
        if save_stats:
            save_path = os.path.join(self.evaluation_stats_path, f'n_samples_{n_samples}_pixel_statistics.npz')
            print(f'\nSaving statistics to {save_path}')
            np.savez(save_path, gen_imgs_flat=gen_imgs_flat, eval_imgs_flat=eval_imgs_flat)

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
            save_path = os.path.join(self.evaluation_fig_path, f'n_samples_{n_samples}_RMSE_MAE_histograms.png')
            print(f'Saving figure to {save_path}')
            fig.savefig(save_path, dpi=600, bbox_inches='tight')

        # Show figure if specified
        if show_figs:
            plt.show()
        else:
            plt.close(fig)

        # Save statistics if specified
        if save_stats:
            save_path = os.path.join(self.evaluation_stats_path, f'n_samples_{n_samples}_RMSE_MAE_statistics.npz')
            print(f'Saving statistics to {save_path}')
            np.savez(save_path, mae_all=mae_all.numpy(), rmse_all=rmse_all.numpy())





    def daily_statistics(self,
                        plot_stats=False,
                        save_plots=False,
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
            save_path = os.path.join(self.evaluation_fig_path, f'n_samples_{n_samples}_spatial_statistics.png')
            print(f'Saving figure to {save_path}')
            fig.savefig(save_path, dpi=600, bbox_inches='tight')

        # Show figure if specified
        if show_figs:
            plt.show()
        else:
            plt.close(fig)


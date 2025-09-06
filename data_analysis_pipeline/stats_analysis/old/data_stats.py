'''
    Script to investigate the full dataset and get some statistics.
    Mainly to get an idea of the data distribution and the range of values
    and how it changes when data is scaled.


'''

import os
import zarr
import argparse
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from data_modules import DANRA_Dataset_cutouts_ERA5_Zarr
from special_transforms import ZScoreTransform, PrcpLogTransform, Scale
from scipy.stats import boxcox, yeojohnson
from scipy.optimize import minimize_scalar
from utils import *

def data_stats_from_args():
    '''
        Function to get arguments from the command line and run the data_stats function
    '''
    parser = argparse.ArgumentParser(description='Compute statistics of the data')
    parser.add_argument('--var', type=str, default='temp', help='The variable to compute statistics for')
    parser.add_argument('--data_type', type=str, default='ERA5', help='The dataset to compute statistics for (DANRA or ERA5)')
    parser.add_argument('--split_type', type=str, default='test', help='The split type of the data (train, val, test)')
    parser.add_argument('--path_data', type=str, default='/Users/au728490/Library/CloudStorage/OneDrive-Aarhusuniversitet/PhD_AU/Python_Scripts/Data/Data_DiffMod/', help='The path to the data')
    parser.add_argument('--create_figs', type=str2bool, default=True, help='Whether to create figures')
    parser.add_argument('--save_figs', type=str2bool, default=True, help='Whether to save the figures')
    parser.add_argument('--show_figs', type=str2bool, default=True, help='Whether to show the figures')
    parser.add_argument('--save_stats', type=str2bool, default=False, help='Whether to save the statistics')
    parser.add_argument('--print_final_stats', type=str2bool, default=True, help='Whether to print the statistics')
    parser.add_argument('--fig_path', type=str, default='../Data_Stats_Figs/', help='The path to save the figures')
    parser.add_argument('--stats_path', type=str, default='../data_statistics', help='The path to save the statistics')
    parser.add_argument('--transformations', type=str2list_of_strings, default=None, help='List of transformations to apply to the data')#, choices=['zscore', 'log', 'log01', 'log_minus1_1', 'log_zscore'])
    parser.add_argument('--show_only_transformed', type=str2bool, default=False, help='Whether to show only the transformed data')
    parser.add_argument('--time_agg', type=str, default='daily', choices=['daily', 'weekly', 'monthly'], help='Time aggregation for statistics (daily, weekly, monthly, yearly)')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers to use for CPU multiprocessing')
    
    args = parser.parse_args()

    print('\nAssigning parameters to DataStats class\n')
    data_stats = DataStats(**vars(args))
    print('Running DataStats analysis...\n')
    data_stats.run()



class DataStats:
    '''
        Class for investigating, visualizing and computing statistics of the data.
        Contains methods to load the data, compute statistics, apply transformations and visualize the data.
        The point is to be able to investigate the data and get an idea of the distribution of the data, as well
        as how it changes when transformed using various methods.

        CURRENTLY IMPLEMENTED:
            - compute_statistics: Method to compute statistics of the data
            - load_data: Method to load the data
            - apply_transformations: Method to apply transformations to the data
            - visualize_data: Method to visualize the data and the statistics

        DEVELOPMENT: 
            - More types of data (water vapor, CAPE, etc.)
            - Possibility for analyzing all data at once (not just one split)
            - Add possibility for monthly/weekkly stats instead of daily timeseries
    '''
    def __init__(self,
                var='prcp',
                data_type='DANRA',
                split_type='valid',
                path_data='/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/',
                create_figs=True,
                save_figs=False,
                show_figs=True,
                save_stats=False,
                print_final_stats=False,
                fig_path='/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/',
                stats_path='/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/',
                transformations=None,
                show_only_transformed=False,
                time_agg='daily',
                n_workers=1):

        self.var = var                  # The variable to compute statistics for (for now, prcp and temp)
        self.data_type = data_type      # The dataset to compute statistics for (DANRA or ERA5)
        self.split_type = split_type    # The split type of the data (train, val, test, all(DEVELOPMENT))
        self.path_data = path_data      # The path to the data
        self.create_figs = create_figs  # Whether to create figures
        self.show_figs = show_figs      # Whether to show the figures
        self.save_stats = save_stats    # Whether to save the statistics
        self.print_final_stats = print_final_stats  # Whether to print the statistics
        self.fig_path = fig_path        # The path to save the figures
        self.stats_path = stats_path    # The path to save the statistics
        self.transformations = transformations if transformations else [] # List of transformations to apply to the data
        self.show_only_transformed = show_only_transformed # Whether to show only the transformed data
        self.time_agg = time_agg        # Time aggregation for statistics (daily, weekly, monthly, yearly)
        self.n_workers = n_workers      # How many CPU processes to spawn for multiprocessing

        self.save_figs = save_figs      # Whether to save the figures
        if self.save_figs:
            self.transformation_str = '_'.join(self.transformations) if self.transformations else 'raw'


        # Set some plot and variable specific parameters
        if self.var == 'temp':
            self.var_str = 't'
            self.cmap = 'plasma'
            self.var_label = 'Temperature [C]'
        elif self.var == 'prcp':
            self.var_str = 'tp'
            self.cmap = 'inferno'
            self.var_label = 'Precipitation [mm]'
            # IMPLEMENT MORE VARIABLES HERE

        # Set some naming parameters
        self.danra_size_str = '589x789'
        # Hard-coded cutout for now
        self.cutout = [170, 170+180, 340, 340+180]

    def compute_statistics(self, data):
        '''
            Basic descriptive stats on a NumPy array.
        '''
        
        mean = np.mean(data)
        median = np.median(data)
        std_dev = np.std(data)
        variance = np.var(data)
        min_temp = np.min(data)
        max_temp = np.max(data)
        percentiles = np.percentile(data, [25, 50, 75])
        return mean, median, std_dev, variance, min_temp, max_temp, percentiles

    def parse_file_date(self, filename):
        '''
            Attempt to parse filename into a datetime object.
            Example file name format: 'tp_tot_20030101' (xxxxxx_YYYYMMDD)
        '''
        if len(filename) < 8:
            return None
        date_str = filename[-8:]
        try:
            return datetime.datetime.strptime(date_str, '%Y%m%d')
        except ValueError:
            return None

    def _process_single_file(self, zarr_group_img, file):
        '''
            Method called by each CPU worker to process a single file.
            1) Reads data from zarr
            2) Cuts out a specific region
            3) Basic variable correction
            4) Computes statistics
            returns: (stats_dict_for_this_file, data_array)
        '''
        # Print progress

        # Try read
        try:
            data = zarr_group_img[file][self.var_str][:].squeeze()
        except KeyError:
            data = None
            for fallback_key in ['arr_0', 'data']:
                if fallback_key in zarr_group_img[file]:
                    data = zarr_group_img[file][fallback_key][:].squeeze()
                    break
            if data is None:
                # Could not read data
                return None
        
        # Cutout
        data = data[self.cutout[0]:self.cutout[1], self.cutout[2]:self.cutout[3]]
        
        # Adjust for var
        if self.var == 'temp':
            # Check the mean before conversion
            # print(f"Mean before conversion: {np.mean(data)}")
            data = data - 273.15
        elif self.var == 'prcp':
            data[data <= 0] = 1e-8
            if self.data_type == 'ERA5':
                # Transform from [m] to [mm]
                data = data*1000 
        
        # Compute log-min and log-max for prcp
        if self.var == 'prcp':
            data_log = np.log(data)
            file_min_log = np.min(data_log)
            file_max_log = np.max(data_log)
            file_mean_log = np.mean(data_log)
            file_std_log = np.std(data_log)
        else:
            file_min_log = None
            file_max_log = None
            file_mean_log = None
            file_std_log = None

        # Compute stats
        mean, median, std_dev, variance, min_val, max_val, _ = self.compute_statistics(data)
        date_obj = self.parse_file_date(file)
        # Build a single-file stats dict
        single_stats = {
            "file": file,
            "date": date_obj if date_obj else file,
            "mean": mean,
            "median": median,
            "std_dev": std_dev,
            "variance": variance,
            "min": min_val,
            "max": max_val,
        }

        # also store min-log, max-log, mean-log and std-log for prcp
        if self.var == 'prcp':
            single_stats["min_log"] = file_min_log
            single_stats["max_log"] = file_max_log
            single_stats["mean_log"] = file_mean_log
            single_stats["std_log"] = file_std_log

        return (single_stats, data)

    def load_data(self, plot_cutout=True):
        '''
            Method to load the data from the zarr files and compute statistics.
            Optionally plot the cutout for the first file.
            If n_workers > 1, parallelize the CPU processing of the files.
        '''
        path_data = os.path.join(
            self.path_data,
            f"data_{self.data_type}",
            f"size_{self.danra_size_str}",
            f"{self.var}_{self.danra_size_str}",
            "zarr_files"
        )
        data_dir_zarr = os.path.join(path_data, f"{self.split_type}.zarr")
        print(f"\nOpening zarr group: {data_dir_zarr}\n")
        zarr_group_img = zarr.open_group(data_dir_zarr, mode='r')
        files = list(zarr_group_img.keys())
        files.sort()

        # Prepare final stats_dict
        stats_dict = {
            "date": [],
            "mean": [],
            "median": [],
            "std_dev": [],
            "variance": [],
            "min": [],
            "max": [],
        }

        # If prcp, also add min-log, max-log, mean-log, std-log
        if self.var == 'prcp':
            stats_dict["min_log"] = []
            stats_dict["max_log"] = []
            stats_dict["mean_log"] = []
            stats_dict["std_log"] = []
        all_data_list = []

        # Prepare figure and stats paths (only if save_figs or save_stats)

        if self.save_figs and not os.path.exists(self.fig_path):
            os.makedirs(self.fig_path, exist_ok=True)
        if self.save_stats and not os.path.exists(self.stats_path):
            os.makedirs(self.stats_path, exist_ok=True)

        # (1) If user wants to parallelize, create a Pool, using imap to show progress
        if self.n_workers > 1:
            import multiprocessing
            from functools import partial

            worker_fn = partial(self._process_single_file, zarr_group_img)
            with multiprocessing.Pool(self.n_workers) as pool:
                # imap returns results in the order of fields, but yields them as they finish
                results_iter = pool.imap(worker_fn, files)

                # Loop over results
                for idx, out in enumerate(results_iter):
                    if out is None:
                        continue
                    single_stats, data = out

                    # Store daily stats
                    for k in ["date", "mean", "median", "std_dev", "variance", "min", "max"]:
                        stats_dict[k].append(single_stats[k])

                    # Also store min-log, max-log, mean-log, std-log for prcp
                    if self.var == 'prcp':
                        for k in ["min_log", "max_log", "mean_log", "std_log"]:
                            stats_dict[k].append(single_stats[k])

                    # Save data if we want global distribution
                    all_data_list.append(data)

                    # Print progress every 100 files
                    if idx % 100 == 0:
                        print(f"Processed {idx+1}/{len(files)} files...", flush=True)

                    # Optionally plot cutout for the first file
                    if idx == 0 and plot_cutout:
                        self._plot_cutout(data, single_stats["file"])
                        
        else:
            # Single process
            for idx, file in enumerate(files):
                out = self._process_single_file(zarr_group_img, file)
                if out is None:
                    continue
                single_stats, data = out

                # Store daily stats
                for k in ["date", "mean", "median", "std_dev", "variance", "min", "max"]:
                    stats_dict[k].append(single_stats[k])

                # Also store min-log, max-log, mean-log, std-log for prcp
                if self.var == 'prcp':
                    for k in ["min_log", "max_log", "mean_log", "std_log"]:
                        stats_dict[k].append(single_stats[k])

                # Save data if we want global distribution
                all_data_list.append(data)

                # Print progress every 100 files
                if idx % 100 == 0:
                    print(f"Processed {idx+1}/{len(files)} files...", flush=True)

                # Optionally plot cutout for the first file 
                if idx == 0 and plot_cutout:
                    self._plot_cutout(data, single_stats["file"])

        # Convert lists to arrays
        for k in ["mean","median","std_dev","variance","min","max"]:
            stats_dict[k] = np.array(stats_dict[k], dtype=float)

        # Also convert min-log, max-log, mean-log, std-log for prcp
        if self.var == 'prcp':
            for k in ["min_log", "max_log", "mean_log", "std_log"]:
                stats_dict[k] = np.array(stats_dict[k], dtype=float)


        # --- Compute global stats all_data_list ---
        all_data_flat = np.concatenate(all_data_list, axis=0).flatten()
        glob_mean = np.mean(all_data_flat)
        glob_median = np.median(all_data_flat)
        glob_std_dev = np.std(all_data_flat)
        glob_variance = np.var(all_data_flat)
        glob_min = np.min(all_data_flat)
        glob_max = np.max(all_data_flat)
        # For prcp, also compute min-log, max-log, mean-log, std-log
        if self.var == 'prcp':
            all_data_flat_log = np.log(all_data_flat + 1e-8)
            glob_min_log = np.min(all_data_flat_log)
            glob_max_log = np.max(all_data_flat_log)
            glob_mean_log = np.mean(all_data_flat_log)
            glob_std_log = np.std(all_data_flat_log)



        # Print final stats
        if self.print_final_stats:
            print(f"\nFinal stats for {self.data_type} {self.var.capitalize()} {self.split_type}:")
            print(f"Global mean: {glob_mean:.5f}")
            print(f"Global median: {glob_median:.5f}")
            print(f"Global std_dev: {glob_std_dev:.5f}")
            print(f"Global variance: {glob_variance:.5f}")
            print(f"Global min: {glob_min:.5f}")
            print(f"Global max: {glob_max:.5f}")
            if self.var == 'prcp':
                print(f"Global min-log: {glob_min_log:.5f}")
                print(f"Global max-log: {glob_max_log:.5f}")
                print(f"Global mean-log: {glob_mean_log:.5f}")
                print(f"Global std-log: {glob_std_log:.5f}")

        # Optionally save stats (with date, mean, median, std_dev, variance, min, max - and min-log, max-log, mean-log, std-log for prcp)
        if self.save_stats:
            if not os.path.exists(self.stats_path):
                os.makedirs(self.stats_path)
            out_csv = os.path.join(self.stats_path, f"{self.var}_{self.split_type}_{self.data_type}_stats.csv")
            
            if self.var == 'prcp':
                np.savez(out_csv, date=stats_dict["date"], mean=stats_dict["mean"], median=stats_dict["median"],
                        std_dev=stats_dict["std_dev"], variance=stats_dict["variance"], min=stats_dict["min"],
                        max=stats_dict["max"], min_log=stats_dict["min_log"], max_log=stats_dict["max_log"],
                        mean_log=stats_dict["mean_log"], std_log=stats_dict["std_log"])
            else:
                np.savez(out_csv, date=stats_dict["date"], mean=stats_dict["mean"], median=stats_dict["median"],
                        std_dev=stats_dict["std_dev"], variance=stats_dict["variance"], min=stats_dict["min"],
                        max=stats_dict["max"])
                
        return all_data_list, stats_dict
    
    def _plot_cutout(self, data, file_label):
        """
            Helper to avoid duplicating the cutout plotting code.
        """
        fig, ax = plt.subplots(1,1, figsize=(5,5))
        im = ax.imshow(data, cmap=self.cmap)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=self.var_label)
        ax.set_title(f"First cutout, {self.data_type} {self.var.capitalize()} {self.split_type}: {file_label}")
        ax.invert_yaxis()
        if self.save_figs:
            out_path = os.path.join(self.fig_path, f'{self.var}_{self.split_type}_{self.data_type}_{self.transformation_str}_cutout.png')
            fig.savefig(out_path, dpi=300, bbox_inches='tight')

        if self.show_figs:
            plt.show(block=False)
            plt.pause(2)
            plt.close(fig)
        else:
            plt.close(fig)

    def apply_transformations(self,
                                data_1d):
        '''
            Method to apply transformations to the data.
            Currently implemented transformations:
                - Z-Score Normalization
                - Log Transformation

            DEVELOPMENT:
                - Add raw -> log -> [0,1] transformation
                
        '''

        transformed_data = {}
        for transform in self.transformations:
            if transform == 'zscore':
                mu = np.mean(data_1d)
                sigma = np.std(data_1d) + 1e-8
                transformed_data['zscore'] = (data_1d - mu) / sigma
            elif transform == 'log':
                transformed_data['log'] = np.log(data_1d + 1e-8)
            elif transform == 'log01':
                # Log transformation to [0,1] range
                data_log = np.log(data_1d + 1e-8)
                transformed_data['log01'] = (data_log - np.min(data_log)) / (np.max(data_log) - np.min(data_log))
            elif transform == 'log_zscore':
                # Log transformation followed by z-score normalization
                data_log = np.log(data_1d + 1e-8)
                mu = np.mean(data_log)
                sigma = np.std(data_log) + 1e-8
                transformed_data['log_zscore'] = (data_log - mu) / sigma
            elif transform == 'log_minus1_1':
                # Log transformation to [-1,1] range
                data_log = np.log(data_1d + 1e-8)
                data_log_max = np.max(data_log)
                data_log_min = np.min(data_log)
                # Normalize to [-1,1]
                transformed_data['log_minus1_1'] = 2 * (data_log - data_log_min) / (data_log_max - data_log_min) - 1
            else:
                print(f"Transformation {transform} not implemented yet.")
            # Add more transformations as needed

        return transformed_data

    def aggregate_stats(self, stats_dict):
        '''
            Aggregate daily stats into weekly or monthly bins as requested
            NOTE: Not the ACTUAL complete stats, just the mean of the daily stats.
        '''
        if self.time_agg == 'daily':
            return stats_dict

        # Group daily stats into weekly or monthly bins
        agg_map = {} # (year,weekOrMonth) -> list of indices
        date_list = stats_dict["date"]

        # Build groups of daily indices
        for i, d_obj in enumerate(date_list):
            # If not a datetime, skip or try to parse again
            if not isinstance(d_obj, datetime.datetime):
                continue
            if self.time_agg == 'weekly':
                # isocalendar give (year, week, weekday) tuple
                y, w, _ = d_obj.isocalendar()
                key = (y, w)
            else:
                key = (d_obj.year, d_obj.month)

            if key not in agg_map:
                agg_map[key] = []
            agg_map[key].append(i)

        # Prepare new agg. dict
        agg_stats_dict = {
            "date": [],
            "mean": [],
            "median": [],
            "std_dev": [],
            "variance": [],
            "min": [],
            "max": [],
        }

        if self.var == 'prcp':
            agg_stats_dict["min_log"] = []
            agg_stats_dict["max_log"] = []
            agg_stats_dict["mean_log"] = []
            agg_stats_dict["std_log"] = []

        # For labeling the new x-axis
        for key in sorted(agg_map.keys()):
            indices = agg_map[key]

            # Now just do average of daily stats in that bin
            mean_val = np.mean(stats_dict["mean"][indices])
            median_val = np.mean(stats_dict["median"][indices])
            std_val = np.mean(stats_dict["std_dev"][indices])
            var_val = np.mean(stats_dict["variance"][indices])
            min_val = np.mean(stats_dict["min"][indices])
            max_val = np.mean(stats_dict["max"][indices])

            # Also for prcp, do average of min-log, max-log, mean-log, std-log
            if self.var == 'prcp':
                min_log_val = np.mean(stats_dict["min_log"][indices])
                max_log_val = np.mean(stats_dict["max_log"][indices])
                mean_log_val = np.mean(stats_dict["mean_log"][indices])
                std_log_val = np.mean(stats_dict["std_log"][indices])

            # For new date array, just use the first date in the bin
            d_first = date_list[indices[0]]
            # If weekly, label with year-week, if monthly, label with year-month, if yearly, just year
            if self.time_agg == 'weekly':
                agg_label = f"{key[0]}-W{key[1]}"
            elif self.time_agg == 'monthly':
                agg_label = f"{key[0]}-{key[1]:02d}"
            elif self.time_agg == 'yearly':
                agg_label = f"{key[0]}"

            # Append to new stats dict
            agg_stats_dict["date"].append(agg_label)
            agg_stats_dict["mean"].append(mean_val)
            agg_stats_dict["median"].append(median_val)
            agg_stats_dict["std_dev"].append(std_val)
            agg_stats_dict["variance"].append(var_val)
            agg_stats_dict["min"].append(min_val)
            agg_stats_dict["max"].append(max_val)

            # Also for prcp, append min-log, max-log, mean-log, std-log
            if self.var == 'prcp':
                agg_stats_dict["min_log"].append(min_log_val)
                agg_stats_dict["max_log"].append(max_log_val)
                agg_stats_dict["mean_log"].append(mean_log_val)
                agg_stats_dict["std_log"].append(std_log_val)

        # Convert to np.array
        for k in ["mean", "median", "std_dev", "variance", "min", "max"]:
            agg_stats_dict[k] = np.array(agg_stats_dict[k], dtype=float)

        # Also convert min-log, max-log, mean-log, std-log for prcp
        if self.var == 'prcp':
            for k in ["min_log", "max_log", "mean_log", "std_log"]:
                agg_stats_dict[k] = np.array(agg_stats_dict[k], dtype=float)

        return agg_stats_dict

    def visualize_data(self,
                        all_data_list,
                        stats_dict,
                        ):
        '''
            Visualize either time series stats or distribution histograms.
        '''

        # 1) Possibly aggregate stats
        agg_stats_dict = self.aggregate_stats(stats_dict) if self.time_agg != 'daily' else stats_dict
        
        # 2) Time-series stats 
        # stats_dict has arrays for each field
        if self.create_figs and len(agg_stats_dict["date"]) > 1:
            fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
            fig.suptitle(f'{self.data_type} {self.var.capitalize()} {self.split_type} Statistics', fontsize=14)

            # Integer x-axis
            x_vals = np.arange(len(agg_stats_dict["date"])) # or keep as strings on x-ticks

            # Plot mean +/- std_dev
            ax[0].errorbar(x_vals, agg_stats_dict["mean"],
                           yerr=agg_stats_dict["std_dev"],
                           label='Mean', marker='.', lw=0.5,
                           fmt='-.', ecolor='gray', capsize=2)
            ax[0].set_ylabel('Mean')
            ax[0].legend()

            # Plot min
            ax[1].plot(x_vals, agg_stats_dict["min"], label='Min', marker='.', lw=0.5)
            ax[1].set_ylabel('Min')
            ax[1].legend()

            # Plot max
            ax[2].plot(x_vals, agg_stats_dict["max"], label='Max', marker='.', lw=0.5)
            ax[2].set_ylabel('Max')
            ax[2].set_xlabel(f'Time ({self.time_agg.capitalize()})')
            ax[2].legend()

            # Use the date labels, but only show every 5th
            x_ticks = agg_stats_dict["date"]
            ax[2].set_xticks(x_vals[::5])
            ax[2].set_xticklabels(x_ticks[::5], rotation=45, ha='right')

            fig.tight_layout()

            if self.save_figs:
                out_path = os.path.join(self.fig_path, f'{self.var}_{self.split_type}_{self.data_type}_{self.time_agg}_{self.transformation_str}_timeseries.png')
                fig.savefig(out_path, dpi=300, bbox_inches='tight')
            if self.show_figs:
                # Show for 5 seconds
                plt.show(block=False)
                plt.pause(5)
                plt.close(fig)
            else:
                plt.close(fig)


        # 3) Global Distribution histograms (values)
        # For histograms of entire dataset, we need all in memory
        all_data_flat = np.concatenate(all_data_list, axis=0).flatten()
        if self.create_figs:
            # Original data histogram
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            if not self.show_only_transformed:
                mu = np.mean(all_data_flat)
                std = np.std(all_data_flat)
                label_orig = f'Original, mu={mu:.2f}, std={std:.2f}'
                ax.hist(all_data_flat, bins=100, alpha=0.7, label=label_orig)

            # Transformed data histograms
            transformed_data = self.apply_transformations(all_data_flat)
            for key, arr in transformed_data.items():
                mu_t = np.mean(arr)
                std_t = np.std(arr)
                # Make non-nan array 
                arr = arr[~np.isnan(arr)]
                label_t = f'{key.capitalize()} Transformed, mu={mu_t:.2f}, std={std_t:.2f}'
                ax.hist(arr, bins=100, alpha=0.7, label=label_t)

            ax.set_title(f"Global Distribution - {self.data_type} {self.var.capitalize()}, {self.split_type}")
            if self.var == 'temp':
                ax.set_xlabel('Temperature [C]')
            elif self.var == 'prcp':
                ax.set_xlabel('Precipitation [mm]')
                # IMPLEMENT MORE VARIABLES HERE
            ax.set_ylabel('Frequency')
            ax.legend()
            # If transformation 'log', set y-scale to log
            if any([t in self.transformations for t in ['log', 'log01', 'log_zscore', 'log_minus1_1']]):
                ax.set_yscale('log')

            if self.save_figs:
                out_path = os.path.join(self.fig_path, f'{self.var}_{self.split_type}_{self.data_type}_{self.transformation_str}_all_data.png')
                fig.savefig(out_path, dpi=300, bbox_inches='tight')
            if self.show_figs:
                # Show for 5 seconds
                plt.show(block=False)
                plt.pause(5)
                plt.close(fig)
            else:
                plt.close(fig)


        # 4) Global Distribution histograms (daily stats, mean, std_dev, etc.)
        # For histograms of entire dataset, we need all in memory
        if self.create_figs:
            n_plots = len(agg_stats_dict.keys()) - 1
            print(f"Plotting {n_plots} histograms for daily stats...")
            fig, ax = plt.subplots(2, n_plots//2, figsize=(12, 8))
            fig.suptitle(f'{self.data_type} {self.var.capitalize()} {self.split_type} Daily Stats Distribution', fontsize=14)
            axes = ax.flatten()
            for i, k in enumerate(agg_stats_dict.keys()):
                if k == 'date':
                    continue
                axes[i-1].hist(agg_stats_dict[k], bins=100, alpha=0.7)
                axes[i-1].set_title(f'{k.capitalize()} Distribution')
                axes[i-1].set_xlabel(k.capitalize())
                axes[i-1].set_ylabel('Frequency')

            fig.tight_layout()

            if self.save_figs:
                out_path = os.path.join(self.fig_path, f'{self.var}_{self.split_type}_{self.data_type}_{self.transformation_str}_{self.time_agg}_stats.png')
                fig.savefig(out_path, dpi=300, bbox_inches='tight')
            if self.show_figs:
                plt.show(block=False)
                plt.pause(5)
                plt.close(fig)
            else:
                plt.close(fig)



    def run(self):
        '''
            Method to run the DataStats class.
            Calls the load_data, apply_transformations and visualize_data methods.
        '''
        all_data_list, stats_dict = self.load_data(plot_cutout=True)
        self.visualize_data(all_data_list, stats_dict)

    




if __name__ == '__main__':
    data_stats_from_args()








# def data_stats_from_args():
#     parser = argparse.ArgumentParser(description='Compute statistics of the data')
#     parser.add_argument('--HR_VAR', type=str, default='temp', help='The high resolution variable')
#     parser.add_argument('--HR_SIZE', type=int, default=64, help='The shape of the high resolution data')
#     parser.add_argument('--data_type', type=str, default='DANRA', help='The dataset to compute statistics for (DANRA or ERA5)')
#     parser.add_argument('--split_type', type=str, default='train', help='The split type of the data (train, val, test)')
#     parser.add_argument('--path_data', type=str, default='/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/', help='The path to the data')
#     parser.add_argument('--create_figs', type=bool, default=True, help='Whether to create figures')
#     parser.add_argument('--save_figs', type=bool, default=False, help='Whether to save the figures')
#     parser.add_argument('--show_figs', type=bool, default=True, help='Whether to show the figures')
#     parser.add_argument('--path_save', type=str, default='/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/', help='The path to save the figures')
    
#     args = parser.parse_args()
#     data_stats(args)

# def compute_statistics(data):
#     # Compute statistics of 2D data
#     mean = np.mean(data)
#     median = np.median(data)
#     std_dev = np.std(data)
#     variance = np.var(data)
#     min_temp = np.min(data)
#     max_temp = np.max(data)
#     percentiles = np.percentile(data, [25, 50, 75])
    
#     return mean, median, std_dev, variance, min_temp, max_temp, percentiles

# def data_stats(args):
#     var = args.HR_VAR
#     data_type = args.data_type
#     split_type = args.split_type

#     if var == 'temp':
#         var_str = 't'
#         cmap = 'plasma'
#     elif var == 'prcp':
#         var_str = 'tp'
#         cmap = 'inferno'

#     danra_size_str = '589x789'
#     cutout = [170, 170+180, 340, 340+180]

#     PATH_HR = args.path_data + 'data_DANRA/size_' + danra_size_str + '/' + var + '_' + danra_size_str +  '/zarr_files/'
#     data_dir_zarr = PATH_HR + split_type + '.zarr'

#     print('\nOpening zarr directory:')
#     print(data_dir_zarr)

#     zarr_group_img = zarr.open_group(data_dir_zarr, mode='r')
#     files = list(zarr_group_img.keys())

#     all_data = []

#     df_stats = {file: {} for file in files}

#     if args.save_figs:
#         if not os.path.exists(args.path_save):
#             os.makedirs(args.path_save)

#     print(f'\n\nNumber of files: {len(files)}')

#     for idx, file in enumerate(files):
#         if idx % 10 == 0:
#             print(f'\n\nProcessing File {idx+1}/{len(files)}')

#         try:
#             data = zarr_group_img[file][var_str][:].squeeze()
#         except:
#             data = zarr_group_img[file]['data'][:].squeeze()

#         data = data[cutout[0]:cutout[1], cutout[2]:cutout[3]]
        
#         if var == 'temp':
#             data = data - 273.15
#         all_data.append(data)

#         mean, median, std_dev, variance, min_temp, max_temp, percentiles = compute_statistics(data)
#         df_stats[file] = {
#             'mean': mean,
#             'median': median,
#             'std_dev': std_dev,
#             'variance': variance,
#             'min': min_temp,
#             'max': max_temp,
#             'percentiles': percentiles
#         }

#         if idx == 0 and args.create_figs:
#             fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#             ax.imshow(data, cmap=cmap)
#             cbar = plt.colorbar(ax.imshow(data, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
#             ax.set_xticks([])
#             ax.set_yticks([])
#             ax.invert_yaxis()

#             if args.save_figs:
#                 fig.savefig(args.path_save + f'{var}_{split_type}_{data_type}_cutout_example.png', dpi=600, bbox_inches='tight')

#     all_data = np.concatenate(all_data, axis=0).flatten()
#     print(f'Number of data points: {len(all_data.flatten())}')

#     # if var == 'prcp':
#     #     # yeojohnson_transformed_data, yeojohnson_lambda = yeojohnson(all_data)
#     #     # all_data_no_zeros = all_data.copy()
#     #     # all_data_no_zeros[all_data_no_zeros <= 0] = 1e-4
#     #     # boxcox_transformed_data, boxcox_lambda = boxcox(all_data_no_zeros)
#     #     log_transformed_data = PrcpLogTransform()

        
#     #     if args.create_figs:
#     #         fig, ax = plt.subplots(1, 1, figsize=(8, 5))
#     #         # ax.hist(all_data_no_zeros, bins=100, alpha=0.7, label=f'Original, mu={np.mean(all_data_no_zeros):.2f}, std={np.std(all_data_no_zeros):.2f}')
#     #         # ax.hist(yeojohnson_transformed_data, bins=100, alpha=0.7, label=f'Yeo-Johnson, mu={np.mean(yeojohnson_transformed_data):.2f}, std={np.std(yeojohnson_transformed_data):.2f}')
#     #         # ax.hist(boxcox_transformed_data, bins=100, alpha=0.7, label=f'Box-Cox, mu={np.mean(boxcox_transformed_data)::.2f}, std={np.std(boxcox_transformed_data):.2f}')
#     #         ax.hist(log_transformed_data, bins=100, alpha=0.7, label=f'Log-transformed')
#     #         # ax.set_yscale('log')

#     #         if var == 'temp':
#     #             ax.set_xlabel('Temperature [C]')
#     #         elif var == 'prcp':
#     #             ax.set_xlabel('Precipitation [mm]')
#     #         ax.set_ylabel('Frequency')
#     #         ax.legend()

#     #         if args.save_figs:
#     #             fig.savefig(args.path_save + f'{var}_{split_type}_{data_type}_log_transformed_data.png', dpi=600, bbox_inches='tight')

#     print('\n\nGLOBAL STATISTICS BEFORE TRANSFORMATION\n\n')
#     mean, median, std_dev, variance, min_temp, max_temp, percentiles = compute_statistics(all_data)
#     print('Global statistics')
#     print('Mean: ', mean)
#     print('Median: ', median)
#     print('Standard Deviation: ', std_dev)
#     print('Variance: ', variance)
#     print('Min: ', min_temp)
#     print('Max: ', max_temp)
#     print('Percentiles: ', percentiles)

#     if var == 'temp'
#         all_data_zScore = (all_data - mean) / (std_dev + 1e-8)
#     elif var == 'prcp':
#         all_data_zScore = torch.log(all_data + 1e-8)

#     if args.create_figs:
#         fig, ax = plt.subplots(1, 1, figsize=(8, 5))
#         ax.hist(all_data, bins=100, alpha=0.7, label=f'Original, mu={mean:.2f}, std={std_dev:.2f}')
#         if var == 'temp':
#             ax.set_xlabel('Temperature [C]')
#         elif var == 'prcp':
#             ax.set_xlabel('Precipitation [mm]')
#         ax.set_ylabel('Frequency')
#         ax.hist(all_data_zScore, bins=100, alpha=0.7, label=f'Z-Score Normalized, mu={np.mean(all_data_zScore):.2f}, std={np.std(all_data_zScore):.2f}')
#         # if var == 'prcp':
#         #     mean_bc, std_bc = boxcox_transformed_data.mean(), boxcox_transformed_data.std()
#         #     bc_zScore = (boxcox_transformed_data - mean_bc) / (std_bc + 1e-8)

#         #     ax.hist(boxcox_transformed_data, bins=100, alpha=0.7, label=f'Box-Cox Transformed, mu={mean_bc:.2f}, std={std_bc:.2f}')
#         #     ax.hist(bc_zScore, bins=100, alpha=0.7, label=f'Box-Cox Z-Score Normalized, mu={np.mean(bc_zScore)::.2f}, std={np.std(bc_zScore):.2f}')
#         #     ax.set_yscale('log')
#         ax.legend()
#         fig.tight_layout()

#         if args.save_figs:
#             fig.savefig(args.path_save + f'{var}_{split_type}_{data_type}_all_data.png', dpi=600, bbox_inches='tight')

#         n_plots = 6
#         fig, ax = plt.subplots(2, n_plots//2, figsize=(10, 7))
        
#         stats_columns = ['mean', 'median', 'std_dev', 'variance', 'min', 'max']
#         for i, col in enumerate(stats_columns):
#             ax_i = ax.flatten()[i]
#             ax_i.hist([df_stats[file][col] for file in df_stats], bins=100, alpha=0.7)
#             ax_i.set_title(col)
#             ax_i.set_xlabel(col)
#             ax_i.set_ylabel('Frequency')

#         fig.tight_layout()

#         if args.save_figs:
#             fig.savefig(args.path_save + f'{var}_{split_type}_{data_type}_stats_distributions.png', dpi=600, bbox_inches='tight')

#     print('\n\nGLOBAL STATISTICS AFTER TRANSFORMATIONS\n\n')

#     if var == 'temp':
#         mean, median, std_dev, variance, min_temp, max_temp, percentiles = compute_statistics(all_data_zScore)
#         print('\nGlobal statistics')
#         print('Mean: ', mean)
#         print('Median: ', median)
#         print('Standard Deviation: ', std_dev)
#         print('Variance: ', variance)
#         print('Min: ', min_temp)
#         print('Max: ', max_temp)
#         print('Percentiles: ', percentiles)

#     elif var == 'prcp':
#         mean, median, std_dev, variance, min_temp, max_temp, percentiles = compute_statistics(all_data_zScore)
#         print('\nGlobal statistics')
#         print('Mean: ', mean)
#         print('Median: ', median)
#         print('Standard Deviation: ', std_dev)
#         print('Variance: ', variance)
#         print('Min: ', min_temp)
#         print('Max: ', max_temp)
#         print('Percentiles: ', percentiles)

#         # mean, median, std_dev, variance, min_temp, max_temp, percentiles = compute_statistics(yeojohnson_transformed_data)
#         # print('\nYeo Johnson Transformed')
#         # print('Mean: ', mean)
#         # print('Median: ', median)
#         # print('Standard Deviation: ', std_dev)
#         # print('Variance: ', variance)
#         # print('Min: ', min_temp)
#         # print('Max: ', max_temp)
#         # print('Percentiles: ', percentiles)

#         # mean, median, std_dev, variance, min_temp, max_temp, percentiles = compute_statistics(boxcox_transformed_data)
#         # print('\nBox-Cox Transformed')
#         # print('Mean: ', mean)
#         # print('Median: ', median)
#         # print('Standard Deviation: ', std_dev)
#         # print('Variance: ', variance)
#         # print('Min: ', min_temp)
#         # print('Max: ', max_temp)
#         # print('Percentiles: ', percentiles)

#         # mean, median, std_dev, variance, min_temp, max_temp, percentiles = compute_statistics(bc_zScore)
#         # print('\nBox-Cox Z-Score Normalized')
#         # print('Mean: ', mean)
#         # print('Median: ', median)
#         # print('Standard Deviation: ', std_dev)
#         # print('Variance: ', variance)
#         # print('Min: ', min_temp)
#         # print('Max: ', max_temp)
#         # print('Percentiles: ', percentiles)

#     if args.show_figs:
#         plt.show()

#     return 

# def data_stats_comparison(args):
#     '''
#         Function to compare distributions of two datasets (ERA5 and DANRA)
#     '''
#     return


# if __name__ == '__main__':
#     data_stats_from_args()


# import os
# import zarr     
# import argparse
# import numpy as np
# # import pandas as pd
# import matplotlib.pyplot as plt
# from data_modules import DANRA_Dataset_cutouts_ERA5_Zarr
# from scipy.stats import boxcox, yeojohnson
# from scipy.optimize import minimize_scalar


# def data_stats_from_args():

#     parser = argparse.ArgumentParser(description='Compute statistics of the data')
#     parser.add_argument('--HR_SIZE', type=str, default='temp', help='The variable to compute statistics for')
#     parser.add_argument('--data_type', type=str, default='DANRA', help='The dataset to compute statistics for (DANRA or ERA5)')
#     parser.add_argument('--split_type', type=str, default='train', help='The split type of the data (train, val, test)')
#     parser.add_argument('--path_data', type=str, default='/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/', help='The path to the data')
#     parser.add_argument('--create_figs', type=bool, default=True, help='Whether to create figures')
#     parser.add_argument('--save_figs', type=bool, default=False, help='Whether to save the figures')
#     parser.add_argument('--show_figs', type=bool, default=True, help='Whether to show the figures')
#     parser.add_argument('--path_save', type=str, default='/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/', help='The path to save the figures')
    
#     args = parser.parse_args()

#     data_stats(args)


# def compute_statistics(data):
#     # Compute statistics of 2D data
#     mean = np.mean(data)
#     median = np.median(data)
#     std_dev = np.std(data)
#     variance = np.var(data)
#     min_temp = np.min(data)
#     max_temp = np.max(data)
#     percentiles = np.percentile(data, [25, 50, 75])

    
#     return mean, median, std_dev, variance, min_temp, max_temp, percentiles


# def data_stats(args):
#     '''
#         Based on arguments check if data exists and in the right format
#         If not, create right format
#     '''
#     var = args.HR_SIZE
#     data_type = args.data_type
#     split_type = args.split_type


#     if var == 'temp':
#         var_str = 't'
#         cmap = 'plasma'
#     elif var == 'prcp':
#         var_str = 'tp'
#         cmap = 'inferno'

#     danra_size_str = '589x789'

#     cutout = [170, 170+180, 340, 340+180]

#     data_dir = args.path_data #'/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/'
#     # To HR data: Path + '/data_DANRA/size_589x789/' + var + '_' + danra_size_str +  '/zarr_files/train.zarr'
#     PATH_HR = data_dir + 'data_' + data_type + '/size_' + danra_size_str + '/' + var + '_' + danra_size_str +  '/zarr_files/'
#     # Path to DANRA data (zarr files), full danra, to enable cutouts
#     data_dir_zarr = PATH_HR + split_type + '.zarr'

#     zarr_group_img = zarr.open_group(data_dir_zarr, mode='r')

#     files = list(zarr_group_img.keys())

#     all_data = []

#     # Create a df to store daily stats of the data (mean, median, std_dev, variance, min, max, percentiles)
#     df_stats = pd.DataFrame(columns=['mean', 'median', 'std_dev', 'variance', 'min', 'max', 'percentiles'], index=files)

#     if args.save_figs:
#         if not os.path.exists(args.path_save):
#             os.makedirs(args.path_save)


#     print(f'\n\nNumber of files: {len(files)}')
#     # Get the data (all files)
#     for idx, file in enumerate(files):
#         if idx % 10 == 0:
#             print(f'\n\nProcessing File {idx+1}/{len(files)}')

#         try:
#             data = zarr_group_img[file][var_str][:].squeeze()
#         except:
#             data = zarr_group_img[file]['data'][:].squeeze()

#         data = data[cutout[0]:cutout[1], cutout[2]:cutout[3]]
#         # Convert to Celsius if temperature
#         if var == 'temp':
#             data = data - 273.15
#         all_data.append(data)


#         # Compute statistics
#         mean, median, std_dev, variance, min_temp, max_temp, percentiles = compute_statistics(data)
#         df_stats.loc[file] = [mean, median, std_dev, variance, min_temp, max_temp, percentiles]

#         if idx == 0 and args.create_figs:
#             fig, ax = plt.subplots(1, 1, figsize=(5, 5), layout='tight')
#             ax.imshow(data, cmap=cmap)
#             # Add colorbar
#             cbar = plt.colorbar(ax.imshow(data, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
#             # Remove ticks
#             ax.set_xticks([])
#             ax.set_yticks([])
#             # Flip the y-axis to match the image
#             ax.invert_yaxis()

#             if args.save_figs:
#                 fig.savefig(args.path_save + f'{var}_{split_type}_cutout_example.png', dpi=600, bbox_inches='tight')
            




#     # Concatenate the data
#     all_data = np.concatenate(all_data, axis=0).flatten()
#     print(f'Number of data points: {len(all_data.flatten())}')

#     # If precipitation, transform the data
#     if var == 'prcp':
#         # Yeo-Johnson transform
#         yeojohnson_transformed_data, yeojohnson_lambda = yeojohnson(all_data)
#         # Replace negative values with 0 for boxcox
#         all_data_no_zeros = all_data.copy()
#         all_data_no_zeros[all_data_no_zeros <= 0] = 1e-4
#         boxcox_transformed_data, boxcox_lambda = boxcox(all_data_no_zeros)
#         # Square root transform
#         #all_data_sqrt = np.sqrt(all_data_no_zeros)
        
#         if args.create_figs:
#             # Plot different transforms
#             fig, ax = plt.subplots(1, 1, figsize=(8,5))
#             ax.hist(all_data_no_zeros, bins=100, alpha=0.7, label=f'Original, mu={np.mean(all_data_no_zeros):.2f}, std={np.std(all_data_no_zeros):.2f}')
#             #ax.hist(all_data_sqrt, bins=100, alpha=0.7, label='Square Root')
#             ax.hist(yeojohnson_transformed_data, bins=100, alpha=0.7, label=f'Yeo-Johnson, mu={np.mean(yeojohnson_transformed_data):.2f}, std={np.std(yeojohnson_transformed_data):.2f}')
#             ax.hist(boxcox_transformed_data, bins=100, alpha=0.7, label=f'Box-Cox, mu={np.mean(boxcox_transformed_data):.2f}, std={np.std(boxcox_transformed_data):.2f}')
#             # Set log scale for y-axis
#             ax.set_yscale('log')
            

#             if var == 'temp':
#                 ax.set_xlabel('Temperature [C]')
#             elif var == 'prcp':
#                 ax.set_xlabel('Precipitation [mm]')
#             ax.set_ylabel('Frequency')
#             ax.legend()

#             if args.save_figs:
#                 fig.savefig(args.path_save + f'{var}_{split_type}_transformed_data.png', dpi=600, bbox_inches='tight')


#     print('\n\nGLOBAL STATISTICS BEFORE TRANSFORMATION\n\n')
#     # Compute global statistics
#     mean, median, std_dev, variance, min_temp, max_temp, percentiles = compute_statistics(all_data)
#     print('Global statistics')
#     print('Mean: ', mean)
#     print('Median: ', median)
#     print('Standard Deviation: ', std_dev)
#     print('Variance: ', variance)
#     print('Min: ', min_temp)
#     print('Max: ', max_temp)
#     print('Percentiles: ', percentiles)

#     # Pool all pixels together and plot
#     all_data = all_data.flatten()

#     all_data_zScore = (all_data - mean) / (std_dev + 1e-8) # Add a small number to avoid division by zero

#     if args.create_figs:
#         fig, ax = plt.subplots(1, 1, figsize=(8,5))
#         ax.hist(all_data, bins=100, alpha=0.7, label=f'Original, mu={mean:.2f}, std={std_dev:.2f}')
#         if var == 'temp':
#             ax.set_xlabel('Temperature [C]')
#         elif var == 'prcp':
#             ax.set_xlabel('Precipitation [mm]')
#         ax.set_ylabel('Frequency')
#         ax.hist(all_data_zScore, bins=100, alpha=0.7, label=f'Z-Score Normalized, mu={np.mean(all_data_zScore):.2f}, std={np.std(all_data_zScore):.2f}')
#         if var == 'prcp':
#             mean_bc, std_bc = boxcox_transformed_data.mean(), boxcox_transformed_data.std()
#             bc_zScore = (boxcox_transformed_data - mean_bc) / (std_bc + 1e-8)

#             ax.hist(boxcox_transformed_data, bins=100, alpha=0.7, label=f'Box-Cox Transformed, mu={mean_bc:.2f}, std={std_bc:.2f}')
#             ax.hist(bc_zScore, bins=100, alpha=0.7, label=f'Box-Cox Z-Score Normalized, mu={np.mean(bc_zScore):.2f}, std={np.std(bc_zScore):.2f}')

#             ax.set_yscale('log')
#         ax.legend()
#         fig.tight_layout()

#         if args.save_figs:
#             fig.savefig(args.path_save + f'{var}_{split_type}_all_data.png', dpi=600, bbox_inches='tight')



#         # Plot the distributions of individual means, medians, std_devs, variances, mins, maxs
#         n_plots = 6
#         fig, ax = plt.subplots(2, n_plots//2, figsize=(10, 7))
        
#         for i in range(n_plots):
#             ax_i = ax.flatten()[i]
#             ax_i.hist(df_stats.iloc[:, i], bins=100, alpha=0.7)
#             ax_i.set_title(df_stats.columns[i])
#             ax_i.set_xlabel(df_stats.columns[i])
#             ax_i.set_ylabel('Frequency')

#         fig.tight_layout()

#         if args.save_figs:
#             fig.savefig(args.path_save + f'{var}_{split_type}_stats_distributions.png', dpi=600, bbox_inches='tight')



#     print('\n\nGLOBAL STATISTICS AFTER TRANSFORMATIONS\n\n')

#     # Compute global statistics on the transformed data
#     if var == 'temp':
#         mean, median, std_dev, variance, min_temp, max_temp, percentiles = compute_statistics(all_data_zScore)
#         print('\nGlobal statistics')
#         print('Mean: ', mean)
#         print('Median: ', median)
#         print('Standard Deviation: ', std_dev)
#         print('Variance: ', variance)
#         print('Min: ', min_temp)
#         print('Max: ', max_temp)
#         print('Percentiles: ', percentiles)

#     elif var == 'prcp':
#         mean, median, std_dev, variance, min_temp, max_temp, percentiles = compute_statistics(all_data_zScore)
#         print('\nGlobal statistics')
#         print('Mean: ', mean)
#         print('Median: ', median)
#         print('Standard Deviation: ', std_dev)
#         print('Variance: ', variance)
#         print('Min: ', min_temp)
#         print('Max: ', max_temp)
#         print('Percentiles: ', percentiles)

#         mean, median, std_dev, variance, min_temp, max_temp, percentiles = compute_statistics(yeojohnson_transformed_data)
#         print('\nYeo Johnson Transformed')
#         print('Mean: ', mean)
#         print('Median: ', median)
#         print('Standard Deviation: ', std_dev)
#         print('Variance: ', variance)
#         print('Min: ', min_temp)
#         print('Max: ', max_temp)
#         print('Percentiles: ', percentiles)

#         mean, median, std_dev, variance, min_temp, max_temp, percentiles = compute_statistics(boxcox_transformed_data)
#         print('\nBox-Cox Transformed')
#         print('Mean: ', mean)
#         print('Median: ', median)
#         print('Standard Deviation: ', std_dev)
#         print('Variance: ', variance)
#         print('Min: ', min_temp)
#         print('Max: ', max_temp)
#         print('Percentiles: ', percentiles)

#         mean, median, std_dev, variance, min_temp, max_temp, percentiles = compute_statistics(bc_zScore)
#         print('\nBox-Cox Z-Score Normalized')
#         print('Mean: ', mean)
#         print('Median: ', median)
#         print('Standard Deviation: ', std_dev)
#         print('Variance: ', variance)
#         print('Min: ', min_temp)
#         print('Max: ', max_temp)
#         print('Percentiles: ', percentiles)





#     if args.show_figs:
#         plt.show()


#     return 




# if __name__ == '__main__':
#     data_stats_from_args()



# '''
#     This script is used to examine the data and get some statistics about it.
#     Data available: temperature, precipitation, water vapor flux.
#     Types of statistics: mean, std, min, max, median, 25th and 75th percentile.
#     Statistics are calculated for:
#     - Full dataset statistics (for normalization)
#     - Daily statistics (for time series and temporal statistics)
#     - Pointwise statistics (for spatial statistics)
#     - Climatology 

# '''

# import zarr
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import netCDF4 as nc


# class dataset_statistics():
#     '''
#         Class to examine given dataset and get statistics about it.
#         Can take temperature, precipitation, water vapor flux and other relevant data.
#         Data needs to be in daily zarr format. If a larger domain is available, the data
#         will be cropped to the desired size, centered around Denmark. 
#     '''
#     def __init__(self,
#                  data_dir_zarr:str,
#                  data_size:int,
#                  variable:str,
#                  var_str:str = 't2m',
#                  dataset_str:str = 'DANRA',
#                  cutout:list = [170, 170+180, 340, 340+180]
#                  ):
#         '''
#             Initialize the dataset statistics class.
#         '''
#         self.data_dir_zarr = data_dir_zarr
#         self.data_size = data_size
#         self.variable = variable
#         self.var_str = var_str
#         self.dataset_str = dataset_str
#         self.data_group = zarr.open_group(data_dir_zarr, mode='r')
#         self.files = list(self.data_group.keys())
#         self.n_samples = len(self.files)
#         self.cache_size = self.n_samples
#         self.image_size = (self.data_size, self.data_size)
#         self.cutout = cutout

#     def get_daily_statistics(self):
#         '''
#             Get full dataset statistics.
#         '''
#         # Initialize empty list and dataframe to store data and statistics
#         all_data = []
#         df_stats = pd.DataFrame(columns = ['mean', 'std', 'min', 'max', 'median', 'percentiles'],index = self.files)

#         print(f'Number of files to process: {self.n_samples}')

#         # Get all data
        
#         for i, file in enumerate(self.files):
#             if i % 100 == 0:
#                 print(f'Processing file {i} of {self.n_samples}')
#             try:
#                 data = self.data_group[file][self.var_str][:].squeeze()
#             except:
#                 data = self.data_group[file]['data'][:].squeeze()

#             data = data[self.cutout[0]:self.cutout[1], self.cutout[2]:self.cutout[3]]

#             if self.variable == 'temp':
#                 data = data - 273.15
#             elif self.variable == 'prcp' and self.dataset_str == 'DANRA':
#                 data = data / 1000

#             all_data.append(data)

#             # Compute statistics
#             mean, std, min_val, max_val, median, percentiles = self.compute_statistics(data)
#             df_stats.loc[file] = [mean, std, min_val, max_val, median, percentiles]

#         all_data = np.concatenate(all_data, axis = 0).flatten()
#         return all_data, df_stats

#     def get_full_dataset_statistics(self):

#         # Call get_daily_statistics to get statistics
#         all_data, df_stats = self.get_daily_statistics()

#         # Compute statistics
#         mean, std, min_val, max_val, median, percentiles = self.compute_statistics(all_data)

#         return mean, std, min_val, max_val, median, percentiles



        


#     def compute_statistics(data):
#         '''
#             Compute statistics for given data.
#         '''
#         mean = np.mean(data)
#         std = np.std(data)
#         min_val = np.min(data)
#         max_val = np.max(data)
#         median = np.median(data)
#         percentiles = np.percentile(data, [25, 50, 75])

#         return mean, std, min_val, max_val, median, percentiles
    

# if __name__ == '__main__':
#     var = 'temp'
#     # Set paths to zarr data
#     path_data = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/'
#     danra_size_str = '589x789'
#     dataset_str = 'DANRA'

#     data_dir_zarr = path_data + 'data_' + dataset_str + '/size_' + danra_size_str + '/' + var + '_' + danra_size_str + '/zarr_files/'

#     datasplit_str = 'test'

#     full_path = data_dir_zarr + datasplit_str + '.zarr'

#     # # Plot some statistics
#     # plt.imshow(mean)
#     # plt.colorbar()
#     # plt.show()
#     # plt.imshow(std)
#     # plt.colorbar()
#     # plt.show()
#     # plt.imshow(min_val)
#     # plt.colorbar()
#     # plt.show()
#     # plt.imshow(max_val)
#     # plt.colorbar()
#     # plt.show()
#     # plt.imshow(median)
#     # plt.colorbar()
#     # plt.show()
#     # plt.imshow(percentile_25)
#     # plt.colorbar()
#     # plt.show()
#     # plt.imshow(percentile_75)
#     # plt.colorbar()
#     # plt.show()
    
#     # # Save statistics
#     # np.save('mean.npy', mean)
#     # np.save('std.npy', std)
#     # np.save('min.npy', min_val)
#     # np.save('max.npy', max_val)
#     # np.save('median.npy', median)
#     # np.save('percentile_25.npy', percentile_25)
#     # np.save('percentile_75.npy', percentile_75)




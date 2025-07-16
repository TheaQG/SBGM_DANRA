"""
This script creates a small data batch for testing purposes.
It takes a number of .npz or .nc files from all variable directories
and creates a small data batch with a specified number of files.
The small data batch is then saved under e.g. 'Data_DiffMod/data_DANRA/size_589x789/all_small/'
which can then be used to generate 'train_small', 'val_small' and 'test_small' datasets.
The small data batch is created by taking a random sample of files from the original data batch
and saving them in a new directory, then converting them to zarr format.

Functions:
    - select_sample_files: Selects a random sample of files from the input directory (all samples dir).
        Input:  model_list              [list]  : List of models to create small data batch for
                var_list                [list]  : List of variables to create small data batch for
                dir_all_data__name      [str]   : Name of the directory containing the full dataset
                n_samples               [int]   : Number of files to select for the small data batch
                sample_method           [str]   : Method to select the sample files (random or sequential)
        Output: file_names              [list]  : List of file names selected for the small data batch
    - copy_to_dirs: Copies the selected files to the small data batch directory and splits them into train, val and test directories.
        Input:  model_list              [list]  : List of models to create small data batch for
                var_list                [list]  : List of variables to create small data batch for
                file_names              [list]  : List of file names selected for the small data batch
        Output: small_data_dir          [str]   : Directory containing the small data batch
    - dirs_to_zarr: Converts the small data batch directories to zarr format.
        Input:  model_list              [list]  : List of models to create small data batch for
                var_list                [list]  : List of variables to create small data batch for
                small_data_dirs         [str]   : Directories containing the small data batches (for each model and variable)
        Output: None
"""

import os
import shutil
import random
import numpy as np
import zarr
import argparse
import netCDF4 as nc
from daily_files_to_zarr import convert_npz_to_zarr, convert_nc_to_zarr
from utils import str2list


def select_sample_files(model_list=['DANRA', 'ERA5'],
                        var_list=['temp', 'prcp'],
                        n_samples_total=100,
                        sample_method='random',
                        data_path='Data/Data_DiffMod/',
                        all_data__dir_name='all_filtered',
                        all_data__dir_name_small='all_small',
                        ):
    """
    Selects a random sample of files from the input directory (all samples dir).
    Check that the selected files exist in all model and variable directories.
    Input:  model_list              [list]  : List of models to create small data batch for
            var_list                [list]  : List of variables to create small data batch for
            dir_all_data__name      [str]   : Name of the directory containing the full dataset. Needs to be added to a full data path specifying model and variable.
            n_samples               [int]   : Number of files to select for the small data batch
            sample_method           [str]   : Method to select the sample files (random or sequential)
    Output: file_names              [list]  : List of file names selected for the small data batch
    """

    # Check that the sample method is valid
    if sample_method not in ['random', 'sequential']:
        raise ValueError("sample_method must be either 'random' or 'sequential'")

    # Create a list to store the file names
    file_names = []

    # Use the first variable and model to get the selected file names
    var = var_list[0]
    model = model_list[0]
    # Get the path to the input directory
    input_dir = os.path.join(data_path, f"data_{model}/size_589x789/{var}_589x789/{all_data__dir_name}")
    # Get the list of files in the input directory
    all_files = os.listdir(input_dir)
    # Check if the input directory is empty
    if len(all_files) == 0:
        raise ValueError(f"Input directory {input_dir} is empty")
    # Check if the number of samples is greater than the number of files
    if n_samples_total > len(all_files):
        raise ValueError(f"Number of samples {n_samples_total} is greater than the number of files {len(all_files)}")
    # Select the sample files
    if sample_method == 'random':
        file_names = random.sample(all_files, n_samples_total)
    elif sample_method == 'sequential':
        file_names = all_files[:n_samples_total]

    # Strip the date from the file names and remove the file extension
    file_dates = [file_name.split('_')[-1].split('.')[0] for file_name in file_names]
    
    # Check that the selected files exist in all model and variable directories
    for model in model_list:
        for var in var_list:
            # Get the path to the input directory
            input_dir = os.path.join(data_path, f"data_{model}/size_589x789/{var}_589x789/{all_data__dir_name}")
            # Get the list of dates in the input directory
            all_dates_tmp = os.listdir(input_dir)
            all_dates_tmp = [file_name.split('_')[-1].split('.')[0] for file_name in all_dates_tmp]
            for file_date, file_name in zip(file_dates, file_names):
                if file_date not in all_dates_tmp:
                    raise ValueError(f"File {file_name} does not exist in {input_dir}")
                    # Remove the file name from the list
                    file_names.remove(file_name)

                    
    
    # Print the selected file names
    print(f"Selected {len(file_names)} files for the small data batch:")
    for file_name in file_names:
        print(file_name)

    # Return the list of file names
    return file_names

def split_and_copy_to_dirs(model_list=['DANRA', 'ERA5'],
                 var_list=['temp', 'prcp'],
                 file_names=[],
                 n_samples_total=100,
                 sample_method='random',
                 data_path='Data/Data_DiffMod/',
                 all_data__dir_name='all_filtered',
                 all_data__dir_name_small='all_small',
                 split_ratio=[0.7, 0.15, 0.15],
                 ):
    """
    Copies the selected files to the small data batch directory and splits
    them into train, val and test directories.
    Runs the function select_sample_files to get the selected file names if not provided.
    Input:  model_list                  [list]  : List of models to create small data batch for
            var_list                    [list]  : List of variables to create small data batch for
            file_names                  [list]  : List of file names selected for the small data batch
            data_path                   [str]   : Path to the data directory
            all_data__dir_name          [str]   : Name of the directory containing the full dataset
            all_data__dir_name_small    [str]   : Name of the directory containing the small dataset
    Output: small_data_dirs             [str]   : Directory containing the small data batch
    """

    # Check if the file names are provided, if not run select_sample_files
    if len(file_names) == 0:
        print(f'No file names provided, selecting {n_samples_total} files with {sample_method} method')
        file_names = select_sample_files(model_list=model_list,
                                         var_list=var_list,
                                         n_samples_total=n_samples_total,
                                         sample_method=sample_method,
                                         data_path=data_path,
                                         all_data__dir_name=all_data__dir_name,
                                         all_data__dir_name_small=all_data__dir_name_small)
    # Get the file dates from the file names
    file_dates = [file_name.split('_')[-1].split('.')[0] for file_name in file_names]

    # Split the file names into train, val and test sets based on the split ratio
    n_train = int(n_samples_total * split_ratio[0])
    n_val = int(n_samples_total * split_ratio[1])
    n_test = n_samples_total - n_train - n_val
    print(f"Splitting {n_samples_total} files into {n_train} train, {n_val} val and {n_test} test sets")

    train_files = file_names[:n_train]
    val_files = file_names[n_train:n_train + n_val]
    test_files = file_names[n_train + n_val:]

    train_dates = file_dates[:n_train]
    val_dates = file_dates[n_train:n_train + n_val]
    test_dates = file_dates[n_train + n_val:]
    
    
    # Create an empty dictionary to store dictionaries of models and variables
    # small_data_dirs = {'DANRA': {'temp': {}, 'prcp': {'train': "..."}},
    #                   'ERA5': {}}

    small_data_dirs = {}

    # Copy to new (empty) directories
    for model in model_list:
        small_data_dirs[model] = {}
        for var in var_list:
            # Set the prescript of the file names (depending on the model and variable)
            if model == 'DANRA':
                if var == 'temp':
                    file_prefix = 't2m_ave'
                elif var == 'prcp':
                    file_prefix = 'tp_tot'
            elif model == 'ERA5':
                if var == 'temp':
                    file_prefix = 'temp_589x789'
                elif var == 'prcp':
                    file_prefix = 'prcp_589x789'

            small_data_dirs[model][var] = {}
            # Create the small data batch directory 
            small_data_dir = os.path.join(data_path, f"data_{model}/size_589x789/{var}_589x789/{all_data__dir_name_small}")
            
            # Create the train, val and test directories
            os.makedirs(small_data_dir, exist_ok=True)
            

            train_dir = os.path.join(small_data_dir, 'train')
            val_dir = os.path.join(small_data_dir, 'valid')
            test_dir = os.path.join(small_data_dir, 'test')

            # Create and clear the train, val and test directories
            if os.path.exists(train_dir):
                shutil.rmtree(train_dir)
            os.makedirs(train_dir, exist_ok=True)
            if os.path.exists(val_dir):
                shutil.rmtree(val_dir)
            os.makedirs(val_dir, exist_ok=True)
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
            os.makedirs(test_dir, exist_ok=True)
            

            # Copy the files to the new directories (with prefix and date)
            for file_date in train_dates:
                file_name = f"{file_prefix}_{file_date}.npz"
                src = os.path.join(data_path, f"data_{model}/size_589x789/{var}_589x789/{all_data__dir_name}", file_name)
                dst = os.path.join(train_dir, file_name)
                if not os.path.exists(dst):
                    # Copy the file to the new directory NOT move it
                    shutil.copyfile(src, dst)
            for file_date in val_dates:
                file_name = f"{file_prefix}_{file_date}.npz"
                src = os.path.join(data_path, f"data_{model}/size_589x789/{var}_589x789/{all_data__dir_name}", file_name)
                dst = os.path.join(val_dir, file_name)
                if not os.path.exists(dst):
                    # Copy the file to the new directory NOT move it
                    shutil.copyfile(src, dst)
            for file_date in test_dates:
                file_name = f"{file_prefix}_{file_date}.npz"
                src = os.path.join(data_path, f"data_{model}/size_589x789/{var}_589x789/{all_data__dir_name}", file_name)
                dst = os.path.join(test_dir, file_name)
                if not os.path.exists(dst):
                    # Copy the file to the new directory NOT move it
                    shutil.copyfile(src, dst)
            small_data_dirs[model][var]['train'] = train_dir
            small_data_dirs[model][var]['valid'] = val_dir
            small_data_dirs[model][var]['test'] = test_dir

    # Print the small data batch directories
    print(f"Small data batch directories:")
    print(small_data_dirs)
    # Return the small data batch directories
    return small_data_dirs

def dirs_to_zarr(data_path,
                small_data_dirs,
                all_data__dir_name_small='all_small',
                ):
    """
    Converts the small data batch directories to zarr format.
    Input:  small_data_dirs         [str]   : Directories containing the small data batches (for each model and variable)
    Output: None
    """
    
    # Loop through all models and variables
    for model in small_data_dirs.keys():
        for var in small_data_dirs[model].keys():
            small_data_dir_zarr = os.path.join(data_path, f"data_{model}/size_589x789/{var}_589x789/{all_data__dir_name_small}_zarr")
            os.makedirs(small_data_dir_zarr, exist_ok=True)
            # Get the train, val and test directories
            train_dir = small_data_dirs[model][var]['train']
            val_dir = small_data_dirs[model][var]['valid']
            test_dir = small_data_dirs[model][var]['test']
            # Convert the directories to zarr format
            print(f"Converting {train_dir} to zarr format...")
            convert_npz_to_zarr(train_dir, os.path.join(small_data_dir_zarr, 'train.zarr'), VERBOSE=True)
            print(f"Converting {val_dir} to zarr format...")
            convert_npz_to_zarr(val_dir, os.path.join(small_data_dir_zarr, 'valid.zarr'), VERBOSE=True)
            print(f"Converting {test_dir} to zarr format...")
            convert_npz_to_zarr(test_dir, os.path.join(small_data_dir_zarr, 'test.zarr'), VERBOSE=True)


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Create a small data batch for testing purposes.')
    # Add the arguments
    parser.add_argument('--model_list', type=str2list, default=['DANRA', 'ERA5'], help='List of models to create small data batch for')
    parser.add_argument('--var_list', type=str2list, default=['temp', 'prcp'], help='List of variables to create small data batch for')
    parser.add_argument('--n_samples_total', type=int, default=100, help='Number of files to select for the small data batch')
    parser.add_argument('--sample_method', type=str, default='random', help='Method to select the sample files (random or sequential)')
    parser.add_argument('--data_path', type=str, default='Data/Data_DiffMod/', help='Path to the data directory')
    parser.add_argument('--all_data__dir_name', type=str, default='all_filtered', help='Name of the directory containing the full dataset')
    parser.add_argument('--all_data__dir_name_small', type=str, default='all_small', help='Name of the directory containing the small dataset')
    parser.add_argument('--split_ratio', type=str2list, default=[0.7, 0.15, 0.15], help='Split ratio for train, val and test sets')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function to create the small data batch
    small_data_dirs = split_and_copy_to_dirs(model_list=args.model_list,
                                             var_list=args.var_list,
                                             n_samples_total=args.n_samples_total,
                                             sample_method=args.sample_method,
                                             data_path=args.data_path,
                                             all_data__dir_name=args.all_data__dir_name,
                                             all_data__dir_name_small=args.all_data__dir_name_small,
                                             split_ratio=args.split_ratio)
    
    # Convert the small data batch directories to zarr format
    dirs_to_zarr(data_path=args.data_path,
                 small_data_dirs=small_data_dirs,
                 all_data__dir_name_small=args.all_data__dir_name_small)
        
    print("Small data batch created and converted to zarr format.")
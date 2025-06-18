'''
    Utility functions for the project.

    Functions:
    ----------
    - model_summary: Print the model summary
    - SimpleLoss: Simple loss function
    - HybridLoss: Hybrid loss function
    - SDFWeightedMSELoss: Custom loss function for SDFs
    - convert_npz_to_zarr: Convert DANRA .npz files to zarr files
    - create_concatenated_data_files: Create concatenated data files
    - convert_npz_to_zarr_based_on_time_split: Convert DANRA .npz files to zarr files based on a time split
    - convert_npz_to_zarr_based_on_percent_split: Convert DANRA .npz files to zarr files based on a percentage split
    - convert_nc_to_zarr: Convert ERA5 .nc files to zarr files
    - extract_samples: Extract samples from dictionary
    
    For argparse:
    - str2bool: Convert string to boolean
    - str2list: Convert string to list
    - str2list_of_strings: Convert string to list of strings
    - str2dict: Convert string to dictionary

'''

import argparse
import torch 
import zarr
import os
import json
# import yaml

import netCDF4 as nc
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable



def model_summary(model):
    '''
        Simple function to print the model summary
    '''

    print("model_summary")
    print()
    print("Layer_name" + "\t"*7 + "Number of Parameters")
    print("="*100)
    
    model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
    layer_name = [child for child in model.children()]
    j = 0
    total_params = 0
    print("\t"*10)
    for i in layer_name:
        print()
        param = 0
        try:
            bias = (i.bias is not None)
        except:
            bias = False  
        if not bias:
            param =model_parameters[j].numel()+model_parameters[j+1].numel()
            j = j+2
        else:
            param =model_parameters[j].numel()
            j = j+1
        print(str(i) + "\t"*3 + str(param))
        total_params+=param
    print("="*100)
    print(f"Total Params:{total_params}")     


class SimpleLoss(nn.Module):
    def __init__(self):
        super(SimpleLoss, self).__init__()
        self.mse = nn.MSELoss()#nn.L1Loss()#$

    def forward(self, predicted, target):
        return self.mse(predicted, target)

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, T=10):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.T = T
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):
        loss = self.mse(predictions[-1], targets[0])
        
        for t in range(1, self.T):
            loss += self.alpha * self.mse(predictions[t-1], targets[t])
        
        return loss

class SDFWeightedMSELoss(nn.Module):
    '''
        Custom loss function for SDFs.

    '''
    def __init__(self, max_land_weight=1.0, min_sea_weight=0.5, device=None):
        super().__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.to(self.device)

        self.max_land_weight = max_land_weight
        self.min_sea_weight = min_sea_weight
#        self.mse = nn.MSELoss(reduction='none')

    def forward(self, input, target, sdf):
        # Convert SDF to weights, using a sigmoid function (or similar)
        # Scaling can be adjusted to control sharpness of transition
        weights = torch.sigmoid(sdf) * (self.max_land_weight - self.min_sea_weight) + self.min_sea_weight

        # Calculate the squared error
        input = input.to(self.device)
        target = target.to(self.device)
        squared_error = (input - target)**2

        squared_error = squared_error.to(self.device)
        weights = weights.to(self.device)

        # Apply the weights
        weighted_squared_error = weights * squared_error

        # Return mean of weighted squared error
        return weighted_squared_error.mean()
    

def convert_npz_to_zarr(npz_directory, zarr_file, VERBOSE=False):
    '''
        Function to convert DANRA .npz files to zarr files
        
        Parameters:
        -----------
        npz_directory: str
            Directory containing .npz files
        zarr_file: str
            Name of zarr file to be created
    '''
    print(f'\n\nConverting {len(os.listdir(npz_directory))} .npz files to zarr file...')

    # Create zarr group (equivalent to a directory) 
    zarr_group = zarr.open_group(zarr_file, mode='w')
    
    # Make iterator to keep track of progress
    i = 0

    # Loop through all .npz files in the .npz directory
    for npz_file in os.listdir(npz_directory):
        
        # Check if the file is a .npz file (not dir or .DS_Store)
        if npz_file.endswith('.npz'):
            if VERBOSE:
                print(os.path.join(npz_directory, npz_file))
            # Load the .npz file
            npz_data = np.load(os.path.join(npz_directory, npz_file))            

            # Loop through all keys in the .npz file
            for key in npz_data:

                if i == 0:
                    print(f'Key: {key}')
                # Save the data as a zarr array
                zarr_group.array(npz_file.replace('.npz', '') + '/' + key, npz_data[key], chunks=True, dtype=np.float32)
            
            # Print progress if iterator is a multiple of 100
            if (i+1) % 100 == 0:
                print(f'Converted {i+1} files...')
            i += 1


def create_concatenated_data_files(data_dir_all:list, data_dir_concatenated:str, variables:list, n_images:int=4):
    '''
        Function to create concatenated data files.
        The function will concatenate the data from the data_dir_all directories
        and save the concatenated data to the data_dir_concatenated directory.

        Parameters:
        -----------
        data_dir_all: list
            List of directories containing the data
        data_dir_concatenated: str
            Directory to save the concatenated data
        variables: list
            List of variables to concatenate
        n_images: int
            Number of images to concatenate
    '''
    print(f'\n\nCreating concatenated data files from {len(data_dir_all)} directories...')

    # Create zarr group (equivalent to a directory)
    zarr_group = zarr.open_group(data_dir_concatenated, mode='w')

    # Loop through all directories in the data_dir_all list
    for data_dir in data_dir_all:
        # Loop through all files in the directory
        for data_file in os.listdir(data_dir):
            # Check if the file is a .npz file (not dir or .DS_Store)
            if data_file.endswith('.npz'):
                # Load the .npz file
                npz_data = np.load(os.path.join(data_dir, data_file))
                # Loop through all variables in the .npz file
                for var in variables:
                    # Select the data from the variable
                    data = npz_data[var][:n_images]
                    # Save the data as a zarr array
                    zarr_group.array(data_file.replace('.npz', '') + '/' + var, data, chunks=True, dtype=np.float32)
    
    print(f'Concatenated data saved to {data_dir_concatenated}...')




class data_preperation():
    '''
        Class to handle data preperation for the project.
        All data is located in an /all/ directory.
        This class can handle:
            - Check if data already exists
            - Creating train, val and test splits (based on years or percentages) - and saving to zarr
            - Clean up after training (remove zarr files)

    '''
    def __init__(self, args):
        self.args = args
        self.data_dir_all = args.data_dir_all

    def create_concatenated_data_files(self, data_dir_all, data_dir_concatenated, n_images=4):
        '''
            Function to create concatenated data files.
            The function will concatenate the data from the data_dir_all directories
            and save the concatenated data to the data_dir_concatenated directory.

            Parameters:
            -----------
            data_dir_all: list
                List of directories containing the data
            data_dir_concatenated: str
                Directory to save the concatenated data
            n_images: int
                Number of images to concatenate
        '''
        print(f'\n\nCreating concatenated data files from {len(data_dir_all)} directories...')

        # Create zarr group (equivalent to a directory)
        zarr_group = zarr.open_group(data_dir_concatenated, mode='w')

        # Loop through all directories in the data_dir_all list
        for data_dir in data_dir_all:
            # Loop through all files in the directory
            for data_file in os.listdir(data_dir):
                # Check if the file is a .npz file (not dir or .DS_Store)
                if data_file.endswith('.npz'):
                    # Load the .npz file
                    npz_data = np.load(os.path.join(data_dir, data_file))
                    # Loop through all variables in the .npz file
                    for var in npz_data:
                        # Select the data from the variable
                        data = npz_data[var][:n_images]
                        # Save the data as a zarr array
                        zarr_group.array(data_file.replace('.npz', '') + '/' + var, data, chunks=True, dtype=np.float32)
        
        print(f'Concatenated data saved to {data_dir_concatenated}...')


def convert_npz_to_zarr_based_on_time_split(npz_directory:str, year_splits:list):
    '''
        Function to convert DANRA .npz files to zarr files based on a time split.
        The function will select the files based on the year_splits list.
        Will create train, val and test splits based on the years specified in the list.

        Parameters:
        -----------
        npz_directory: str
            Directory containing all .npz data files
        year_splits: list
            List of 3 lists containing the years for train, val and test splits
    '''
    # Print the years for each split
    # Print number of files in the split years
    print(f'\nTrain years: {year_splits[0]}')
    print(f'Number of files in train years: {len([f for f in os.listdir(npz_directory) if any(str(year) in f for year in year_splits[0])])}')

    print(f'\nVal years: {year_splits[1]}')
    print(f'Number of files in val years: {len([f for f in os.listdir(npz_directory) if any(str(year) in f for year in year_splits[1])])}')

    print(f'\nTest years: {year_splits[2]}')
    print(f'Number of files in test years: {len([f for f in os.listdir(npz_directory) if any(str(year) in f for year in year_splits[2])])}')

    # 



def convert_npz_to_zarr_based_on_percent_split(npz_directory:str, percent_splits:list, random_selection:bool=False):
    '''
        Function to convert DANRA .npz files to zarr files based on a percentage split.
        The function will randomly select the percentage of files specified in the percent_splits list
        or select the first files in the directory.
        Will create train, val and test splits based on the percentages specified in the list.
        
        Parameters:
        -----------
        npz_directory: str
            Directory containing .npz files
        percent_splits: list
            List of floats representing the percentage splits
    
    '''
    print(f'\n\nConverting {len(os.listdir(npz_directory))} .npz files to zarr file...')



def convert_nc_to_zarr(nc_directory, zarr_file, VERBOSE=False):
    '''
        Function to convert ERA5 .nc files to zarr files
        
        Parameters:
        -----------
        nc_directory: str
            Directory containing .nc files
        zarr_file: str
            Name of zarr file to be created
    '''
    print(f'Converting {len(os.listdir(nc_directory))} .nc files to zarr file...')
    # Create zarr group (equivalent to a directory)
    zarr_group = zarr.open_group(zarr_file, mode='w')
    
    # Loop through all .nc files in the .nc directory 
    for nc_file in os.listdir(nc_directory):
        # Check if the file is a .nc file (not dir or .DS_Store)
        if nc_file.endswith('.nc'):
            if VERBOSE:
                print(os.path.join(nc_directory, nc_file))
            # Load the .nc file
            nc_data = nc.Dataset(os.path.join(nc_directory, nc_file))
            # Loop through all variables in the .nc file
            for var in nc_data.variables:
                # Select the data from the variable
                data = nc_data[var][:]
                # Save the data as a zarr array
                zarr_group.array(nc_file.replace('.nc', '') + '/' + var, data, chunks=True, dtype=np.float32)

def extract_samples(samples, device=None):
    """
        Extract samples from the dictionary returned by the dataset class.
        Expected keys:
            - HR image: key ending with '_hr' (e.g. 'prcp_hr') [ignoring keys ending with '_original']
            - Classifier: key 'classifier'
            - LR conditions: key(s) ending with '_lr' (e.g. 'prcp_lr')
            - HR mask: key 'lsm_hr'
            - Land/sea mask: key 'lsm'
            - SDF: key 'sdf'
            - Topography: key 'topo'
            - Points: keys 'hr_point' and 'lr_point'
        If multiple LR condition keys are present, they are concatenated along the channel dimension
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # HR image (choose key ending with '_hr' not containing 'original')
    hr_keys = [k for k in samples.keys() if k.endswith('_hr') and not k.endswith('_original')]
    # If key 'lsm_hr' in hr_keys, remove it
    if 'lsm_hr' in hr_keys:
        hr_keys.remove('lsm_hr')
    
    if len(hr_keys) == 0:
        raise ValueError('No HR image found in samples dictionary.')
    hr_img = samples[hr_keys[0]].to(device).float()
    # if len(hr_keys) > 1:
    #     print(f'Warning: Multiple HR images found. Using the first one: {hr_keys[0]}')
    
    # Classifier (if available)
    classifier = samples.get('classifier', None)
    if classifier is not None:
        classifier = classifier.to(device)#.float()

    # LR conditions: if multiple, stack along channel dimensio
    lr_keys = [k for k in samples.keys() if k.endswith('_lr') and not k.endswith('_original')]
    if len(lr_keys) == 0:
        lr_img = None
    elif len(lr_keys) == 1:
        lr_img = samples[lr_keys[0]].to(device).float()
    else:
        lr_list = [samples[k].to(device).float() for k in sorted(lr_keys)]
        lr_img = torch.cat(lr_list, dim=1)

    # HR mask (LSM)
    lsm_hr = samples.get('lsm_hr', None)
    if lsm_hr is not None:
        lsm_hr = lsm_hr.to(device).float()
    
    # Land/sea mask (LSM)
    lsm = samples.get('lsm', None)
    if lsm is not None:
        lsm = lsm.to(device).float()
    
    # SDF
    sdf = samples.get('sdf', None)
    if sdf is not None:
        sdf = sdf.to(device).float()

    # Topography
    topo = samples.get('topo', None)
    if topo is not None:
        topo = topo.to(device).float()

    # HR crop points (if available)
    hr_points = samples.get('hr_point', None)
    if hr_points is not None:
        hr_points = hr_points.to(device).float()

    # LR crop points (if available)
    lr_points = samples.get('lr_point', None)
    if lr_points is not None:
        lr_points = lr_points.to(device).float()

    # Return all extracted samples
    return hr_img, classifier, lr_img, lsm_hr, lsm, sdf, topo, hr_points, lr_points


def build_data_path(base_path, model, var, full_domain_dims, split):
    """
    Construct a path for high-resolution data.
    Example: base_path + 'data_DANRA/size_589x789/temp_589x789/zarr_files/train.zarr'
    """
    return os.path.join(base_path, f"data_{model}", f"size_{full_domain_dims[0]}x{full_domain_dims[1]}", f"{var}_{full_domain_dims[0]}x{full_domain_dims[1]}", "zarr_files", f"{split}.zarr")


def str2bool(v):
    '''
        Function to convert string to boolean.
        Used for argparse.
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2list(v):
    """
    Convert a string to a list.

    If the string is in JSON list format (begins with '[' and ends with ']'),
    this function uses json.loads to parse it. Otherwise, it splits the string
    on commas and tries to convert each element to an integer, then float,
    and if neither conversion applies, leaves it as a string.
    
    Examples:
      - '[1,2,3]' --> [1, 2, 3]
      - '1,2,3'   --> [1, 2, 3]
      - '["a", "b", "c"]' --> ["a", "b", "c"]
      - '1.1,2.2,3.3' --> [1.1, 2.2, 3.3]
    """
    v = v.strip()
    if v.startswith('[') and v.endswith(']'):
        try:
            return json.loads(v)
        except Exception:
            # If JSON loading fails, fallback to splitting
            pass

    # If input is None, return None
    if v is None:
        return None

    # Fallback: split on commas and try to convert each element.
    items = [x.strip() for x in v.split(',')]
    result = []
    for x in items:
        try:
            result.append(int(x))
        except ValueError:
            try:
                result.append(float(x))
            except ValueError:
                result.append(x)
    return result    

# def str2list(v):
#     '''
#         Function to convert string to list.
#         Used for argparse.
#     '''
#     try:
#         # Try to split commas and convert each part to an integer
#         return [int(i) for i in v.split(',')]
#     except:
#         # If there is a ValueError, it means the list contains floats or strings
#         return [float(x) if '.' in x else x for x in v.split(',')]
    

def str2list_of_strings(v):
    # Split the input string by commas, strip any extra whitespace around the strings
    return [s.strip() for s in v.split(',')]

import ast
def str2dict(v):
    try:
        # Try to safely evaluate the string as a Python literal
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Invalid dictionary format.")


def build_data_path(base_path, model, var, full_domain_dims, split, zarr_file=True):
    """
    Construct a path for high-resolution data.
    Example: base_path + 'data_DANRA/size_589x789/temp_589x789/zarr_files/train.zarr'
    
    """
    if zarr_file:
        data_path = os.path.join(base_path, f"data_{model}", f"size_{full_domain_dims[0]}x{full_domain_dims[1]}", f"{var}_{full_domain_dims[0]}x{full_domain_dims[1]}", "zarr_files", f"{split}.zarr")
    else:
        data_path = os.path.join(base_path, f"data_{model}", f"size_{full_domain_dims[0]}x{full_domain_dims[1]}", f"{var}_{full_domain_dims[0]}x{full_domain_dims[1]}", f"{split}/")
    
    return data_path




def plot_sample(sample,
                hr_model,
                hr_units,
                lr_model,
                lr_units,
                var,
                show_ocean=False,
                force_matching_scale=False,
                global_min=None,
                global_max=None,
                extra_keys=None,
                hr_cmap='plasma',
                lr_cmap_dict=None, # e.g. {"prcp": "inferno", "temp": "plasma", ...}
                default_lr_cmap='inferno',
                extra_cmap_dict=None, # e.g. {"topo": "terrain", "sdf": "coolwarm","lsm": "binary"}
                figsize=(15, 4)):
    """
        Plot a single sample (dictionary from the dataset class) in a consistent layout
        
        Expected keys in sample:
            - HR image: f"{var}_hr" (and optionally f"{var}_hr_original")
            - LR condition(s): keys ending with "_lr" (and optionally "_lr_original")
            - HR mask for ocean masking: "lsm_hr" (used only for HR images)
            - Extra keys (e.g. geo variables) if provided via extra keys

        Parameters:
            - sample: Dictionary containing the sample
            - hr_model: String label for the HR model (e.g. "DANRA")
            - hr_units: Units for the HR image (e.g. "°C")
            - lr_model: String label for the LR model (e.g. "ERA5")
            - lr_units: A list of units for each LR condition (order must match the keys) e.g. ["°C", "mm/day"]
            - var: HR variable name (e.g. "temp" or "prcp")
            - show_ocean: Boolean to show ocean or not (i.e. masking HR images)
            - force_matching_scale: If True, and global_min/max are provided, use to set the colorscale - only for matching variables
            - global_min, global_max: Dictionaries mapping keys (e.g. "prcp_hr", "prcp_lr") to scalar min and max
            - extra_keys: List of extra keys to plot (e.g. "topo", "sdf")
            - hr_cmap: Colormap for HR images (default: 'plasma')
            - lr_cmap_dict: Dictionary mapping LR variable base names to colormaps (e.g. {"prcp": "inferno", "temp": "plasma"})
            - default_lr_cmap: Default colormap for LR images (if not specified in lr_cmap_dict)
            - extra_cmap_dict: Dictionary mapping extra keys to colormaps (e.g. {"topo": "terrain", "sdf": "coolwarm","lsm": "binary"})
            - figsize: Tuple for figure size

        Returns:
            - fig: The matplotlib Figure object
    """

    # Build list of keys for "variable" images:
    hr_key = f"{var}_hr"
    # Find LR keys from sample (assume keys ending with '_lr'): sort alphabetically for consistency
    lr_keys = sorted([k for k in sample.keys() if k.endswith('_lr')])

    # Scaled keys: HR and LR images
    scaled_keys = [hr_key] + lr_keys
    # Original keys: If available, ending with '_original'
    original_keys = []
    for key in scaled_keys:
        orig_key = key + "_original"
        if orig_key in sample:
            original_keys.append(orig_key)
    # Combing: extra keys (e.g. geo) will be appended later
    plot_keys = scaled_keys + original_keys
    if extra_keys is not None:
        plot_keys += extra_keys

    n_keys = len(plot_keys)

    # Create subplots in one row (one column per key)
    fig, axs = plt.subplots(1, n_keys, figsize=figsize)
    fig.suptitle(f"Sample from train dataset, {var} (HR: {hr_model}, LR: {lr_model})", fontsize=16)
    # Ensure axs is iteravle (if only one subplot, wrap in list)
    if n_keys == 1:
        axs = [axs]

    # Loop over each key and plot
    for idx, key in enumerate(plot_keys):
        ax = axs[idx]
        if key not in sample or sample[key] is None:
            ax.axis('off')
            continue

        # Get the image data; if a tensor, convert to np array
        img_data = sample[key]
        if torch.is_tensor(img_data):
            img_data = img_data.squeeze().cpu().numpy()

        # For HR images (keys ending with '_hr' or '_hr_original'), if show_ocean is False, apply masking using lsm_hr
        if not show_ocean and (key.endswith("_hr") or key.endswith("_hr_original")):
            if "lsm_hr" in sample and sample["lsm_hr"] is not None:
                mask = sample["lsm_hr"].squeeze().cpu().numpy()
                # ASsume mask values below 1 indicates ocean - set pixels to NaN
                img_data = np.where(mask < 1, np.nan, img_data)

        # Determine the colormap based on the key:
        if key.endswith('_hr') or key.endswith('_hr_original'):
            cmap = hr_cmap
        elif key.endswith('_lr') or key.endswith('_lr_original'):
            # Remove suffix to get the base condition name
            if key.endswith('_lr'):
                base = key[:-3]
            elif key.endswith('_lr_original'):
                base = key[:-12]
            # print(f"Base: {base}")
            if lr_cmap_dict is not None and base in lr_cmap_dict:
                cmap = lr_cmap_dict[base]
            else:
                cmap = default_lr_cmap
        else:
            # For extra keys, use the provided cmap_dict or default to 'viridis'
            if extra_cmap_dict is not None and key in extra_cmap_dict:
                cmap = extra_cmap_dict[key]
            else:
                cmap = 'viridis' # Default colormap for extra keys

        # Determine vmin and vmax: if force_matching_scale is True and dicts are provided, use them, otherwise compute from data
        if force_matching_scale and global_min is not None and global_max is not None:
            vmin = global_min.get(key, np.nanmin(img_data)) # get min from dict or compute from data
            vmax = global_max.get(key, np.nanmax(img_data)) # get max from dict or compute from data
        else:
            vmin = np.nanmin(img_data)
            vmax = np.nanmax(img_data)

        # Plot the image 
        im = ax.imshow(img_data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.invert_yaxis()  # Invert y-axis to match the original image orientation
        ax.set_xticks([])
        ax.set_yticks([])

        # Set column title
        if key.endswith('_hr'):
            title = f"HR {hr_model} ({var})\nscaled"
        elif key.endswith('_hr_original'):
            title = f"HR {hr_model} ({var})\noriginal [{hr_units}]"
        elif key.endswith('_lr'):
            title = f"LR {lr_model} ({base})\nscaled"
        elif key.endswith('_lr_original'):
            title = f"LR {lr_model} ({base})\noriginal [{lr_units[lr_keys.index(base)]}]"
        elif key in extra_keys:
            if key == "topo":
                title = f"Topography"
            elif key == "sdf":
                title = f"SDF"
            elif key == "lsm":
                title = f"Land/Sea Mask"
            else:
                title = f"{key}"
        else:
            title = f"{key}"
        ax.set_title(title, fontsize=10)



        # Create an axes divider to add a colorbar and (for variable images) a boxplot
        divider = make_axes_locatable(ax)
        if key.endswith('_hr') or key.endswith('_lr') or key.endswith('_hr_original') or key.endswith('_lr_original'):
            bax = divider.append_axes("right", size="10%", pad=0.1)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            # Boxplot settings
            flierprops = dict(marker='o', markerfacecolor='none', markersize=2,
                              linestyle='None', markeredgecolor='darkgreen', alpha=0.4)
            medianprops = dict(linestyle='-', linewidth=2, color='black')
            meanpointprops = dict(marker='x', markerfacecolor='firebrick', markersize=5, markeredgecolor='firebrick')
            # Exclude Nans.
            if torch.is_tensor(img_data):
                mask = ~torch.isnan(img_data)
                img_bp = img_data[mask].flatten().cpu().numpy()
            else:
                img_bp = img_data[~np.isnan(img_data)].flatten()
                if len(img_bp) > 0:
                    bax.boxplot(img_bp,
                                vert=True,
                                widths=2,
                                showmeans=True,
                                meanprops=meanpointprops,
                                flierprops=flierprops,
                                medianprops=medianprops,)
                bax.set_xticks([])
                bax.set_yticks([])
                bax.set_frame_on(False)
        else:
            # For extra keys, just add a colorbar
            cax = divider.append_axes("right", size="5%", pad=0.1)
            bax = None

        fig.colorbar(im, cax=cax, orientation='vertical')

    fig.tight_layout()

    return fig, ax

def plot_samples(samples, hr_model, hr_units, 
                        lr_model, lr_units,
                        var,
                        show_ocean=False,
                        force_matching_scale=True,
                        global_min=None, global_max=None,
                        extra_keys=None, 
                        hr_cmap='plasma', 
                        lr_cmap_dict=None,  # e.g., {"prcp": "inferno", "temp": "plasma"}
                        default_lr_cmap='viridis',
                        extra_cmap_dict=None,  # e.g., {"topo": "terrain", "lsm": "binary", "sdf": "coolwarm"}
                        n_samples_threshold=3,
                        figsize=(15, 8)):
    """
    Plot a batch of samples (provided as a list of sample dictionaries) in a grid where each row is a sample and
    each column corresponds to a particular key (e.g., HR, LR, originals, geo).
    
    If the number of samples exceeds n_samples_threshold, only the first n_samples_threshold will be plotted.
    
    Parameters:
      - sample_list: List of sample dictionaries.
      - hr_model: String (e.g., "DANRA") for HR model name (used in titles).
      - hr_units: Units for HR image (e.g., "mm" or "°C").
      - lr_model: String (e.g., "ERA5") for LR model name.
      - lr_units: List of units for LR conditions (order corresponding to sorted LR keys).
      - var: HR variable name (e.g., "prcp" or "temp").
      - show_ocean: If False, HR images are masked using the HR mask ("lsm_hr").
      - force_matching_scale: If True and if global_min/global_max dictionaries are provided, those values are used.
      - global_min, global_max: Dictionaries mapping keys (e.g., "prcp_hr", "prcp_lr") to scalar min and max values.
      - extra_keys: List of extra keys to plot (e.g., ["topo", "sdf"]).
      - hr_cmap: Colormap for HR images.
      - lr_cmap_dict: Dictionary mapping LR condition base names to colormaps.
      - default_lr_cmap: Default colormap for LR conditions if not found in lr_cmap_dict.
      - extra_cmap_dict: Dictionary mapping extra key names to colormaps.
      - n_samples_threshold: Maximum number of samples (rows) to plot.
      - figsize: Overall figure size.
      
    Returns:
      - fig: The matplotlib Figure object.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # If single batch dict is passed, unpack it to a list
    if isinstance(samples, dict):
        # Figure out batch size from first tensor we find:
        for v in samples.values():
            if torch.is_tensor(v):
                batch_size = v.shape[0]
                break
            if isinstance(v, list) and all(torch.is_tensor(x) for x in v):
                batch_size = len(v)
                break
            else:
                raise ValueError("No tensor found in the sample dictionary.")
        
        sample_list = []
        for i in range(batch_size):
            single = {}
            for k, v in samples.items():
                if torch.is_tensor(v):
                    # Slice tensor on batch dim
                    single[k] = v[i]
                elif isinstance(v, (list, tuple)) and len(v) == batch_size:
                    # Truly per-sample list
                    single[k] = v[i]
                else:
                    # Some constant list or metadata: leave as-is
                    single[k] = v
            sample_list.append(single)
    else:
        sample_list = samples

    # print(f"Plotting first {n_samples_threshold} samples out of {len(sample_list)} provided.")
    sample_list = sample_list[:n_samples_threshold]
    
    # Construct the keys:
    # HR key is "var_hr" (e.g., "prcp_hr")
    hr_key = f"{var}_hr"
    # Assume LR keys end with '_lr'
    lr_keys = sorted([key for key in sample_list[0].keys() if key.endswith('_lr')])
    scaled_keys = [hr_key] + lr_keys

    # Determine original keys if available.
    original_keys = []
    for key in scaled_keys:
        orig_key = key + "_original"
        if orig_key in sample_list[0]:
            original_keys.append(orig_key)
    
    # Build final list of keys. Append extra keys if provided.
    plot_keys = scaled_keys + original_keys
    if extra_keys is not None:
        plot_keys += extra_keys

    num_samples = len(sample_list)
    num_keys = len(plot_keys)

    # Create a grid with rows = number of samples and columns = number of keys
    fig, axs = plt.subplots(num_samples, num_keys, figsize=figsize)
    # Set figure title 
    fig.suptitle(f"Sample images for {var} (HR: {hr_model} and LR: {lr_model})", fontsize=16)
    if num_samples == 1:
        axs = np.expand_dims(axs, axis=0)
    if num_keys == 1:
        axs = np.expand_dims(axs, axis=1)

    for row, sample in enumerate(sample_list):
        for col, key in enumerate(plot_keys):
            ax = axs[row, col]
            if key not in sample or sample[key] is None:
                ax.axis('off')
                continue
            # Retrieve image data
            img_data = sample[key]
            if torch.is_tensor(img_data):
                img_data = img_data.squeeze().cpu().numpy()
            # For HR images mask out ocean using lsm_hr if needed.
            if not show_ocean and (key.endswith('_hr') or key.endswith('_hr_original')):
                if "lsm_hr" in sample and sample["lsm_hr"] is not None:
                    mask = sample["lsm_hr"].squeeze().cpu().numpy()
                    img_data = np.where(mask < 1, np.nan, img_data)
            # Determine color limits.
            if force_matching_scale and global_min is not None and global_max is not None:
                vmin = global_min.get(key, np.nanmin(img_data))
                vmax = global_max.get(key, np.nanmax(img_data))
            else:
                vmin, vmax = np.nanmin(img_data), np.nanmax(img_data)
            # Choose colormap:
            if key.endswith('_hr') or key.endswith('_hr_original'):
                cmap = hr_cmap
            elif key.endswith('_lr') or key.endswith('_lr_original'):
                if key.endswith('_lr'):
                    base = key[:-3]
                else:
                    base = key[:-12]
                if lr_cmap_dict is not None and base in lr_cmap_dict:
                    cmap = lr_cmap_dict[base]
                else:
                    cmap = default_lr_cmap
            else:
                if extra_cmap_dict is not None and key in extra_cmap_dict:
                    cmap = extra_cmap_dict[key]
                else:
                    cmap = 'viridis'
            im = ax.imshow(img_data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            divider = make_axes_locatable(ax)
            # For keys that correspond to variable fields, add a boxplot next to the colorbar.
            if key.endswith('_hr') or key.endswith('_lr') or key.endswith('_hr_original') or key.endswith('_lr_original'):
                bax = divider.append_axes("right", size="10%", pad=0.1)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                flierprops = dict(marker='o', markerfacecolor='none', markersize=2,
                                  linestyle='none', markeredgecolor='darkgreen', alpha=0.4)
                medianprops = dict(linestyle='-', linewidth=2, color='black')
                meanpointprops = dict(marker='x', markerfacecolor='firebrick', markersize=5, markeredgecolor='firebrick')
                img_flat = img_data[~np.isnan(img_data)].flatten()
                if len(img_flat) > 0:
                    bax.boxplot(img_flat,
                                vert=True,
                                widths=2,
                                patch_artist=True,
                                showmeans=True,
                                meanprops=meanpointprops,
                                medianprops=medianprops,
                                flierprops=flierprops)
                bax.set_xticks([])
                bax.set_yticks([])
                bax.set_frame_on(False)
            else:
                cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(im, cax=cax)
            # Set column title (only for top row)
            if row == 0:
                if key.endswith('_hr'):
                    title = f"HR {hr_model} ({var})\nscaled"
                elif key.endswith('_hr_original'):
                    title = f"HR {hr_model} ({var})\noriginal [{hr_units}]"
                elif key.endswith('_lr'):
                    title = f"LR {lr_model} ({base})\nscaled"
                elif key.endswith('_lr_original'):
                    title = f"LR {lr_model} ({base})\noriginal [{lr_units[lr_keys.index(base)]}]"
                elif key in extra_keys:
                    if key == "topo":
                        title = f"Topography"
                    elif key == "sdf":
                        title = f"SDF"
                    elif key == "lsm":
                        title = f"Land/Sea Mask"
                    else:
                        title = f"{key}"
                else:
                    title = f"{key}"
                ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig, axs


def plot_samples_and_generated(
        samples,
        generated,
        *,
        hr_model,
        hr_units,
        lr_model,
        lr_units,
        var,
        scaling=True,
        show_ocean=False,
        force_matching_scale=True,
        global_min=None,
        global_max=None,
        extra_keys=None,
        hr_cmap="plasma",
        lr_cmap_dict=None,
        default_lr_cmap="viridis",
        extra_cmap_dict=None,
        n_samples_threshold=3,
        figsize=(15, 8),
        transform_back_bf_plot=False,
        back_transforms=None,
):
    """
    Like ``plot_samples`` but adds an extra left-most column with “Generated”
    images (one per sample).

    Parameters
    ----------
    samples : dict | list[dict]
        The usual batch/list accepted by ``plot_samples``.
    generated : torch.Tensor | np.ndarray | list
        Shape (B,1,H,W) or list/tuple of length B with 2-D arrays.
    transform_back_bf_plot : bool, default False
        Apply inverse scaling before display.
    back_transforms : dict[str, Callable], optional
        Mapping *plot-key* → inverse-transform function.  Only used when
        *transform_back_bf_plot* is ``True``.
    """

    # ------------------------------------------------------------------ utils
    def to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def maybe_inverse(k, arr):
        if transform_back_bf_plot and back_transforms and k in back_transforms:
            return back_transforms[k](arr)
        return arr
    # print(f'Samples: {samples}')
    # print(f'Generated: {generated}')
    # -------------------------------------------------------- unpack samples
    if isinstance(samples, dict):              # turn single batch-dict → list
        for v in samples.values():
            if torch.is_tensor(v):
                B = v.shape[0]; break
            if isinstance(v, list) and v and torch.is_tensor(v[0]):
                B = len(v); break
        sample_list = []
        for i in range(B):
            d = {}
            for k, v in samples.items():
                if torch.is_tensor(v):
                    d[k] = v[i]
                elif isinstance(v, (list, tuple)) and len(v) == B:
                    d[k] = v[i]
                else:
                    d[k] = v
            sample_list.append(d)
    else:
        sample_list = list(samples)

    sample_list = sample_list[:n_samples_threshold]

    # ------------------------------------------------------- generated batch
    print(f"Generated shape: {generated.shape}")
    gen_np = to_numpy(generated)
    if gen_np.ndim == 4:               # (B, 1, H, W), multiple samples with 1 channel
        gen_np = gen_np[:, 0, :, :]
    elif gen_np.ndim == 3:             # (B, H, W), multiple samples
        pass
    elif gen_np.ndim == 2:             # (H, W), single samples
        gen_np = np.expand_dims(gen_np, axis=0)
    else:
        raise ValueError(f"Unexpected shape for generated samples: {gen_np.shape}")

    gen_np = gen_np[:len(sample_list)]
    print(f"Generated shape after slicing: {gen_np.shape}")

    # inject into dicts
    gen_key = "generated"
    for d, im in zip(sample_list, gen_np):
        d[gen_key] = im

    # --------------------------------------------------- assemble key order
    hr_key = f"{var}_hr"
    lr_keys = sorted(k for k in sample_list[0] if k.endswith("_lr"))
    original_keys = [k + "_original"
                     for k in (hr_key, *lr_keys)
                     if k + "_original" in sample_list[0]]

    plot_keys = [gen_key, hr_key, *lr_keys, *original_keys]
    if extra_keys:
        plot_keys.extend(extra_keys)

    # ----------------------------------------------------------- colourlims
    if force_matching_scale and global_min is not None:
        # share HR limits with generated if user hasn’t provided any
        global_min.setdefault(gen_key, global_min.get(hr_key))
        global_max.setdefault(gen_key, global_max.get(hr_key))

    # -------------------------------------------------------------- figure
    n_rows, n_cols = len(sample_list), len(plot_keys)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(f"Generated vs. data – {var}  "
                 f"(HR {hr_model} / LR {lr_model})", fontsize=16)

    # Ensure axs is always 2D
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1:
        axs = axs[np.newaxis, :]
    if n_cols == 1:
        axs = axs[:, np.newaxis]

    fig.suptitle(f"Generated vs. conditions – {var} (HR {hr_model} / LR {lr_model}) ")
    
    # --------------------------------------------------------- draw images
    for r, sample in enumerate(sample_list):
        for c, key in enumerate(plot_keys):
            ax = axs[r, c]
            if key not in sample or sample[key] is None:
                ax.axis("off")
                continue
            
            # Print key and shape for debugging
            print(f"Key: {key}")
            print(f"Shape: {sample[key].shape}")

            img = to_numpy(sample[key]).squeeze()
            img = maybe_inverse(key, img)

            # mask ocean for HR & generated columns
            if not show_ocean and key in {gen_key, hr_key, f"{hr_key}_original"}:
                if "lsm_hr" in sample and sample["lsm_hr"] is not None:
                    mask = to_numpy(sample["lsm_hr"]).squeeze()
                    img = np.where(mask < 1, np.nan, img)

            # choose colormap
            if key in {gen_key, hr_key, f"{hr_key}_original"}:
                cmap = hr_cmap
            elif key.endswith("_lr") or key.endswith("_lr_original"):
                base = key.replace("_lr", "").replace("_lr_original", "")
                cmap = (lr_cmap_dict or {}).get(base, default_lr_cmap)
            else:
                cmap = (extra_cmap_dict or {}).get(key, "viridis")

            if force_matching_scale and global_min is not None:
                vmin = global_min.get(key, np.nanmin(img))
                vmax = global_max.get(key, np.nanmax(img))
            else:
                vmin, vmax = np.nanmin(img), np.nanmax(img)

            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
                           interpolation="nearest")
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])

            # colour-bar
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

            # column headers
            if r == 0:
                if scaling:
                    if transform_back_bf_plot and back_transforms and key in back_transforms:
                        titles = {
                            gen_key: "Generated",
                            hr_key: f"HR {hr_model}, {var}\nback-transformed [{hr_units}]",
                            **{k: f"LR {lr_model} ({k[:-3]})\nback-transformed [{lr_units[lr_keys.index(k[:-3])] if k[:-3] in lr_keys else 'unknown'}]" for k in lr_keys},
                            **{k: f"LR {lr_model} ({k[:-12]})\nscaled" for k in original_keys},
                        }
                    else:
                        titles = {
                            gen_key: "Generated",
                            hr_key: f"HR {hr_model}, {var}\nscaled",
                            **{k: f"LR {lr_model} ({k[:-3]})\nscaled" for k in lr_keys},
                            **{k: f"LR {lr_model} ({k[:-12]})\noriginal [{lr_units[lr_keys.index(k[:-12])] if k[:-12] in lr_keys else 'unknown'}]" for k in original_keys},
                        }
                else:
                    titles = {
                        gen_key: "Generated",
                        hr_key: f"HR {hr_model}, {var}\nno scaling [{hr_units}]",
                        **{k: f"LR {lr_model} ({k[:-3]})\nno scaling [{lr_units[lr_keys.index(k[:-3])] if k[:-3] in lr_keys else 'unknown'}]" for k in lr_keys},
                        **{k: f"LR {lr_model}" for k in lr_keys},
                    }
                
                ax.set_title(titles.get(key, key), fontsize=9)

    fig.tight_layout()
    return fig, axs

# def load_config(yaml_file):
#     """
#         Loads a YAML configuration file and returns a dictionary
#     """
#     with open(yaml_file, 'r') as f:
#         config = yaml.safe_load(f)
#         return config
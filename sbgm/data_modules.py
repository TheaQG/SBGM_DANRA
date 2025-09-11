"""
    Script for generating a pytorch dataset for the DANRA data.
    The dataset can be used for training and testing the SBGM_SD model.
"""

# Import libraries and modules 
import zarr
import re
import random
import torch
import logging
# import multiprocessing

import numpy as np
import torch.nn.functional as F

from typing import Optional, List, Tuple
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from scipy.ndimage import distance_transform_edt as distance

from sbgm.special_transforms import Scale, get_transforms_from_stats
from sbgm.utils import correct_variable_units

# Set logging
logger = logging.getLogger(__name__)

def preprocess_lsm_topography(lsm_path, topo_path, target_size, scale=False, flip=False):
    '''
        Preprocess the lsm and topography data.
        Function loads the data, converts it to tensors, normalizes the topography data to [0, 1] interval,
        and upscales the data to match the target size.

        Input:
            - lsm_path: path to lsm data
            - topo_path: path to topography data
            - target_size: tuple containing the target size of the data
    '''
    # 1. Load the Data and flip upside down if flip=True
    if flip:
        lsm_data = np.flipud(np.load(lsm_path)['data']).copy() # Copy to avoid negative strides
        topo_data = np.flipud(np.load(topo_path)['data']).copy() # Copy to avoid negative strides
        
    else:
        lsm_data = np.load(lsm_path)['data']
        topo_data = np.load(topo_path)['data']

    # 2. Convert to Tensors
    lsm_tensor = torch.tensor(lsm_data).float().unsqueeze(0)  # Add channel dimension
    topo_tensor = torch.tensor(topo_data).float().unsqueeze(0)
    
    if scale: # SHOULD THIS ALSO BE A Z-SCALE TRANSFORM?
        # 3. Normalize Topography to [0, 1] interval
        topo_tensor = (topo_tensor - topo_tensor.min()) / (topo_tensor.max() - topo_tensor.min())
    
    # 4. Upscale the Fields to match target size
    resize_lsm = transforms.Resize(target_size, interpolation=InterpolationMode.NEAREST, antialias=False) # Nearest for masks to avoid interpolation artifacts (keep values 0 and 1 only)
    resize_topo = transforms.Resize(target_size, interpolation=InterpolationMode.BILINEAR, antialias=True) # Bilinear for continuous topo data
    lsm_tensor = resize_lsm(lsm_tensor)
    topo_tensor = resize_topo(topo_tensor)

    return lsm_tensor, topo_tensor

def preprocess_lsm_topography_from_data(lsm_data, topo_data, target_size, scale=True):
    '''
        Preprocess the lsm and topography data.
        Function loads the data, converts it to tensors, normalizes the topography data to[0, 1] interval (if scale=True)),
        and upscales the data to match the target size.

        Input:
            - lsm_data: lsm data
            - topo_data: topography data
            - target_size: tuple containing the target size of the data
            - scale: whether to scale the topography data to [0, 1] interval
    '''    
    # 1. Convert to Tensors
    lsm_tensor = torch.tensor(lsm_data.copy()).float().unsqueeze(0)  # Add channel dimension
    topo_tensor = torch.tensor(topo_data.copy()).float().unsqueeze(0)
    
    if scale:
        # 2. Normalize Topography to [0, 1] interval
        topo_tensor = (topo_tensor - topo_tensor.min()) / (topo_tensor.max() - topo_tensor.min())
    
    # 3. Upscale the Fields to match target size
    resize_lsm = transforms.Resize(target_size, interpolation=InterpolationMode.NEAREST, antialias=False) # Nearest for masks to avoid interpolation artifacts (keep values 0 and 1 only)
    resize_topo = transforms.Resize(target_size, interpolation=InterpolationMode.BILINEAR, antialias=True) # Bilinear for continuous topo data
    lsm_tensor = resize_lsm(lsm_tensor)
    topo_tensor = resize_topo(topo_tensor)

    return lsm_tensor, topo_tensor

def generate_sdf(mask):
    # Ensure mask is boolean
    binary_mask = mask > 0 

    # Distance transform for sea
    dist_transform_sea = distance(~binary_mask)

    # Set land to 1 and subtract sea distances
    sdf = 10*binary_mask.float() - dist_transform_sea

    return sdf

def normalize_sdf(sdf):
    # Find min and max in the SDF
    if isinstance(sdf, torch.Tensor):
        min_val = torch.min(sdf)
        max_val = torch.max(sdf)
    elif isinstance(sdf, np.ndarray):
        min_val = np.min(sdf)
        max_val = np.max(sdf)
    else:
        raise ValueError('SDF must be either torch.Tensor or np.ndarray')

    # Normalize the SDF
    sdf_normalized = (sdf - min_val) / (max_val - min_val)
    return sdf_normalized

class DateFromFile:
    '''
    General class for extracting date from filename.
    Can take .npz, .nc and .zarr files.
    Not dependent on the file extension.
    '''
    def __init__(self, filename):
        # Remove file extension
        self.filename = filename.split('.')[0]
        # Get the year, month and day from filename ending (YYYYMMDD)
        self.year = int(self.filename[-8:-4])
        self.month = int(self.filename[-4:-2])
        self.day = int(self.filename[-2:])

    def determine_season(self):
        # Determine season based on month
        if self.month in [3, 4, 5]:
            return 1
        elif self.month in [6, 7, 8]:
            return 2
        elif self.month in [9, 10, 11]:
            return 3
        else:
            return 4

    def determine_month(self):
        # Returns the month as an integer in the interval [1, 12]
        return self.month

    @staticmethod
    def is_leap_year(year):
        """Check if a year is a leap year"""
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            return True
        return False

    def determine_day(self):
        # Days in month for common years and leap years
        days_in_month_common = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        days_in_month_leap = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        # Determine if the year is a leap year
        if self.is_leap_year(self.year):
            days_in_month = days_in_month_leap
        else:
            days_in_month = days_in_month_common

        # Compute the day of the year
        day_of_year = sum(days_in_month[:self.month]) + self.day # Now 1st January is 1 instead of 0
        return day_of_year
    
def FileDate(filename):
    """
    Extract the last 8 digits from the filename as the date string.
    E.g. for 't2m_ave_19910122' or 'temp_589x789_19910122', returns '19910122'
    """

    m = re.search(r'(\d{8})$', filename)
    if m:
        return m.group(1)
    else:
        raise ValueError(f"Could not extract date from filename: {filename}")


def find_rand_points(rect, crop_size):
    '''
    Randomly selects a crop region within a given rectangle
    Input:
        - rect (list or tuple): [x1, x2, y1, y2] rectangle to crop from
        - crop_size (tuple): (crop_width, crop_height) size of the desired crop
    Output:
        - point (list): [x1_new, x2_new, y1_new, y2_new] random crop region

    Raises: 
        - ValueError if crop_size is larger than the rectangle
    '''
    x1 = rect[0]
    x2 = rect[1]
    y1 = rect[2]
    y2 = rect[3]

    crop_width = crop_size[0]
    crop_height = crop_size[1]

    full_width = x2 - x1
    full_height = y2 - y1

    if crop_width > full_width or crop_height > full_height:
        raise ValueError('Crop size is larger than the rectangle dimensions.')

    # Calculate available offsets
    max_x_offset = full_width - crop_width
    max_y_offset = full_height - crop_height

    offset_x = random.randint(0, max_x_offset)
    offset_y = random.randint(0, max_y_offset)

    x1_new = x1 + offset_x
    x2_new = x1_new + crop_width
    y1_new = y1 + offset_y
    y2_new = y1_new + crop_height

    point = [x1_new, x2_new, y1_new, y2_new]
    return point


def random_crop(data, target_size):
    """
        Randomly crops a 2D 'data' to shape (target_size[0], target_size[1]).
        Assumes data is a 2D numpy array
        Input:
            - data: 2D numpy array
            - target_size: tuple containing the target size of the data
        Output:
            - data: 2D numpy array with shape (target_size[0], target_size[1])
        Raises:
            - ValueError if target size is larger than the data dimensions    
    """
    H, W = data.shape

    if target_size[0] > H or target_size[1] > W:
        raise ValueError('Target size is larger than the data dimensions.')
    
    if H == target_size[0] and W == target_size[1]:
        return data

    y = random.randint(0, H - target_size[0])
    x = random.randint(0, W - target_size[1])
    return data[y:y + target_size[0], x:x + target_size[1]]


def make_tensor_resize(target_size):
    """
        Create a transform that resizes a tensor to the target size.
        Necessary because the Resize transform in torchvision expects a PIL image.
        Resize otherwise silently no-ops on a 2D tensor.
    """
    return transforms.Lambda(
        lambda t:
        F.interpolate(
            t.unsqueeze(0), # [1, C, H, W]
            size=target_size, # (new_height, new_width)
            mode='bilinear', # Or 'nearest' or 'bicubic' etc
            align_corners=False,
        ).squeeze(0)        # Back to [C, H, W]
    )

class SafeToTensor:
    def __call__(self, x):

        if isinstance(x, np.ndarray):
            return transforms.ToTensor()(x)
        elif isinstance(x, torch.Tensor):
            return x
        else:
            raise TypeError(f"Unexpected input type: {type(x)}. Expected np.ndarray or torch.Tensor.")

class ResizeTensor:
    """
        Create a transform that resizes a tensor to the target size.
        Necessary because the Resize transform in torchvision expects a PIL image.
        Resize otherwise silently no-ops on a 2D tensor.
        
        Resize a torch.Tensor of shape [C,H,W] to [C,new_H,new_W]
        by torch.nn.functional.interpolate.  No PIL, no 8-bit quantization.
    """
    def __init__(self, size, mode='bilinear', align_corners=False):
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2: # [H, W]
            x = x.unsqueeze(0).unsqueeze(0)  # Add channel and batch dimensions → [1, 1, H, W]
        elif x.ndim == 3: # [C, H, W]
            x = x.unsqueeze(0) # [1, C, H, W]
        elif x.ndim != 4:
            raise ValueError(f"ResizeTensor: Unsupported shape {x.shape}")
        
        # Align_corners is only used for mode='bilinear' or 'bicubic'
        if self.mode in ["bilinear", "bicubic"]:
            # interpolate → [1, C, new_H, new_W]
            x = F.interpolate(x, size=self.size, mode=self.mode, align_corners=self.align_corners)
        elif self.mode == "nearest":
            x = F.interpolate(x, size=self.size, mode=self.mode)
        else:
            raise ValueError(f"ResizeTensor: Unsupported mode {self.mode}")

        # remove batch dim → [C, new_H, new_W]
        return x.squeeze(0) # → [C, H, W] or [1, H, W] depending on input



def list_all_keys(zgroup):
    all_keys = []
    for key in zgroup.keys():
        all_keys.append(key)
        member = zgroup[key]
        if isinstance(member, zarr.Group):
            sub_keys = list_all_keys(member)
            all_keys.extend([f"{key}/{sub}" for sub in sub_keys])
    return all_keys


# Helper for robust array extraction
def _first_hw_slice(arr: np.ndarray) -> np.ndarray:
    """
        Return a 2D (H, W) view by taking the first slice along any leading dimensions.
        Works for shapes like (H, W), (1, H, W), (N, H, W), (N, 1, H, W), etc.
    """
    arr = np.asarray(arr) # Ensure it's a numpy array
    if arr.ndim < 2:
        raise ValueError(f"Array must have at least 2 dimensions, got shape {arr.shape}")
    H, W = arr.shape[-2], arr.shape[-1]
    return arr.reshape(-1, H, W)[0] # First slice of shape (H, W)

# Helper for extracting field from zarr group entry with common dataset keys per variable
def _extract_2d_from_zarr_entry(zgroup: zarr.Group, file_key: str, var_name: str) -> np.ndarray:
    """
        Load a 2D field from a zarr group entry, trying common dataset keys per variable.
        Returns a numpy array with shape (H, W).
    """
    entry = zgroup[file_key]

    KEY_CANDIDATES = {
        'temp': ['t', 'data', 'arr_0'],
        't2m': ['t', 'data', 'arr_0'],
        'prcp': ['tp', 'data', 'arr_0'],
        'tp': ['tp', 'data', 'arr_0'],
        '_default': ['data', 'arr_0'],
    }

    candidates = KEY_CANDIDATES.get(var_name, []) + KEY_CANDIDATES['_default']
    for k in candidates:
        if k in entry:
            arr = entry[k][()] # Load the array
            return _first_hw_slice(arr) # Return as (H, W)
        
    # Fallback: try any array-like members under the entry
    for k in entry.keys():
        try:
            arr = entry[k][()]
            return _first_hw_slice(arr)
        except Exception:
            continue
    raise KeyError(f"Could not find a suitable data array in zarr entry '{file_key}' for variable '{var_name}'. Tried keys: {candidates} and all members.")

# all_keys = list_all_keys(self.lr_cond_zarr_dict[cond])
# logger.debug(all_keys)


class DANRA_Dataset_cutouts_ERA5_Zarr(Dataset):
    '''
        Class for setting the DANRA dataset with option for random cutouts from specified domains.
        Along with DANRA data, the land-sea mask and topography data is also loaded at same cutout.
        Possibility to sample more than n_samples if cutouts are used.
        Option to shuffle data or load sequentially.
        Option to scale data to new interval.
        Option to use conditional (classifier) sampling (season, month or day).
    '''
    def __init__(self, 
                # Must-have parameters
                hr_variable_dir_zarr:str,           # Path to high resolution data
                hr_data_size:tuple,                 # Size of data (2D image, tuple)
                # HR target variable and its scaling parameters
                hr_variable:str = 'temp',           # Variable to load (temp or prcp)
                hr_model:str = 'DANRA',             # Model name (e.g. 'DANRA', 'ERA5')
                hr_scaling_method:str = 'zscore',   # Scaling method for high resolution data
                # hr_scaling_params:dict = {'glob_mean':8.69251, 'glob_std':6.192434}, # Scaling parameters for high resolution data (if prcp, 'log_minus1_1' or 'log_01' include 'glob_min_log' and 'glob_max_log' and optional buffer_frac)
                # LR conditions and their scaling parameters (not including geo variables. they are handled separately)
                lr_conditions:list = ['temp'],      # Variables to load as low resolution conditions
                lr_model:str = 'ERA5',              # Model name (e.g. 'DANRA', 'ERA5')
                lr_scaling_methods:list = ['zscore'], # Scaling methods for low resolution conditions
                # lr_scaling_params:list = [{'glob_mean':8.69251, 'glob_std':6.192434}], # Scaling parameters for low resolution conditions
                lr_cond_dirs_zarr:Optional[dict] = None,      # Path to directories containing conditional data (in format dict({'condition1':dir1, 'condition2':dir2}))
                # NEW: LR conditioning area size (if cropping is desired)
                lr_data_size:Optional[tuple] = None,         # Size of low resolution data (2D image, tuple), e.g. (589,789) for full LR domain
                # Optionally a separate cutout domain for LR conditions
                lr_cutout_domains:Optional[list] = None,     # Domains to use for cutouts for LR conditions
                resize_factor: int = 1,             # Resize factor for input conditions (1 for full HR size, 2 for half HR size, etc. Mainly used for testing on smaller data)
                # Geo variables (stationary) and their full domain arrays
                geo_variables:Optional[list] = ['lsm', 'topo'], # Geo variables to load
                lsm_full_domain = None,             # Land-sea mask of full domain
                topo_full_domain = None,            # Topography of full domain
                # Configuration information
                cfg: dict | None = None,
                split: str = "train",
                # Other dataset parameters
                n_samples:int = 365,                # Number of samples to load
                cache_size:int = 365,               # Number of samples to cache
                shuffle:bool = False,               # Whether to shuffle data (or load sequentially)
                cutouts:bool = False,               # Whether to use cutouts 
                cutout_domains:Optional[list] = None,         # Domains to use for cutouts
                n_samples_w_cutouts:Optional[int] = None,     # Number of samples to load with cutouts (can be greater than n_samples)
                sdf_weighted_loss:bool = False,     # Whether to use weighted loss for SDF
                scale:bool = True,                  # Whether to scale data to new interval
                save_original:bool = False,         # Whether to save original data
                conditional_seasons:bool = False,   # Whether to use seasonal conditional sampling
                n_classes:Optional[int] = None                # Number of classes for conditional sampling
                ):                          
        '''
        Initializes the dataset.
        '''
        
        # Basic dataset parameters
        self.hr_variable_dir_zarr = hr_variable_dir_zarr
        self.n_samples = n_samples
        self.hr_data_size = hr_data_size
        self.cache_size = cache_size

        # LR conditions and scaling parameters
        # (Remove any geo variable from conditions list, if accidentally included)
        self.geo_variables = geo_variables
        # Check that there are the same number of scaling methods and parameters as conditions
        if len(lr_conditions) != len(lr_scaling_methods): # or len(lr_conditions) != len(lr_scaling_params):
            raise ValueError('Number of conditions and scaling methods must be the same')

        # Go through the conditions, and if condition is in geo_variables, remoce from list, and remove scaling methods and params associated with it
        # But only if any geo_variables exist
        if self.geo_variables is not None:
            for geo_var in self.geo_variables:
                if geo_var in lr_conditions:
                    idx = lr_conditions.index(geo_var)
                    lr_conditions.pop(idx)
                    lr_scaling_methods.pop(idx)
                    # lr_scaling_params.pop(idx)
        self.lr_conditions = lr_conditions
        self.lr_model = lr_model
        self.lr_scaling_methods = lr_scaling_methods
        # self.lr_scaling_params = lr_scaling_params
        # If any conditions exist, set with_conditions to True
        self.with_conditions = len(self.lr_conditions) > 0

        # Save new LR parameters
        self.lr_data_size = lr_data_size
        self.lr_cutout_domains = lr_cutout_domains
        
        # Check whether lr_cutout_domains are parsed as a list or tuple - even if 'None' - and set correctly to None if not
        if isinstance(self.lr_cutout_domains, list) or isinstance(self.lr_cutout_domains, tuple):
            if len(self.lr_cutout_domains) == 0 or (len(self.lr_cutout_domains) == 1 and str(self.lr_cutout_domains[0]).lower() == 'none'):
                self.lr_cutout_domains = None
            else:
                self.lr_cutout_domains = self.lr_cutout_domains
        
        # Specify target LR size (if different from HR size)
        self.target_lr_size = self.lr_data_size if self.lr_data_size is not None else self.hr_data_size
        
        # Resize factor for input conditions (for running with smaller data)
        self.resize_factor = resize_factor
        if self.resize_factor > 1:
            self.hr_size_reduced = (int(self.hr_data_size[0]/self.resize_factor), int(self.hr_data_size[1]/self.resize_factor))
            self.lr_size_reduced = (int(self.target_lr_size[0]/self.resize_factor), int(self.target_lr_size[1]/self.resize_factor))
        elif self.resize_factor == 1:
            self.hr_size_reduced = self.hr_data_size
            self.lr_size_reduced = self.target_lr_size
        else:
            raise ValueError('Resize factor must be greater than 0')


        # Save LR condition directories
        # lr_cond_dirs_zarr is a dict mapping each condition to its own zarr directory path
        self.lr_cond_dirs_zarr = lr_cond_dirs_zarr
        # Open each LR condition's zarr group and list its files
        self.lr_cond_zarr_dict = {} 
        self.lr_cond_files_dict = {}
        if self.lr_cond_dirs_zarr is not None:
            for cond in self.lr_cond_dirs_zarr:
                logger.info(f'Loading zarr group for condition {cond}')
                # logger.info(f'Path to zarr group: {self.lr_cond_dirs_zarr[cond]}')
                group = zarr.open_group(self.lr_cond_dirs_zarr[cond], mode='r')
                self.lr_cond_zarr_dict[cond] = group
                self.lr_cond_files_dict[cond] = list(group.keys())
        else:
            raise ValueError('LR condition directories (lr_cond_dirs_zarr) must be provided as a dictionary.')

        # HR target variable parameters
        self.hr_variable = hr_variable
        self.hr_model = hr_model
        self.hr_scaling_method = hr_scaling_method
        # self.hr_scaling_params = hr_scaling_params
        
        # Save geo variables full-domain arrays
        self.lsm_full_domain = lsm_full_domain
        self.topo_full_domain = topo_full_domain

        # Save classifier-free guidance parameters
        self.cfg = cfg
        self.split = split 

        # Save other parameters
        self.shuffle = shuffle
        self.cutouts = cutouts
        self.cutout_domains = cutout_domains
        self.sdf_weighted_loss = sdf_weighted_loss
        self.scale = scale
        self.save_original = save_original
        self.conditional_seasons = conditional_seasons
        self.n_classes = n_classes
        self.n_samples_w_cutouts = self.n_samples if n_samples_w_cutouts is None else n_samples_w_cutouts
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
        #                               #
        # PRINT INFORMATION ABOUT SCALING
        #                               #
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! #


        # Build file maps based on the date in the file name      
        # Open main (HR) zarr group, and get HR file keys (pure filenames)
        self.zarr_group_img = zarr.open_group(hr_variable_dir_zarr, mode='r')
        hr_files_all = list(self.zarr_group_img.keys())
        self.hr_file_map = {}
        for file in hr_files_all:
            try:
                date = FileDate(file)
                self.hr_file_map[date] = file
            except Exception as e:
                logger.warning(f"Could not extract date from file {file}. Skipping file. Error: {e}")
                

        # For each LR condition, build a file map: date -> file key
        self.lr_file_map = {}
        for cond in self.lr_conditions:
            self.lr_file_map[cond] = {}
            for file in self.lr_cond_files_dict[cond]:
                try:
                    date = FileDate(file)
                    self.lr_file_map[cond][date] = file
                except Exception as e:
                    logger.warning(f"Could not extract date from file {file} for condition {cond}. Skipping file. Error: {e}")

        # Compute common dates across HR and all LR conditions
        common_dates = set(self.hr_file_map.keys())
        for cond in self.lr_conditions:
            common_dates = common_dates.intersection(set(self.lr_file_map[cond].keys()))
        self.common_dates = sorted(list(common_dates))
        if len(self.common_dates) < self.n_samples:
            self.n_samples = len(self.common_dates)
            logger.warning(f"Not enough common dates ({len(self.common_dates)}) to sample {self.n_samples} samples. Reducing n_samples to {self.n_samples}.")

        if self.shuffle:
            self.common_dates = random.sample(self.common_dates, self.n_samples)
        
        # Set cache for data loading - if cache_size is 0, no caching is used
        # If num_workers > 0 each worker has its own Dataset instance
        # self.cache = multiprocessing.Manager().dict()
        self.cache = {}  # Use a simple dict for caching, if cache_size > 0

        if self.scale:
            # 1. Set condition transforms
            self.lr_transforms_dict = {}
            domain_str_hr = f"{cfg['highres']['full_domain_dims'][0]}x{cfg['highres']['full_domain_dims'][1]}" if cfg is not None else f"{self.hr_data_size[0]}x{self.hr_data_size[1]}"
            domain_str_lr = f"{cfg['lowres']['full_domain_dims'][0]}x{cfg['lowres']['full_domain_dims'][1]}" if cfg is not None else f"{self.target_lr_size[0]}x{self.target_lr_size[1]}"
            crop_region_hr = cfg['highres']['cutout_domains'] if (cfg is not None and self.cutouts and self.cutout_domains is not None) else "full"
            crop_region_hr_str = '_'.join(map(str, crop_region_hr)) # if (cfg is not None and self.cutouts and self.cutout_domains is not None) else "full"
            crop_region_lr = cfg['lowres']['cutout_domains'] if (cfg is not None and self.cutouts and self.lr_cutout_domains is not None) else "full"
            crop_region_lr_str = '_'.join(map(str, crop_region_lr)) # if (cfg is not None and self.cutouts and self.lr_cutout_domains is not None) else "full"
            split = 'all' # Need to use 'all' for global stats. If not computed yet, used needs to run statistics script first
            stats_load_dir = cfg['paths']['stats_load_dir'] if cfg is not None else './stats'

            for cond_var, trans_type in zip(self.lr_conditions, self.lr_scaling_methods):
                logger.info(f"LR condition: {cond_var}, scaling method: {trans_type}")
                transform_list = [
                    SafeToTensor(),
                    ResizeTensor(self.lr_size_reduced)
                ]

                transform_list.append(get_transforms_from_stats(
                    variable=cond_var,
                    model=self.lr_model,
                    domain_str=domain_str_lr,
                    crop_region_str=crop_region_lr_str,
                    split=split,
                    transform_type=trans_type,
                    buffer_frac=cfg['lowres'].get('buffer_frac', 0.5) if cfg is not None else 0.5,
                    stats_file_path=stats_load_dir,
                ))
                self.lr_transforms_dict[cond_var] = transforms.Compose(transform_list)

            ############### OLD CODE - BEFORE USING get_transforms_from_stats ###############
            # for cond, method, params in zip(self.lr_conditions, self.lr_scaling_methods, self.lr_scaling_params):
            #     # Base transform: to tensor and resize
            #     transform_list = [
            #         SafeToTensor(),
            #         ResizeTensor(self.lr_size_reduced)
            #     ]
            #     # Use per-variable buffer_frac
            #     buff = params.get('buffer_frac', 0.5)
            #     if method == 'zscore':
            #         # ADD BUFFER FRACTION TO ZSCORE TRANSFORM
            #         transform_list.append(ZScoreTransform(params['glob_mean'], params['glob_std']))
            #     elif method in ['log', 'log_01', 'log_minus1_1', 'log_zscore']:
            #         transform_list.append(PrcpLogTransform(eps=1e-10,
            #                                                scale_type=method,
            #                                                glob_mean_log=params['glob_mean_log'],
            #                                                glob_std_log=params['glob_std_log'],
            #                                                glob_min_log=params['glob_min_log'],
            #                                                glob_max_log=params['glob_max_log'],
            #                                                buffer_frac=buff))
            #     elif method == '01':
            #         transform_list.append(Scale(0, 1, params['glob_min'], params['glob_max']))
            #     self.lr_transforms_dict[cond] = transforms.Compose(transform_list)


            # 2. Set HR target transform
            hr_transform_list = [
                SafeToTensor(),
                ResizeTensor(self.hr_size_reduced)
            ]
            hr_buff = cfg['highres'].get('buffer_frac', 0.5) if cfg is not None else 0.5
            hr_transform_list.append(get_transforms_from_stats(
                variable=self.hr_variable,
                model=self.hr_model,
                domain_str=domain_str_hr,
                crop_region_str=crop_region_hr_str,
                split='all', # Need to use 'all' for global stats. If not computed yet, used needs to run statistics script first
                transform_type=self.hr_scaling_method,
                buffer_frac=hr_buff,
                stats_file_path=stats_load_dir,
            ))
            self.hr_transform = transforms.Compose(hr_transform_list)
            ############### OLD CODE - BEFORE USING get_transforms_from_stats ###############
            # if self.hr_scaling_method == 'zscore':
            #     hr_transform_list.append(ZScoreTransform(self.hr_scaling_params['glob_mean'], self.hr_scaling_params['glob_std']))
            # elif self.hr_scaling_method in ['log', 'log_01', 'log_minus1_1', 'log_zscore']:
            #     hr_transform_list.append(PrcpLogTransform(eps=1e-10,
            #                                               scale_type=self.hr_scaling_method,
            #                                               glob_mean_log=self.hr_scaling_params['glob_mean_log'],
            #                                               glob_std_log=self.hr_scaling_params['glob_std_log'],
            #                                               glob_min_log=self.hr_scaling_params['glob_min_log'],
            #                                               glob_max_log=self.hr_scaling_params['glob_max_log'],
            #                                               buffer_frac=hr_buff))
            # elif self.hr_scaling_method == '01':
            #     hr_transform_list.append(Scale(0, 1, self.hr_scaling_params['glob_min'], self.hr_scaling_params['glob_max']))
            # self.hr_transform = transforms.Compose(hr_transform_list)
        
            # 3. Set geo variable transforms (if any)
            if self.geo_variables is not None:
                if self.topo_full_domain is None:
                    raise ValueError("topo_full_domain must be provided if 'topo' is in geo_variables")
                topo_scale_min = cfg['stationary_conditions']['geographic_conditions']['norm_min'] if (cfg is not None and 'stationary_conditions' in cfg and 'geographic_conditions' in cfg['stationary_conditions']) else -1.0
                topo_scale_max = cfg['stationary_conditions']['geographic_conditions']['norm_max'] if (cfg is not None and 'stationary_conditions' in cfg and 'geographic_conditions' in cfg['stationary_conditions']) else 1.0
                logger.info(f"Topography will be scaled to [{topo_scale_min}, {topo_scale_max}] interval.")
                # Update topo scaling transform to use cfg values if provided
                self.geo_transform_topo = transforms.Compose([
                    transforms.Lambda(lambda x: np.ascontiguousarray(x)), # To make sure np.flipud is not messing up the tensor
                    SafeToTensor(),
                    ResizeTensor(self.lr_size_reduced, mode='bilinear', align_corners=False),
                    Scale(topo_scale_min, topo_scale_max, self.topo_full_domain.min(), self.topo_full_domain.max())
                ])
                self.geo_transform_lsm = transforms.Compose([
                    transforms.Lambda(lambda x: np.ascontiguousarray(x)), # To make sure np.flipud is not messing up the tensor
                    SafeToTensor(),
                    ResizeTensor(self.lr_size_reduced, mode='nearest'), # Nearest for categorical data
                ])
        else:
            # 1. Set condition transforms
            self.lr_transforms_dict = {cond: transforms.Compose([
                SafeToTensor(),
                ResizeTensor(self.lr_size_reduced)
            ]) for cond in self.lr_conditions}

            # 2. Set HR target transform
            self.hr_transform = transforms.Compose([
                SafeToTensor(),
                ResizeTensor(self.hr_size_reduced)
            ])

            # 3. Set geo variable transforms (if any)
            if self.geo_variables is not None:
                self.geo_transform_topo = transforms.Compose([
                    transforms.Lambda(lambda x: np.ascontiguousarray(x)), # To make sure np.flipud is not messing up the tensor
                    SafeToTensor(),
                    ResizeTensor(self.lr_size_reduced)
                ])
                self.geo_transform_lsm = self.geo_transform_topo




    def __len__(self):
        '''
            Return the length of the dataset.
        '''
        return len(self.common_dates)

    def _addToCache(self, idx:int, data):
        '''
            Add item to cache. 
            If cache is full, remove random item from cache.
            Input:
                - idx: index of item to add to cache
                - data: data to add to cache
        '''
        # If cache_size is 0, no caching is used
        if self.cache_size > 0:
            # If cache is full, remove random item from cache
            if len(self.cache) >= self.cache_size:
                # Get keys from cache
                keys = list(self.cache.keys())
                # Select random key to remove
                key_to_remove = random.choice(keys)
                # Remove key from cache
                self.cache.pop(key_to_remove, None) # Safe removal, in case key is not found
            # Add data to cache
            self.cache[idx] = data
    
    def __getitem__(self, idx:int):
        '''
            For each sample:
            - Loads LR conditions from the main zarr group (and, if applicable, from additional condition directories)
            - Loads HR target variable from the main zarr group
            - Loads the stationary geo variables (lsm and topo) from provided full-domain arrays
            - Applies cutouts and the appropriate transforms
        '''

        if self.cache_size > 0 and (self.split != 'train' or not self.cutouts):
            cached = self.cache.get(idx, None)
            if cached is not None:
                return cached

        # Get the common date corresponding to the index
        date = self.common_dates[idx]
        sample_dict = {}

        # Determine crop region, if cutouts are used
        if self.cutouts:
            # hr_point is computed using HR cutout domain and HR data size
            hr_point = find_rand_points(self.cutout_domains, self.hr_data_size)
            if self.lr_data_size is not None:
                # If a separate LR cutout domain is provided, use it
                if self.lr_cutout_domains is not None:
                    lr_point = find_rand_points(self.lr_cutout_domains, self.lr_data_size)
                else:
                    # Otherwise default to the same cutout point as HR
                    lr_point = hr_point
            else:
                # If no LR data size is provided, use the same cutout point as HR
                lr_point = hr_point
        else:
            # If cutouts are not used, set points to None and use full domain
            hr_point = None
            lr_point = None

        # logger.debug(f'HR point: {hr_point}')
        # logger.debug(f'LR point: {lr_point}')
        # Look up HR file using the common date
        hr_file_name = self.hr_file_map[date]

        # Look up LR files for each condition using the common date
        for cond in self.lr_conditions:
            lr_file_name = self.lr_file_map[cond][date]
            # Load LR condition data from its own zarr group
            try:
                ################### OLD WAY - BEFORE CREATING _extract_2d_from_zarr_entry() AND correct_variable_units() ###################
                # # logger.info(f'Loading LR {cond} data for {lr_file_name}')
                # # logger.debug(self.lr_cond_zarr_dict[cond].tree())
                # if cond == "temp":
                #     try:
                #         data = self.lr_cond_zarr_dict[cond][lr_file_name]['t']
                #         data = data[()][0,0,:,:] - 273.15
                #         # logger.debug("Key 't' found")
                #     except:
                #         data = self.lr_cond_zarr_dict[cond][lr_file_name]['arr_0']
                #         data = data[()][:,:] - 273.15
                #         # logger.debug("Key 'data' found")
                # elif cond == "prcp":
                #     try:
                #         data = self.lr_cond_zarr_dict[cond][lr_file_name]['tp']
                #         data = data[()][0,0,:,:] * 1000
                #         data[data <= 0] = 1e-10
                #         # logger.debug("Key 'tp' found")
                #     except:
                #         data = self.lr_cond_zarr_dict[cond][lr_file_name]['arr_0']
                #         data = data[()][:,:] * 1000
                #         data[data <= 0] = 1e-10
                #         # logger.debug("Key 'arr_0' found")
                # else:
                #     # Add custom logic for other LR conditions when needed
                #     data = self.lr_cond_zarr_dict[cond][lr_file_name]['data'][()]
                data = _extract_2d_from_zarr_entry(self.lr_cond_zarr_dict[cond], lr_file_name, cond)
                # Apply unit corrections consistently
                data = correct_variable_units(cond, self.lr_model, data)
            except Exception as e:
                logger.error(f'Error loading {cond} data for {lr_file_name}. Error: {e}')
                data = None
            
            # Crop LR data using lr_point if cutouts are enabled and lr_point is not None
            if self.cutouts and data is not None and lr_point is not None:
                # lr_point is in format [x1, x2, y1, y2] - note: for slicing, use [y1:y2, x1:x2]
                data = data[lr_point[0]:lr_point[1], lr_point[2]:lr_point[3]]
            # logger.debug(f"Data shape for {cond}: {data.shape if data is not None else None}")
                
            # If save_original is True, save original conditional data
            if self.save_original:
                sample_dict[f"{cond}_lr_original"] = data.copy() if data is not None else None

            # Apply specified transform (specific to various conditions)
            if data is not None and self.lr_transforms_dict.get(cond, None) is not None:
                data = self.lr_transforms_dict[cond](data)
            sample_dict[cond + "_lr"] = data
        

        # Load HR target variable data
        try:
            ################### OLD WAY - BEFORE CREATING _extract_2d_from_zarr_entry() AND correct_variable_units() ###################
            # # logger.info(f'Loading HR {self.hr_variable} data for {hr_file_name}')
            # # logger.debug(self.zarr_group_img[hr_file_name].tree())
            # if self.hr_variable == 'temp':
            #     try:
            #         hr = torch.tensor(self.zarr_group_img[hr_file_name]['t'][()][0,0,:,:], dtype=torch.float32) - 273.15
            #     except:
            #         hr = torch.tensor(self.zarr_group_img[hr_file_name]['data'][()][:,:], dtype=torch.float32) - 273.15
            # elif self.hr_variable == 'prcp':
            #     try:
            #         hr = torch.tensor(self.zarr_group_img[hr_file_name]['tp'][()][0,0,:,:], dtype=torch.float32)
            #     except:
            #         hr = torch.tensor(self.zarr_group_img[hr_file_name]['data'][()][:,:], dtype=torch.float32)
            #     # Set all non-positive values to a small positive value (multiplied by a random number for robustness)
            #     hr[hr <= 0] = 1e-10 * np.random.rand()
            # else:
            #     # Add custom logic for other HR variables when needed
            #     hr = torch.tensor(self.zarr_group_img[hr_file_name]['data'][()], dtype=torch.float32)
            hr_np = _extract_2d_from_zarr_entry(self.zarr_group_img, hr_file_name, self.hr_variable)
            # Apply unit corrections consistently
            hr_np = correct_variable_units(self.hr_variable, self.hr_model, hr_np)
            hr = torch.tensor(hr_np, dtype=torch.float32)
        except Exception as e:
            logger.error(f'Error loading HR {self.hr_variable} data for {hr_file_name}. Error: {e}')
            hr = None

        if self.cutouts and (hr is not None) and (hr_point is not None):
            hr = hr[hr_point[0]:hr_point[1], hr_point[2]:hr_point[3]]
        if self.save_original and (hr is not None):
            sample_dict[f"{self.hr_variable}_hr_original"] = hr.clone()
        if hr is not None:
            hr = self.hr_transform(hr)
        sample_dict[self.hr_variable + "_hr"] = hr

        # Process a separate HR mask for geo variables (if 'lsm' is needed for HR SDF and masking HR images)
        if self.geo_variables is not None and 'lsm' in self.geo_variables and self.lsm_full_domain is not None:
            lsm_hr = self.lsm_full_domain
            if self.cutouts and lsm_hr is not None and hr_point is not None:
                lsm_hr = lsm_hr[hr_point[0]:hr_point[1], hr_point[2]:hr_point[3]]
            # Ensure the mask is contiguous and transform
            lsm_hr = np.ascontiguousarray(lsm_hr)
            # Separate geo transform, with resize to HR size
            geo_transform_lsm_hr = transforms.Compose([
                SafeToTensor(),
                ResizeTensor(self.hr_size_reduced, mode='nearest')  # Nearest for masks to avoid interpolation artifacts (keep values 0 and 1 only)
            ])
            lsm_hr = geo_transform_lsm_hr(lsm_hr)
            # Re-binarize after resizing, just in case
            lsm_hr = (lsm_hr > 0.5).to(lsm_hr.dtype)  # Ensure binary mask (0 and 1)
            sample_dict['lsm_hr'] = lsm_hr


        # Load geo variables (stationary) from full-domain arrays (may be cropped using lr_data_size and lr_cutout_domains)
        if self.geo_variables is not None:
            for geo in self.geo_variables:
                if geo == 'lsm':
                    if self.lsm_full_domain is None:
                        raise ValueError("lsm_full_domain must be provided if 'lsm' is in geo_variables")
                    geo_data = self.lsm_full_domain
                    # logger.info('lsm_full_domain shape:', geo_data.shape)
                    geo_transform = self.geo_transform_lsm
                elif geo == 'topo':
                    if self.topo_full_domain is None:
                        raise ValueError("topo_full_domain must be provided if 'topo' is in geo_variables")
                    geo_data = self.topo_full_domain
                    # logger.info('topo_full_domain shape:', geo_data.shape)
                    geo_transform = self.geo_transform_topo
                else:
                    # Add custom logic for other geo variables when needed
                    geo_data = None
                    geo_transform = None
                if geo_data is not None and self.cutouts:
                    # For geo data, if an LR-specific size and domain are provided, use lr_point
                    if self.lr_data_size is not None and self.lr_cutout_domains is not None and lr_point is not None:
                        geo_data = geo_data[lr_point[0]:lr_point[1], lr_point[2]:lr_point[3]]
                    else:
                        if hr_point is not None:
                            geo_data = geo_data[hr_point[0]:hr_point[1], hr_point[2]:hr_point[3]]
                
                if geo_data is not None and geo_transform is not None:
                    geo_data = geo_transform(geo_data)
                    # For lsm, re-binarize after resizing, just in case
                    if geo == 'lsm':
                        geo_data = (geo_data > 0.5).to(geo_data.dtype)  # Ensure binary mask (0 and 1)

                sample_dict[geo] = geo_data

        # Check if conditional sampling on season (or monthly/daily) is used
        if self.conditional_seasons:
            # Determine class from filename
            if self.n_classes is not None:
                # Seasonal condittion
                if self.n_classes == 4:
                    dateObj = DateFromFile(hr_file_name)
                    classifier = dateObj.determine_season()
                # Monthly condition
                elif self.n_classes == 12:
                    dateObj = DateFromFile(hr_file_name)
                    classifier = dateObj.determine_month()
                # Daily condition
                elif self.n_classes == 366:
                    dateObj = DateFromFile(hr_file_name)
                    classifier = dateObj.determine_day()
                else:
                    raise ValueError('n_classes must be 4, 12 or 365')
            else:
                logger.warning("n_classes is not provided, using date as classifier. This will default to daily condition.")
                # If n_classes is not provided, use the date as a classifier
                dateObj = DateFromFile(hr_file_name)
                classifier = dateObj.determine_month()  # Default to daily condition if n_classes is not specified
            # Convert classifier to tensor
            classifier = torch.tensor(classifier, dtype=torch.long)
            sample_dict['classifier'] = classifier
        # else:
            # A batch cannot contain None, so if no classifier is used, don't add it to the sample_dict
            

        # For SDF, ensure that it is computed for the HR mask (lsm_hr) to get it in same shape as HR
        if self.sdf_weighted_loss:
            if 'lsm_hr' in sample_dict and sample_dict['lsm_hr'] is not None:
                sdf = generate_sdf(sample_dict['lsm_hr'])
                sdf = normalize_sdf(sdf)
                sample_dict['sdf'] = sdf
            else:
                raise ValueError("lsm_hr must be provided for SDF computation if sdf_weighted_loss is True")
            
        # Attach cutout points for reference
        if self.cutouts:
            sample_dict['hr_points'] = hr_point
            sample_dict['lr_points'] = lr_point

        # -------------------------------------------------------------------------------
        # Classifier-Free Guidance dropout (training split only)
        # -------------------------------------------------------------------------------
        cfg_guidance = getattr(self, "cfg", {}).get("classifier_free_guidance", {})
        drop_prob = cfg_guidance.get("drop_prob", 0.1)
        dropped = False
        if self.split == "train" and cfg_guidance.get("enabled", False):
            if torch.rand(()) < cfg_guidance.get(drop_prob, 0.1):
                dropped = True

                # 1) z-scored low-res fields --> set to zero
                for key, val in list(sample_dict.items()):
                    if key.endswith("_lr") and val is not None:
                        sample_dict[key] = torch.zeros_like(val)

                # 2) Bounded geo maps (lsm, topo) -> keep value, append MASK channel
                for geo_key in ("lsm", "topo"):
                    geo = sample_dict.get(geo_key)
                    if geo is not None:
                        mask = torch.zeros_like(geo)        # 0 --> dropped
                        sample_dict[geo_key] = torch.cat([geo, mask], dim=0)  # Append mask channel [2, H, W]

                # 3) scalar season / class index --> special NULL token 
                if "classifier" in sample_dict and sample_dict["classifier"] is not None:
                    null_token = 0
                    sample_dict["classifier"].fill_(null_token)  # Set to NULL token, 0
                
        # ----------------------------------------------------------------------------
        # If NOT dropped, still append a mask channel = 1 to keep the channel count fixed
        # ----------------------------------------------------------------------------
        for geo_key in ("lsm", "topo"):
            geo = sample_dict.get(geo_key)
            if geo is not None:
                if geo.shape[0] == 1:       # I.e. mask is not added yet
                    mask_val = 0.0 if dropped else 1.0
                    mask = torch.full_like(geo, mask_val)       # (1, H, W)
                    sample_dict[geo_key] = torch.cat([geo, mask], dim=0) # (2, H, W)
        # Add item to cache
        self._addToCache(idx, sample_dict)

        return sample_dict #sample

    def __name__(self, idx:int):
        '''
            Return the name of the file based on index.
            Input:
                - idx: index of item to get
        '''
        date = self.common_dates[idx]
        return date #self.hr_file_map[date]


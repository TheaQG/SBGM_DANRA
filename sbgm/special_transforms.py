'''
    This file contains custom transforms for data preprocessing.
'''

# Import libraries and modules 
import torch
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Define custom transforms
class Scale(object):
    '''
        Class for scaling the data to a new interval. 
        The data is scaled to the interval [in_low, in_high].
        The data is assumed to be in the interval [data_min_in, data_max_in].
    '''
    def __init__(self,
                 in_low,
                 in_high,
                 data_min_in = 0,
                 data_max_in = 1
                 ):
        '''
            Initialize the class.
            Input:
                - in_low: lower bound of new interval
                - in_high: upper bound of new interval
                - data_min_in: lower bound of data interval
                - data_max_in: upper bound of data interval
        '''
        self.in_low = in_low
        self.in_high = in_high
        self.data_min_in = data_min_in 
        self.data_max_in = data_max_in

    def __call__(self, sample):
        '''
            Call function for the class - scales the data to the new interval.
            Input:
                - sample: datasample to scale to new interval
        '''
        data = sample
        OldRange = (self.data_max_in - self.data_min_in)
        NewRange = (self.in_high - self.in_low)

        # Generating the new data based on the given intervals
        DataNew = (((data - self.data_min_in) * NewRange) / OldRange) + self.in_low

        return DataNew

# Back transform the scaled data
class ScaleBackTransform(object):
    '''
    Class for back-transforming the scaled data.
    The data is back-transformed to the original interval.
    '''
    def __init__(self,
                 data_min_in = 0,
                 data_max_in = 1
                 ):
        '''
        Initialize the class.
        Input:
            - data_min_in: lower bound of data interval
            - data_max_in: upper bound of data interval
        '''
        self.data_min_in = data_min_in
        self.data_max_in = data_max_in

    def __call__(self, sample):
        '''
        Call function for the class - back-transforms the scaled data.
        Input:
            - sample: data sample to be back-transformed
        '''
        data = sample
        OldRange = (1 - 0)
        NewRange = (self.data_max_in - self.data_min_in)

        # Back-transforming the data
        DataNew = (((data - 0) * NewRange) / OldRange) + self.data_min_in

        return DataNew


import torch

class ZScoreTransform(object):
    '''
    Class for Z-score standardizing the data. 
    The data is standardized to have a mean of 0 and a standard deviation of 1.
    The mean and standard deviation of the training data should be provided.
    '''
    def __init__(self, mean, std):
        '''
        Initialize the class.
        Input:
            - mean: the mean of the global training data
            - std: the standard deviation of the global training data
        '''
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        '''
        Call function for the class - standardizes the data.
        Input:
            - sample: data sample to be standardized
        '''
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)  # Ensure the input is a Tensor
        
        # Ensure mean and std are tensors for broadcasting, preserve their shapes if they are not scalars.
        if not isinstance(self.mean, torch.Tensor):
            self.mean = torch.tensor(self.mean, dtype=torch.float32)
        if not isinstance(self.std, torch.Tensor):
            self.std = torch.tensor(self.std, dtype=torch.float32)

        # Expand as necessary to match the sample dimensions
        if len(sample.shape) > len(self.mean.shape):
            shape_diff = len(sample.shape) - len(self.mean.shape)
            for _ in range(shape_diff):
                self.mean = self.mean.unsqueeze(0)
                self.std = self.std.unsqueeze(0)

        # Standardizing the sample
        standardized_sample = (sample - self.mean) / (self.std + 1e-8)  # Add a small epsilon to avoid division by zero

        return standardized_sample
    
# Back transform the standardized data
class ZScoreBackTransform(object):
    '''
    Class for back-transforming the Z-score standardized data.
    The data is back-transformed to the original distribution with mean and standard deviation.
    '''
    def __init__(self, mean, std):
        '''
        Initialize the class.
        Input:
            - mean: the mean of the training data
            - std: the standard deviation of the training data
        '''
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        '''
        Call function for the class - back-transforms the standardized data.
        Input:
            - sample: data sample to be back-transformed
        '''
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)  # Ensure the input is a Tensor

        # Ensure mean and std are tensors for broadcasting, preserve their shapes if they are not scalars.
        if not isinstance(self.mean, torch.Tensor):
            self.mean = torch.tensor(self.mean, dtype=torch.float32)
        if not isinstance(self.std, torch.Tensor):
            self.std = torch.tensor(self.std, dtype=torch.float32)

        # Expand as necessary to match the sample dimensions
        if len(sample.shape) > len(self.mean.shape):
            shape_diff = len(sample.shape) - len(self.mean.shape)
            for _ in range(shape_diff):
                self.mean = self.mean.unsqueeze(0)
                self.std = self.std.unsqueeze(0)

        # Back-transforming the sample
        back_transformed_sample = (sample * (self.std + 1e-8)) + self.mean  # Add a small epsilon to avoid division by zero

        return back_transformed_sample
    




class PrcpLogTransform(object):
    '''
    Class for log-transforming the precipitation data.
    Data is transformed to log-space and optionally scaled to [0, 1] or to mu=0, sigma=1.
    '''
    def __init__(self,
                 eps=1e-10,
                 scale_type='log_zscore', # 'log_zscore', 'log_01', 'log_minus1_1', 'log', 
                 glob_mean_log=None,
                 glob_std_log=None,
                 glob_min_log=None,
                 glob_max_log=None,
                 buffer_frac=0.5,
                 ):
        '''
        Initialize the class.
        '''
        self.eps = eps
        self.scale_type = scale_type
        self.glob_mean_log = glob_mean_log
        self.glob_std_log = glob_std_log
        self.glob_min_log = glob_min_log
        self.glob_max_log = glob_max_log
        self.buffer_frac = buffer_frac

        if self.glob_min_log is not None and self.glob_max_log is not None:
            # Optionally, expand the log range by a fraction of the range
            log_range = self.glob_max_log - self.glob_min_log
            self.glob_min_log = self.glob_min_log - self.buffer_frac * log_range
            self.glob_max_log = self.glob_max_log + self.buffer_frac * log_range

        if self.scale_type == 'log_zscore':
            if (self.glob_mean_log is None) or (self.glob_std_log is None):
                raise ValueError("Global mean and standard deviation not provided. Using local statistics is not recommended.")
        elif self.scale_type == 'log_01':
            if (self.glob_min_log is None) or (self.glob_max_log is None):
                raise ValueError("Min and max log values not provided. Using global statistics is recommended.")
        elif self.scale_type == 'log_minus1_1':
            if (self.glob_min_log is None) or (self.glob_max_log is None):
                raise ValueError("Min and max log values not provided. Using global statistics is recommended.")
        elif self.scale_type == 'log':
            pass
        else:
            raise ValueError("Invalid scale type. Please choose '01' or 'zscore'.")
        
        pass


    def __call__(self, sample):
        '''
        Call function for the class - log-transforms the data.
        Input:
            - sample: data sample to be log-transformed
            - eps: small epsilon to avoid log(0)
        '''
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)  # Ensure the input is a Tensor

        # Log-transform the sample
        log_sample = torch.log(sample + self.eps) # Add a small epsilon to avoid log(0)

        # logger.debug(f"Min log in sample: {torch.min(log_sample)}")
        # logger.debug(f"Max log in sample: {torch.max(log_sample)}")
        # Scale the log-transformed data to [0,1]ß
        if self.scale_type == 'log_01':
            if (self.glob_min_log is None) or (self.glob_max_log is None):
                # If the min and max log values are not provided, find them in the data
                self.glob_min_log = torch.min(log_sample)
                self.glob_max_log = torch.max(log_sample)
                # BUT WE GENERALLY WANT TO USE GLOBAL STATISTICS FOR LARGE DATASETS
            
            # Shift and scale to [0, 1]: (log_sample - glob_min_log) / (glob_max_log - glob_min_log)
            denom = (self.glob_max_log - self.glob_min_log)
            # If denominator is zero, raise an error
            if denom == 0:
                raise ValueError("The log-range of data is zero. Cannot scale to [0, 1]. Please check the data.")
            log_sample = (log_sample - self.glob_min_log) / (denom)
        
        # Scale the log-transformed data to have mean 0 and std 1
        elif self.scale_type == 'log_zscore':
            # Standardize the log-transformed data
            mu = self.glob_mean_log
            sigma = self.glob_std_log

            if mu is None or sigma is None:
                raise ValueError("Global mean and standard deviation must not be None for 'log_zscore' scaling.")

            log_sample = (log_sample - mu) / (sigma + 1e-8)  
            # logger.debug(f"Min log in sample (zscore): {torch.min(log_sample)}")
            # logger.debug(f"Max log in sample (zscore): {torch.max(log_sample)}")
        elif self.scale_type == 'log_minus1_1':
            # Scale the log-transformed data to [-1, 1]
            if self.glob_min_log is None or self.glob_max_log is None:
                raise ValueError("Min and max log values must not be None for 'log_minus1_1' scaling.")
            log_sample = 2 * ((log_sample - self.glob_min_log) / (self.glob_max_log - self.glob_min_log)) - 1

        elif self.scale_type == 'log':
            pass
        else:
            raise ValueError("Invalid scale type. Please choose 'log_01' or 'log_zscore' or 'log'.")

        return log_sample
    
# Back transform the log-transformed data, with min and max values provided
class PrcpLogBackTransform(object):
    '''
    Class for back-transforming the log-transformed precipitation data.
    The data is back-transformed to the original distribution.
    '''
    def __init__(self,
                 scale_type='log_zscore', # 'log_zscore', 'log_01', 'log_minus1_1'
                 glob_mean_log=None,
                 glob_std_log=None,
                 glob_min_log=None,
                 glob_max_log=None,
                 buffer_frac=0.5,
                #  **kwargs # Swallow any unused keys
                 ):
        '''
        Initialize the class.
        '''
        # Silently ignore extra keywords that are irrelevant for the back-transform
        # _ = kwargs # Prevents "unused variable" linters from complaining
        self.scale_type = scale_type
        self.glob_mean_log = glob_mean_log
        self.glob_std_log = glob_std_log
        self.glob_min_log = glob_min_log
        self.glob_max_log = glob_max_log
        self.buffer_frac = buffer_frac

        if self.glob_min_log is not None and self.glob_max_log is not None:
            # Optionally, expand the log range by a fraction of the range
            logger.info(f'Extended log range from [{self.glob_min_log}, {self.glob_max_log}]')
            log_range = self.glob_max_log - self.glob_min_log
            self.glob_min_log = self.glob_min_log - (self.buffer_frac/2) * log_range
            self.glob_max_log = self.glob_max_log + (self.buffer_frac/2) * log_range
            logger.info(f'to [{self.glob_min_log}, {self.glob_max_log}]\n')

        if self.scale_type == 'log_zscore':
            if (self.glob_mean_log is None) or (self.glob_std_log is None):
                raise ValueError("Global mean and standard deviation not provided. Using local statistics is not recommended.")
        elif self.scale_type == 'log_01':
            if (self.glob_min_log is None) or (self.glob_max_log is None):
                raise ValueError("Min and max log values not provided. Using global statistics is recommended.")
        elif self.scale_type == 'log_minus1_1':
            if (self.glob_min_log is None) or (self.glob_max_log is None):
                raise ValueError("Min and max log values not provided. Using global statistics is recommended.")
        elif self.scale_type == 'log':
            pass
                
        else:
            raise ValueError("Invalid scale type. Please choose from ['log_01', 'log_zscore', 'log_minus1_1', 'log'].")

        pass

    def __call__(self, sample):
        '''
        Call function for the class - back-transforms the log-transformed data.
        Input:
            - sample: data sample to be back-transformed
        '''
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)  # Ensure the input is a Tensor

        if self.scale_type == 'log_01':
            # Back-transform the data to log-space
            log_sample = sample
            # Ensure min and max log values are not None
            if self.glob_max_log is None or self.glob_min_log is None:
                raise ValueError("glob_max_log and glob_min_log must not be None for 'log_01' back-transform.")
            # Scale the log-transformed data back to the original range
            back_transformed_sample = log_sample * (self.glob_max_log - self.glob_min_log) + self.glob_min_log
            # Inverse log-transform the data
            back_transformed_sample = torch.exp(back_transformed_sample)
        elif self.scale_type == 'log_zscore':
            # Back-transform the data to log-space
            mu = self.glob_mean_log
            sigma = self.glob_std_log
            if mu is None or sigma is None:
                raise ValueError("Global mean and standard deviation must not be None for 'log_zscore' back-transform.")
            log_sample = (sample * (sigma + 1e-8)) + mu
            # Inverse log-transform the data
            back_transformed_sample = torch.exp(log_sample)
        elif self.scale_type == 'log_minus1_1':
            # Back-transform the data to log-space
            if self.glob_max_log is None or self.glob_min_log is None:
                raise ValueError("glob_max_log and glob_min_log must not be None for 'log_minus1_1' back-transform.")
            log_sample = 0.5 * (sample + 1) * (self.glob_max_log - self.glob_min_log) + self.glob_min_log
            # Inverse log-transform the data
            back_transformed_sample = torch.exp(log_sample)
        elif self.scale_type == 'log':
            back_transformed_sample = torch.exp(sample)
        else:
            raise ValueError("Invalid scale type. Please choose from ['log_01', 'log_zscore', 'log_minus1_1', 'log'].")

        return back_transformed_sample


def build_back_transforms(hr_var,
                          hr_scaling_method, hr_scaling_params,
                          lr_vars, lr_scaling_methods, lr_scaling_params):
    """
    Returns a dict that maps plot-keys (e.g. 'prcp_hr', 'prcp_lr', 'generated')
    to callable inverse-transform objects.
    """
    bt = {}

    # ---------- HR / generated -------------------------------------------
    hr_key = f"{hr_var}_hr"
    if hr_scaling_method in {"log", "log_01", "log_minus1_1", "log_zscore"}:
        inv = PrcpLogBackTransform(scale_type=hr_scaling_method,
                                glob_mean_log=hr_scaling_params["glob_mean_log"],
                                glob_std_log=hr_scaling_params["glob_std_log"],
                                glob_min_log=hr_scaling_params["glob_min_log"],
                                glob_max_log=hr_scaling_params["glob_max_log"],
                                buffer_frac=hr_scaling_params["buffer_frac"])
    elif hr_scaling_method == "zscore":
        inv = ZScoreBackTransform(hr_scaling_params["glob_mean"],
                                  hr_scaling_params["glob_std"])
    elif hr_scaling_method == "01":
        inv = ScaleBackTransform(0, 1,
                                 hr_scaling_params["glob_min"],
                                 hr_scaling_params["glob_max"])
    else:
        raise ValueError(f"Unknown HR scaling method: {hr_scaling_method}")

    # ‘generated’ images are in the same space as the HR target
    bt[hr_key]   = inv
    bt["generated"] = inv

    # ---------- LR conditions --------------------------------------------
    for cond, mth, prm in zip(lr_vars, lr_scaling_methods, lr_scaling_params):
        key = f"{cond}_lr"
        if mth in {"log", "log_01", "log_minus1_1", "log_zscore"}:
            bt[key] = PrcpLogBackTransform(scale_type=mth,
                                           glob_mean_log=prm["glob_mean_log"],
                                           glob_std_log=prm["glob_std_log"],
                                           glob_min_log=prm["glob_min_log"],
                                           glob_max_log=prm["glob_max_log"],
                                           buffer_frac=prm["buffer_frac"])
        elif mth == "zscore":
            bt[key] = ZScoreBackTransform(prm["glob_mean"], prm["glob_std"])
        elif mth == "01":
            bt[key] = ScaleBackTransform(0, 1, prm["glob_min"], prm["glob_max"])
        else:
            raise ValueError(f"Unknown LR scaling method: {mth}")

    return bt
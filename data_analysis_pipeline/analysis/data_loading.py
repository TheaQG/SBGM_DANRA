import os 
import re
import zarr
import numpy as np
from glob import glob
from datetime import datetime

from data_analysis_pipeline.analysis.path_utils import build_data_path
from data_analysis_pipeline.analysis.variable_utils import correct_variable_units, crop_to_region, get_var_name_short
from concurrent.futures import ProcessPoolExecutor


def get_date_from_filename(file_path, data_type, var_name_short):
    """
        Extract the date from the filename.
    """

    # Get filename from file path (basename means removing the directory)
    filename = os.path.basename(file_path)

    if data_type == 'npz':
        # Extract date from filename like varname_YYYYMMDD.npz
        match = re.search(rf"{re.escape(var_name_short)}_(\d{8})\.npz$", filename)        
    elif data_type == 'zarr':
        # For zarr, filepath is like {split}.zarr/varname_YYYYMMDD
        filename = os.path.basename(file_path)
        match = re.search(rf"{re.escape(var_name_short)}_(\d{8})$", filename)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    if match:
        date_str = match.group(1)
        timestamp = datetime.strptime(date_str, "%Y%m%d")
    else:
        raise ValueError(f"Date not found in filename: {filename}")

    return timestamp


class DataLoader:
    def __init__(self,
                 base_dir: str,
                 n_workers: int,
                 variable: str,
                 model: str,
                 domain_size: list,
                 split: str,
                 crop_region: list,
                 subdir: str):

        self.variable = variable
        self.var_name_short = get_var_name_short(variable)
        self.subdir = subdir
        
        self.domain_size = domain_size
        self.crop_region = crop_region
        self.split = split
        self.zarr = self.split in ["train", "valid", "test"]

        self.model_type = model
        self.data_dir = build_data_path(base_dir, self.model_type, self.variable, self.domain_size, self.split, zarr=self.zarr)
        
        self.n_workers = n_workers

    def _get_file_list(self):
        if self.zarr:
            # Access the internal keys for each day from Zarr
            root = self.data_dir
            zarr_keys = sorted(glob(os.path.join(root, f"{self.var_name_short}_*")))
            return zarr_keys
        else:
            return sorted(glob(os.path.join(self.data_dir, f"{self.var_name_short}_*.npz")))


    def load(self):
        file_list = self._get_file_list()
        if self.n_workers > 1:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                results = list(executor.map(self._process_wrapper, file_list))
        else:
            results = [self._process_wrapper(f) for f in file_list]

        # Get the data sorted
        cutouts, timestamps = zip(*results)
        sorted_pairs = sorted(zip(timestamps, cutouts))
        timestamps, cutouts = zip(*sorted_pairs)

        return {
            "cutouts": list(cutouts),
            "timestamps": list(timestamps)
        }
    
    def _process_wrapper(self, file_path):
        cutout, timestamp = process_single_file(file_path, self.variable, self.var_name_short, self.crop_region)
        return cutout, timestamp

def process_single_file(file_path, variable, var_name_short, crop_region):
    """
        Reads file, applies variable correction, crops region etc.
    """
    if file_path.endswith(".npz"):
        with np.load(file_path) as npz:
            # Take the first available key if neither 'data' nor 'arr_0' exists
            key = 'data' if 'data' in npz else ('arr_0' if 'arr_0' in npz else list(npz.keys())[0])
            data = npz[key]
            data = np.array(data)  # Ensure it's a numpy array 

            # Extract date from filename like 'prcp_20200101.npz'
            timestamp = get_date_from_filename(file_path, 'npz', var_name_short)

    elif re.match(r".*\.zarr/" + var_name_short + r"_\d{8}", file_path):
        # Load from Zarr archive subpath
        zarr_root, key = os.path.split(file_path)
        zarr_group = zarr.open_group(zarr_root, mode='r')
        try:
            data = zarr_group[key][...]
        except KeyError:
            raise ValueError(f"Key {key} not found in Zarr group at {zarr_root}")

        # Extract date from filename like '{...}.zarr/prcp_20200101'
        timestamp = get_date_from_filename(file_path, 'zarr', var_name_short)

    else:
        raise ValueError(f"Unsupported file or path: {file_path}")
    
    data = correct_variable_units(variable, data)

    if crop_region is not None:
        data = crop_to_region(data, crop_region)

    if data is None:
        raise ValueError(f"Data is None after processing: {file_path}")

    return data, timestamp


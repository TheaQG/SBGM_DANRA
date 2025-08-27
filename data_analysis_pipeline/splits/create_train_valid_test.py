import os
import shutil
import zarr
import logging
import re
from glob import glob
from tqdm import tqdm
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def extract_date(filename):
    """ Extract the date string (YYYYMMDD) from filename like 'data_YYYYMMDD.npz' """
    match = re.search(r'_(\d{8})\.npz$', filename)
    return match.group(1) if match else None

def create_data_splits(cfg):
    """
        Copies .npz files from 'all/' to 'train/', 'valid/' and 'test/' folders, based on split ratios. 
        Keeps the original files in 'all/' and supports multiple variables.
    """
    data_dir = cfg.paths.data_dir
    hr_vars = cfg.highres.variables
    lr_vars = cfg.lowres.condition_variables
    hr_model = cfg.highres.model
    lr_model = cfg.lowres.model
    hr_domain_size = cfg.highres.domain_size
    hr_domain_size_str = "x".join(map(str, hr_domain_size))
    lr_domain_size = cfg.lowres.domain_size
    lr_domain_size_str = "x".join(map(str, lr_domain_size))
    data_split_type = cfg.split_params.split_type
    data_split_params = cfg.split_params.data_split_params

    overwrite = cfg.split_params.get('overwrite', False)
    seed = cfg.split_params.get('seed', 42)

    if cfg.highres.filtered:
        all_data_str_hr = cfg.highres.filter_str
    else:
        all_data_str_hr = 'all'
    # Set paths to where HR and LR data lives
    HR_DIRS = {
        hr_var: os.path.join(data_dir, f'data_{hr_model}', 'size_' + hr_domain_size_str, f"{hr_var}_{hr_domain_size_str}", all_data_str_hr)
        for hr_var in hr_vars
    }
    LR_DIRS = {
        lr_var: os.path.join(data_dir, f'data_{lr_model}', 'size_' + lr_domain_size_str, f"{lr_var}_{lr_domain_size_str}", 'all')
        for lr_var in lr_vars
    }

    # Collect all available HR files per date 
    date_to_hr_files = {}
    for hr_var, hr_dir in HR_DIRS.items():
        files = sorted(glob(os.path.join(hr_dir, '*.npz')))
        logger.info(f"[HR - {hr_var}] Found {len(files)} files in {hr_dir}")
        for f in files:
            date = extract_date(os.path.basename(f))
            if date:
                date_to_hr_files.setdefault(date, {})[hr_var] = f

    # === SAFETY CHECK ===
    # === STEP 1: Only retain dates where all HR vars are available ===
    valid_dates = [date for date, d in date_to_hr_files.items() if len(d) == len(hr_vars)]
    logger.info(f"Valid dates with all HR vars: {len(valid_dates)}")

    # === STEP 2: Further filter valid_dates based on LR variable availability ===
    for lr_var, lr_dir in LR_DIRS.items():
        lr_dates = set()
        for f in glob(os.path.join(lr_dir, "*npz")):
            date = extract_date(os.path.basename(f))
            if date:
                lr_dates.add(date)

        # Identify which dates are missing
        current_valid_dates = set(valid_dates)
        missing_dates = current_valid_dates - lr_dates

        if missing_dates:
            logger.warning(
                f"[{lr_var}] Missing {len(missing_dates)} dates from LR data."
                f"Examples: {sorted(list(missing_dates))[:5]}..."
            )

        # Restrict to dates that exist for the LR variable
        valid_dates = [d for d in valid_dates if d in lr_dates]

    logger.info(f"Final valid dates after HR+LR check: {len(valid_dates)}")

    # Determine splits
    if data_split_type == "Time":
        train_years = set(map(str, np.arange(*data_split_params['train_years'])))
        valid_years = set(map(str, np.arange(*data_split_params['valid_years'])))
        test_years = set(map(str, np.arange(*data_split_params['test_years'])))

        split_dates = {
            'train': [d for d in valid_dates if d[:4] in train_years],
            'valid': [d for d in valid_dates if d[:4] in valid_years],
            'test':  [d for d in valid_dates if d[:4] in test_years],
        }

    elif data_split_type == "Random":
        np.random.seed(seed)
        np.random.shuffle(valid_dates)
        total = len(valid_dates)

        n_train = int(data_split_params['train_frac'] * total)
        n_valid = int(data_split_params['valid_frac'] * total)

        split_dates = {
            'train': valid_dates[:n_train],
            'valid': valid_dates[n_train: n_train+n_valid],
            'test':  valid_dates[n_train + n_valid:],
        }

    else:
        raise ValueError(f"Unknown split type: {data_split_type}")


    for split, dates in split_dates.items():
        logger.info(f"{split.capitalize()} files: {len(dates)}")


        # HR copy
        for hr_var in hr_vars:
            logger.info(f"Copying HR variable: {hr_var} for split: {split}")
            # HR destination
            dest_dir = os.path.join(data_dir, f"data_{hr_model}", f"size_{hr_domain_size_str}", f"{hr_var}_{hr_domain_size_str}", split)
            os.makedirs(dest_dir, exist_ok=True)
            for d in tqdm(dates, desc=f"[{split.upper()}] Copying HR {hr_var}"):
                f = date_to_hr_files[d][hr_var]
                dst = os.path.join(dest_dir, os.path.basename(f))
                if not os.path.exists(dst) or overwrite:
                    shutil.copy(f, dst)

        # Copy corresponding LR files for each var
        for lr_var in lr_vars:
            logger.info(f"Copying LR variable: {lr_var} for split: {split}")
            lr_dir = LR_DIRS[lr_var]
            lr_dest = os.path.join(data_dir, f'data_{lr_model}', f'size_{lr_domain_size_str}', f"{lr_var}_{lr_domain_size_str}", split)
            os.makedirs(lr_dest, exist_ok=True)

            for d in tqdm(dates, desc=f"[{split.upper()}] Copying LR {lr_var}"):
                matching_file = sorted(glob(os.path.join(lr_dir, f"*{d}*.npz")))
                if matching_file:
                    dst = os.path.join(lr_dest, os.path.basename(matching_file[0]))
                    if not os.path.exists(dst) or overwrite:
                        shutil.copy(matching_file[0], dst)
                else:
                    logger.warning(f"No matching LR file found for date {d} in {lr_dir}")

    logger.info("Data splitting complete.")




def convert_splits_to_zarr(cfg):
    """
        Converts .npz split folders (train/valid/test) into .zarr archives for both HR and LR variabeles
        Each zarr group is saved as: <split>.zarr inside the variable's folder
    """
    data_dir = cfg.paths.data_dir
    hr_vars = cfg.highres.variables
    lr_vars = cfg.lowres.condition_variables
    hr_model = cfg.highres.model
    lr_model = cfg.lowres.model
    hr_domain_size = cfg.highres.domain_size
    hr_domain_size_str = "x".join(map(str, hr_domain_size))
    lr_domain_size = cfg.lowres.domain_size
    lr_domain_size_str = "x".join(map(str, lr_domain_size))
    overwrite = cfg.split_params.get("overwrite_zarr", True)
    keep_npz = cfg.split_params.get("keep_npz_after_zarr", False)


    for split in ["train", "valid", "test"]:
        logger.info(f"\n=== Converting split: {split} ===")

        # ---- High-resolution (DANRA) ----
        for hr_var in hr_vars:
            hr_base = os.path.join(data_dir, f"data_{hr_model}", f"size_{hr_domain_size_str}", f"{hr_var}_{hr_domain_size_str}")
            split_dir = os.path.join(hr_base, split)
            zarr_dir = os.path.join(hr_base, "zarr_files")
            os.makedirs(zarr_dir, exist_ok=True)
            zarr_path = os.path.join(zarr_dir, f"{split}.zarr")

            if overwrite and os.path.exists(zarr_path):
                shutil.rmtree(zarr_path)
                logger.info(f"[{hr_model} - {hr_var}] Overwriting {zarr_path}")

            logger.info(f"[{hr_model} - {hr_var}] Writing Zarr file: {zarr_path}")
            zarr_group = zarr.open_group(zarr_path, mode="w")

            npz_files = sorted(glob(os.path.join(split_dir, "*.npz")))
            for f in tqdm(npz_files, desc=f"{hr_var} {split}"):
                fname = os.path.splitext(os.path.basename(f))[0]
                with np.load(f) as data:
                    for key in data:
                        zarr_group.array(f"{fname}/{key}", data[key], chunks=True, dtype=np.float32)

            if not keep_npz: 
                logger.info(f"Deleting HR {hr_var} .npz files from {split_dir}")
                for f in npz_files:
                    os.remove(f)
        
        
        # ---- Low-res (ERA5) ----
        for lr_var in lr_vars:
            lr_base = os.path.join(data_dir, f"data_{lr_model}", f"size_{lr_domain_size_str}", f"{lr_var}_{lr_domain_size_str}")
            split_dir = os.path.join(lr_base, split)
            zarr_dir = os.path.join(lr_base, "zarr_files")
            os.makedirs(zarr_dir, exist_ok=True)
            zarr_path = os.path.join(zarr_dir, f"{split}.zarr")

            if overwrite and os.path.exists(zarr_path):
                shutil.rmtree(zarr_path)
                logger.info(f"[{lr_model} - {lr_var}] Overwriting: {zarr_path}")
            
            logger.info(f"[{lr_model} - {lr_var}] Writing zarr file: {zarr_path}")
            zarr_group = zarr.open_group(zarr_path, mode="w")

            npz_files = sorted(glob(os.path.join(split_dir, "*.npz")))
            for f in tqdm(npz_files, desc=f"{lr_var} {split}"):
                fname = os.path.splitext(os.path.basename(f))[0]
                with np.load(f) as data:
                    for key in data:
                        zarr_group.array(f"{fname}/{key}", data[key], chunks=True, dtype=np.float32)
            
            if not keep_npz:    
                logger.info(f"Deleting LR {lr_var} .npz files from {split_dir}")
                for f in npz_files:
                    os.remove(f) # Delete after writing
                


    logger.info("Zarr conversion complete.")
            














# from utils import convert_npz_to_zarr


# def create_train_valid_test_data_zarr(args):
#     '''
#         Function to make zarr data from all data in the data directory.
#         Saves the zarr data to the zarr_files directory.
#     '''
#     data_dir = args.path_data
#     hr_var = args.hr_var
#     lr_var = args.lr_var
#     data_split_type = args.data_split_type
#     if data_split_type == 'Time':
#         train_years = args.train_years
#         valid_years = args.valid_years
#         test_years = args.test_years
#         data_split_params = {'train_years': train_years,
#                              'valid_years': valid_years,
#                              'test_years': test_years
#                              }
#         print(f'\nSplitting data in time with the following years:')
#         print(f'Train years: {train_years}')
#         print(f'Valid years: {valid_years}')
#         print(f'Test years: {test_years}\n')

#     elif data_split_type == 'Random':
#         train_frac = args.train_frac
#         valid_frac = args.valid_frac
#         test_frac = args.test_frac
#         data_split_params = {'train_frac': train_frac,
#                              'valid_frac': valid_frac,
#                              'test_frac': test_frac
#                              }
#         print(f'\nSplitting data randomly with the following fractions:')
#         print(f'Train fraction: {train_frac}')
#         print(f'Valid fraction: {valid_frac}')
#         print(f'Test fraction: {test_frac}\n')

#     # Set the paths to all data
#     LR_PATH = data_dir + 'data_ERA5/size_589x789/' + lr_var + '_589x789/'
#     HR_PATH = data_dir + 'data_DANRA/size_589x789/' + hr_var + '_589x789/'

#     LR_PATH_ALL = LR_PATH + 'all_filtered/'
#     HR_PATH_ALL = HR_PATH + 'all_filtered/'

#     # Check if /train, /valid, /test directories exist. If not, create them
#     if not os.path.exists(LR_PATH + 'train'):
#         os.makedirs(LR_PATH + 'train')
#     if not os.path.exists(LR_PATH + 'valid'):
#         os.makedirs(LR_PATH + 'valid')
#     if not os.path.exists(LR_PATH + 'test'):
#         os.makedirs(LR_PATH + 'test')

#     if not os.path.exists(HR_PATH + 'train'):
#         os.makedirs(HR_PATH + 'train')
#     if not os.path.exists(HR_PATH + 'valid'):
#         os.makedirs(HR_PATH + 'valid')
#     if not os.path.exists(HR_PATH + 'test'):
#         os.makedirs(HR_PATH + 'test')

#     # Empty the /train, /valid, /test directories
#     print(f'\nEmptying LR and HR directories')
#     for file in os.listdir(LR_PATH + 'train'):
#         os.remove(LR_PATH + 'train/' + file)
#     for file in os.listdir(LR_PATH + 'valid'):
#         os.remove(LR_PATH + 'valid/' + file)
#     for file in os.listdir(LR_PATH + 'test'):
#         os.remove(LR_PATH + 'test/' + file)

#     for file in os.listdir(HR_PATH + 'train'):
#         os.remove(HR_PATH + 'train/' + file)
#     for file in os.listdir(HR_PATH + 'valid'):
#         os.remove(HR_PATH + 'valid/' + file)
#     for file in os.listdir(HR_PATH + 'test'):
#         os.remove(HR_PATH + 'test/' + file)


#     print(f'\nMoving LR files from {LR_PATH_ALL}')
#     print(f'Moving HR files from {HR_PATH_ALL}\n')

#     if data_split_type == 'Time':
#         print('\nSplitting data in time with the following years:')
#         # Define the splits
#         train_years = data_split_params['train_years']
#         valid_years = data_split_params['valid_years']
#         test_years = data_split_params['test_years']
#         print(f'Train years: {train_years}')
#         print(f'Valid years: {valid_years}')
#         print(f'Test years: {test_years}\n')

#         # Copy all .npz data files to the correct directory. The year is the [-12:-8] part of the filename (including the .npz extension)
#         print('Copying LR files to /train, /valid, /test directories')
#         n_train = 0
#         n_valid = 0
#         n_test = 0
#         for file in sorted(os.listdir(LR_PATH_ALL)):
#             if file[-12:-8] in train_years:
#                 # Copy the files to the correct directories (NOT RENAMING THEM)
#                 shutil.copy(LR_PATH_ALL + file, LR_PATH + 'train/' + file)
#                 n_train += 1
#             elif file[-12:-8] in valid_years:
#                 # Copy the files to the correct directories (NOT RENAMING THEM)
#                 shutil.copy(LR_PATH_ALL + file, LR_PATH + 'valid/' + file)
#                 n_valid += 1
#             elif file[-12:-8] in test_years:
#                 # Copy the files to the correct directories (NOT RENAMING THEM)
#                 shutil.copy(LR_PATH_ALL + file, LR_PATH + 'test/' + file)
#                 n_test += 1
#         print(f'\nCopied {n_train} files to /train, {n_valid} files to /valid and {n_test} files to /test\n')
#         # Now make .zarr files from files in the /train, /valid, /test directories (.npz for both LR and HR)
#         convert_npz_to_zarr(LR_PATH + 'train/', LR_PATH + 'zarr_files/train.zarr')
#         convert_npz_to_zarr(LR_PATH + 'valid/', LR_PATH + 'zarr_files/valid.zarr')
#         convert_npz_to_zarr(LR_PATH + 'test/', LR_PATH + 'zarr_files/test.zarr')

#         print('Copying HR files to /train, /valid, /test directories')
#         n_train = 0
#         n_valid = 0
#         n_test = 0
#         for file in sorted(os.listdir(HR_PATH_ALL)):
#             if file[-12:-8] in train_years:
#                 # Copy the files to the correct directories (NOT RENAMING THEM)
#                 shutil.copy(HR_PATH_ALL + file, HR_PATH + 'train/' + file)
#                 n_train += 1
#             elif file[-12:-8] in valid_years:
#                 #W Copy the files to the correct directories (NOT RENAMING THEM)
#                 shutil.copy(HR_PATH_ALL + file, HR_PATH + 'valid/' + file)
#                 n_valid += 1
#             elif file[-12:-8] in test_years:
#                 # Copy the files to the correct directories (NOT RENAMING THEM)
#                 shutil.copy(HR_PATH_ALL + file, HR_PATH + 'test/' + file)
#                 n_test += 1
#         print(f'\nCopied {n_train} files to /train, {n_valid} files to /valid and {n_test} files to /test\n')
#         # Now make .zarr files from files in the /train, /valid, /test directories (.nc for HR and .npz for LR)
#         convert_npz_to_zarr(HR_PATH + 'valid/', HR_PATH + 'zarr_files/valid.zarr')
#         convert_npz_to_zarr(HR_PATH + 'train/', HR_PATH + 'zarr_files/train.zarr')
#         convert_npz_to_zarr(HR_PATH + 'test/', HR_PATH + 'zarr_files/test.zarr')

#     elif data_split_type == 'Random':
#         # PROBLEM WITH RANDOM: NOT THE SAME DATA IN ERA5 AND DANRA - different years. 
#         # (should not be a problem on LUMI, but locally it is (for now))
        
#         # Define the random splits
#         train_frac = data_split_params['train_frac']
#         valid_frac = data_split_params['valid_frac']
#         test_frac = data_split_params['test_frac']

#         # Figure out, how many files to put in each directory
#         n_files = len(os.listdir(LR_PATH_ALL))
#         n_train = int(n_files * train_frac)
#         n_valid = int(n_files * valid_frac)
#         n_test = n_files - n_train - n_valid

#         # Make a random permutation of the indices 
#         indices = np.random.permutation(n_files)
#         print(indices)

#         # Get the random indices for the train, valid and test sets
#         train_indices = indices[:n_train]
#         valid_indices = indices[n_train:n_train+n_valid]
#         test_indices = indices[n_train+n_valid:]

#         # Make 0/1 arrays for the train, valid and test sets
#         train_mask = np.zeros(n_files)
#         valid_mask = np.zeros(n_files)
#         test_mask = np.zeros(n_files)
#         train_mask[train_indices] = 1
#         valid_mask[valid_indices] = 1
#         test_mask[test_indices] = 1

#         # Make lists of the files in the LR and HR directories to be able to sort them (omit the .DS_Store file)
#         LR_files = sorted(os.listdir(LR_PATH_ALL))
#         HR_files = sorted(os.listdir(HR_PATH_ALL))
#         if '.DS_Store' in LR_files:
#             LR_files.remove('.DS_Store')
#         if '.DS_Store' in HR_files:
#             HR_files.remove('.DS_Store')

#         # Copy all .npz data files to the correct directory
#         print('Copying LR files to /train, /valid, /test directories')
#         for i, file in enumerate(LR_files):
#             if train_mask[i] == 1:
#                 # Copy the files to the correct directories (NOT RENAMING THEM)
#                 shutil.copy(LR_PATH_ALL + file, LR_PATH + 'train/' + file)
#             elif valid_mask[i] == 1:
#                 # Copy the files to the correct directories (NOT RENAMING THEM)
#                 shutil.copy(LR_PATH_ALL + file, LR_PATH + 'valid/' + file)
#             elif test_mask[i] == 1:
#                 # Copy the files to the correct directories (NOT RENAMING THEM)
#                 shutil.copy(LR_PATH_ALL + file, LR_PATH + 'test/' + file)
#         # Now make .zarr files from files in the /train, /valid, /test directories (.nc for HR and .npz for LR)
#         convert_npz_to_zarr(LR_PATH + 'train/', LR_PATH + 'zarr_files/train.zarr')

#         print('Copying HR files to /train, /valid, /test directories')
#         for i, file in enumerate(HR_files):
#             if train_mask[i] == 1:
#                 # Copy the files to the correct directories (NOT RENAMING THEM)
#                 shutil.copy(HR_PATH_ALL + file, HR_PATH + 'train/' + file)
#             elif valid_mask[i] == 1:
#                 # Copy the files to the correct directories (NOT RENAMING THEM)
#                 shutil.copy(HR_PATH_ALL + file, HR_PATH + 'valid/' + file)
#             elif test_mask[i] == 1:
#                 # Copy the files to the correct directories (NOT RENAMING THEM)
#                 shutil.copy(HR_PATH_ALL + file, HR_PATH + 'test/' + file)
#         # Now make .zarr files from files in the /train, /valid, /test directories (.nc for HR and .npz for LR)
#         convert_npz_to_zarr(LR_PATH + 'valid/', LR_PATH + 'zarr_files/valid.zarr')

#     else:
#         raise ValueError('Data split type not recognized')
    
# def launch_split_from_args():
#     parser = argparse.ArgumentParser(description='Split data in time or randomly and make zarr files')
#     parser.add_argument('--path_data', type=str, default='/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/', help='The data directory')
#     parser.add_argument('--hr_var', type=str, default='temp', help='The high resolution variable')
#     parser.add_argument('--lr_var', type=str, default='temp', help='The low resolution variable')
#     parser.add_argument('--data_split_type', type=str, default='Time', help='The data split type')
#     parser.add_argument('--train_years', type=list, default=np.arange(1990, 2015).astype(str), help='The training years')
#     parser.add_argument('--valid_years', type=list, default=np.arange(2015, 2018).astype(str), help='The validation years')
#     parser.add_argument('--test_years', type=list, default=np.arange(2018, 2021).astype(str), help='The test years')
#     parser.add_argument('--train_frac', type=float, default=0.7, help='The training fraction')
#     parser.add_argument('--valid_frac', type=float, default=0.1, help='The validation fraction')
#     parser.add_argument('--test_frac', type=float, default=0.2, help='The test fraction')

#     args = parser.parse_args()

#     create_train_valid_test_data_zarr(args)

    
    


# if __name__ == '__main__':
#     launch_split_from_args()

#     # data_dir = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/'
#     # hr_var = 'temp'
#     # lr_var = 'temp'
#     # data_split_type = 'Time'
#     # train_years = np.arange(1990, 2016).astype(str)
#     # valid_years = np.arange(2016, 2019).astype(str)
#     # test_years = np.arange(2019, 2022).astype(str)

#     # data_split_params = {'train_years': train_years,
#     #                      'valid_years': valid_years,
#     #                      'test_years': test_years
#     #                      }
#     # data_split_type = 'Random'
#     # data_split_params = {'train_frac': 0.7,
#     #                      'valid_frac': 0.1,
#     #                      'test_frac': 0.2
#     #                      }

#     # create_train_valid_test_data_zarr(data_dir, hr_var, lr_var, data_split_type, data_split_params)

    



    


    
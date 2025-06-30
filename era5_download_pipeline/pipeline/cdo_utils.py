'''
    CDO utilities for regridding and converting NetCDF files to daily statistics.
    This module provides functions to convert NetCDF files to daily statistics,
    regrid data to a specific grid, generate regridding weights, and convert daily
    NetCDF data to .npz files for each day.
'''


from datetime import datetime, timedelta

import os
import subprocess

import netCDF4 as nc
import numpy as np


import logging
logger = logging.getLogger(__name__)

def convert_to_daily_stat(input_file,
                          stat,
                          output_file):
    '''
        Convert the input NetCDF file to a daily statistic (e.g., sum, mean, max).
    '''
    logger.info(f"Converting {input_file} to daily {stat} -> {output_file}")

    cdo_command = ["cdo", stat, input_file, output_file]

    subprocess.run(cdo_command, check=True)

def regrid_to_danra(input_nc,
                    output_nc,
                    weights_file=None,
                    grid_file="/path/to/mygrid_danra_small",
                    interpolation_method='bilinear'):
    '''
        Regrid the input NetCDF file to the DANRA grid using bilinear interpolation.
    '''
    logger.info(f"Regridding {input_nc} to {output_nc} using {interpolation_method} interpolation...")
    if weights_file is not None:
        logger.info(f"Using weights file: {weights_file}")

    # If weights file is not provided, just remap directly
    if weights_file is None:
        if interpolation_method == 'bilinear':
            # Use the grid file directly for bilinear interpolation
            cdo_command = [
                "cdo", f"remap,{grid_file}", input_nc, output_nc
            ]
        elif interpolation_method == 'nearest':
            # Use the grid file directly for nearest neighbor interpolation
            cdo_command = [
                "cdo", f"remapnn,{grid_file}", input_nc, output_nc
            ]
        else:
            raise ValueError(f"Unsupported interpolation method: {interpolation_method}")
    # If weights file is provided, use it for regridding
    else:
        if interpolation_method == 'bilinear':
            cdo_command = [
                "cdo", f"remapbil,{grid_file},{weights_file}", input_nc, output_nc
            ]
        elif interpolation_method == 'nearest':
            cdo_command = [
                "cdo", f"remapnn,{grid_file},{weights_file}", input_nc, output_nc
            ]
        else:
            raise ValueError(f"Unsupported interpolation method: {interpolation_method}")

    subprocess.run(cdo_command, check=True)


def generate_regridding_weights(reference_file,
                                weights_file,
                                grid_file="/path/to/mygrid_danra_small"):
    '''
        Generates regridding weights for converting data from a reference grid to a target grid.
        Uses CDO to create bilinear interpolation weights.
        One-time step to create bilinear weights
    '''
    if not os.path.exists(weights_file):
        logger.info(f"Creating weights file: {weights_file}")

        cdo_command = ["cdo", "genbil", grid_file, reference_file, weights_file]

        subprocess.run(cdo_command, check=True)
    else:
        logger.info("Weights file already exists.")


def convert_daily_to_npz(input_nc,
                         output_dir,
                         year,
                         var_str='pev'):
    '''
        Converts daily NetCDF data to individual .npz files for each day.
        Saves the data in a specified output directory.
    '''
    logger.info(f"Converting {input_nc} to daily .npz files in {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    ds = nc.Dataset(input_nc) # type: ignore
    var_keys = list(ds.variables.keys())
    # Exclude coordinate variables
    data_var_key = [k for k in var_keys if k not in ['time', 'latitude', 'longitude']][0]
    data = ds.variables[data_var_key][:]

    os.makedirs(output_dir, exist_ok=True)

    for day_idx in range(data.shape[0]):
        day_data = data[day_idx, :, :]
        date_str = day_to_date(day_idx + 1, int(year)).replace('_', '')
        day_fn = f"{var_str}_589x789_{date_str}.npz"
        day_path = os.path.join(output_dir, day_fn)
        logger.info(f"Saving day: {date_str}...")

        np.savez_compressed(day_path, data=day_data)

    ds.close()





# Define a function to convert a day number to a date string
def day_to_date(day_number, year_use):
    '''
        Converts a day number to a date string in the format YYYY_MM_DD
        If underscores are unwanted add .replace('_','') to the return statement
    '''
    date_format = '%Y_%m_%d'
    # create a datetime object for January 1st of the given year_use
    start_date = datetime(year_use, 1, 1)
    # add the number of days to the start date
    result_date = start_date + timedelta(days=day_number-1)
    # format the date string using the specified format
    return result_date.strftime(date_format)#.replace('_','')


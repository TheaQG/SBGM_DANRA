"""
Modular pipeline for downloading and preprocessing ERA5 data for downscaling.

Split into:
1. download_era5_data()
2. convert_to_daily_stat()
3. regrid_to_danra()
4. generate_regridding_weights()
5. convert_daily_to_npz()
6. transfer_to_lumi()
"""

import os
import subprocess
import cdsapi
import netCDF4 as nc
import numpy as np
from datetime import datetime

# Constants (these can later be moved to a config file)
TMP_DIR = "/tmp/era5_downloads"
GRID_FILE = "/path/to/mygrid_danra_small"
WEIGHTS_FILE = "/path/to/ERA5_to_DANRA_Grid_bil_weights.nc"
LUMI_USER = "your_user"
LUMI_HOST = "lumi.csc.fi"
LUMI_PATH = "/scratch/project_xxx/your_name/data"
os.makedirs(TMP_DIR, exist_ok=True)

# ERA5 configuration
VARIABLES = {
    "potential_evaporation": ("pev", "daysum"),
    "convective_available_potential_energy": ("cape", "daymax"),
    "vertical_integral_of_northward_water_vapour_flux": ("wvf_north", "daymean"),
    "vertical_integral_of_eastward_water_vapour_flux": ("wvf_east", "daymean"),
    "mean_sea_level_pressure": ("msl", "daymean"),
}
AREA = [60, -80, 40, 40]  # [N, W, S, E]
DATASET = "reanalysis-era5-single-levels"
FORMAT = "netcdf"

c = cdsapi.Client()

def download_era5_data(year, variable, target_path):
    print(f"Downloading {variable} for {year}...")
    c.retrieve(
        DATASET,
        {
            "product_type": "reanalysis",
            "format": FORMAT,
            "variable": variable,
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": AREA,
        },
        target=target_path
    )

def convert_to_daily_stat(input_file, stat, output_file):
    print(f"Converting {input_file} to daily {stat} -> {output_file}")
    subprocess.run(["cdo", stat, input_file, output_file], check=True)

def regrid_to_danra(input_file, output_file, weights_file):
    print(f"Regridding {input_file} -> {output_file}")
    subprocess.run(["cdo", "remap", f"{GRID_FILE},{weights_file}", input_file, output_file], check=True)

def generate_regridding_weights(reference_file, weights_file):
    if not os.path.exists(weights_file):
        print(f"Creating weights file: {weights_file}")
        subprocess.run(["cdo", "genbil", GRID_FILE, reference_file, weights_file], check=True)
    else:
        print("Weights file already exists.")

def convert_daily_to_npz(nc_file, output_dir):
    print(f"Converting {nc_file} to daily .npz files in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    ds = nc.Dataset(nc_file)
    time_var = ds.variables["time"]
    times = nc.num2date(time_var[:], time_var.units)
    var_name = [v for v in ds.variables if v not in ("time", "latitude", "longitude")][0]
    data = ds.variables[var_name][:]
    for i, t in enumerate(times):
        out_fn = os.path.join(output_dir, f"{t.strftime('%Y%m%d')}.npz")
        np.savez_compressed(out_fn, data=data[i])
    ds.close()

def transfer_to_lumi(local_dir, lumi_path):
    print(f"Transferring {local_dir} to LUMI...")
    subprocess.run([
        "scp", "-r", local_dir,
        f"{LUMI_USER}@{LUMI_HOST}:{lumi_path}"
    ], check=True)

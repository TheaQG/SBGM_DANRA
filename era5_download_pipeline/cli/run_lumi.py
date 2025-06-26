'''
    Run the lumi pipeline to process ERA5 data into daily statistics and regridded NPZ files.
'''
import multiprocessing
import pathlib
import yaml

from era5_download_pipeline.pipeline import cdo_utils

cfg = yaml.safe_load(open("cfg/era5_pipeline.yaml", "r", encoding="utf-8"))

def process_year(args):
    '''
        Process a single year for a specific variable.
        This function performs the following steps:
        1. Convert hourly data to daily statistics.
        2. Regrid the daily data to the DANRA grid.
        3. Split the regridded data into NPZ files.
    '''
    vshort, yr = args
    raw_nc = pathlib.Path(cfg['lumi']['raw_dir'].format(var=vshort)) / f"{vshort}_{yr}.nc"
    daily_nc = pathlib.Path(cfg['lumi']['daily_dir'].format(var=vshort)) / f"{vshort}_{yr}_daily.nc"
    npz_dir = pathlib.Path(cfg['lumi']['npz_dir'].format(var=vshort)) / str(yr)

    # 1. day-stats
    cdo_utils.convert_to_daily_stat(raw_nc, cfg['variables_by_short']['vshort']['daily_stat'], daily_nc)
    raw_nc.unlink()  # Remove the hourly raw file after processing

    # 2. regrid 
    if not pathlib.Path(cfg['weights_file']).exists():
        cdo_utils.generate_regridding_weights(daily_nc, cfg['weights_file'], cfg['grid_file'])
    rg_nc = daily_nc.with_suffix("_DG.nc")
    cdo_utils.regrid_to_danra(daily_nc, rg_nc, cfg['weights_file'], cfg['grid_file'])
    daily_nc.unlink()  # Remove the daily file after regridding

    # 2. split to npz
    cdo_utils.convert_daily_to_npz(rg_nc, npz_dir, yr, vshort)
    rg_nc.unlink()  # Remove the regridded file after conversion




if __name__ == "__main__":
    # build a reverse lookup once
    cfg['variables_by_short'] = {v['short']: v for v in cfg['variables'].values()}
    todo = [(v['short'], yr) for v in cfg['variables'].values()
            for yr in range(cfg['years'][0], cfg['years'][1] + 1)]

    with multiprocessing.Pool(processes=cfg.get('n_workers', 4)) as pool:
        pool.map(process_year, todo)

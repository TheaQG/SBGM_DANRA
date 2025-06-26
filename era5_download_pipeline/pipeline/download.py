'''
    Download ERA5 data for specified variables and years using the CDS API.
    This module provides functionality to download ERA5 reanalysis data in parallel
'''

import concurrent.futures
import pathlib

import cdsapi
from .utils import ensure_dir, hours, months, days

def download_year(var_long:str,
                  year:int,
                  out_nc: pathlib.Path,
                  cfg):
    """ Download ERA5 data for a specific variable and year.
    """
    # Set up the CDS API client and request parameters
    c = cdsapi.Client()
    req = dict(product_type="reanalysis",
               variable=var_long,
               year=str(year),
               month=months(),
               day=days(),
               time=hours(),
               area=cfg['area'],
               format=cfg['format'],
    )

    # Ensure the output directory exists
    ensure_dir(out_nc.parent)

    # Retrieve the data from the CDS API
    c.retrieve(cfg['dataset'],
               req,
               out_nc.as_posix() # Ensure the output path is a string for the API call
               )

def pull_all(cfg):
    '''
        Pull all ERA5 data for the specified variables and years in parallel.
    '''
    jobs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['max_workers']) as executor:
        for var_long, vinfo in cfg['variables'].items():
            for year in range(cfg['years'][0], cfg['years'][1] + 1):
                out_nc = pathlib.Path(cfg['tmp_dir']) / vinfo['short'] / f"{vinfo['short']}_{year}.nc"
                job = executor.submit(download_year, var_long, year, out_nc, cfg)
                jobs.append(job)
        for j in jobs:
            j.result() # Propagate exceptions if any (this will block until all jobs are done)

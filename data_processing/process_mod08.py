import sys
sys.path.insert(0, '/home/disk/p/jkcm/Code')

import numpy as np
import glob
import xarray as xr
import os
import matplotlib.pyplot as plt
import datetime as dt
from tools.LoopTimer import LoopTimer


def chop_dataset(ds, list_to_keep):
    ds = ds.drop([x for x in ds.data_vars.keys() if x not in list(set([j.split(':')[0] for i in vars_to_keep for j in ds[i].dims]))+vars_to_keep])
    rename_dict = {k: k.split(':')[0] for k in list(set([j for i in vars_to_keep for j in ds[i].dims]))}
    ds = ds.rename(rename_dict)
    ds = ds.rename({"YDim": "lat", "XDim": "lon"})
    ds = ds.set_coords([i for i in rename_dict.values() if i not in ['XDim', 'YDim']] + ['lat', 'lon'])
    return ds


if __name__ == "__main__":
    vars_to_keep = ['Cloud_Fraction_Day_JHisto_vs_Pressure', 'Cloud_Fraction_Day_Mean', 'Cloud_Fraction_Day_Pixel_Counts']
    modis_files = sorted(glob.glob(r'/home/disk/eos9/jkcm/Data/modis/MYD08/MYD08_D3.A20*.061.*.hdf'))
    modis_date_strs = [i.split('MYD08_D3.A')[1][:7] for i in modis_files]
    modis_dates = [dt.datetime.strptime(i, "%Y%j") + dt.timedelta(hours=12) for i in modis_date_strs]
    lt = LoopTimer(3*12)
    for year in [2014, 2015, 2016]:
        for month in np.arange(1,13):
            lt.update()
            (files, dates) = zip(*[(f, i) for (f,i) in zip(modis_files, modis_dates) if (i.year==year and i.month==month)])
            data = xr.open_mfdataset(files, concat_dim='time', #compat='override', coords='minimal',
                                           preprocess=lambda x: chop_dataset(x, vars_to_keep))
            data['time'] = (('time'), list(dates))                
            savename = f'/home/disk/eos9/jkcm/Data/modis/MYD08/nc/MYD08_D3.061.{year}-{month:02}.cloudfrac_subset.nc'
            comp = dict(zlib=True, complevel=2)
            data.to_netcdf(savename, engine='h5netcdf', encoding={var: comp for var in data.data_vars})

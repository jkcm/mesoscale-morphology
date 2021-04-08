import xarray as xr
import numpy as np
from itertools import product
from functools import reduce
from tools.LoopTimer import LoopTimer
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys

def nan_correlate(x,y):
    idx = np.logical_and(~np.isnan(x), ~np.isnan(y))
    return np.corrcoef(x[idx],y[idx])[0][1]


def quick_regrid(in_arr, reg_arr):
    """Fast way to regrid all values in in_arr onto reg_arr.
        Requires that reg_arr is regularly space. and we'll check, so don't screw around"""
    spacing = np.diff(reg_arr)
    if not (np.allclose(spacing.astype(float), spacing.astype(float)[0])):
        print(np.unique(spacing))
        raise ValueError('not equally spacing, cannot quick regrid')
    spacing = spacing[0]
    return (np.round(in_arr.astype(float)/spacing.astype(float))*spacing.astype(float)).astype(in_arr[0].dtype)
    


def process_data(daily_data, var, dims=None):
    if not dims:
        dims = daily_data.dims
    for di, dim in enumerate(dims):
        savename = (f'/home/disk/eos4/jkcm/Data/MEASURES/correlations/lag_correlations.{var}.{dim}.nc')
        remaining_dims = [i for i in daily_data.dims if not i==dim]
        dim_axis = daily_data.dims.index(dim)
        other_axes = [i for i in np.arange(len(daily_data.dims)) if not i==dim_axis]
        ax_iters = [daily_data[i].values for i in remaining_dims]
        empties = np.full([daily_data.shape[i] for i in other_axes], np.nan)
        all_lags = np.arange(1,10)
        end_result = daily_data.loc[{dim: daily_data[dim].values[0]}].copy(data=empties).expand_dims(lag=all_lags).copy()
        lt = LoopTimer(reduce((lambda x, y: x * y), end_result.shape))
        print(f'working on loop {di}/{len(dims)}: {dim}')
        for lag in all_lags:
            a = daily_data.isel({dim:slice(lag,None,lag)})
            a_shift = daily_data.isel({dim:slice(None,-lag,lag)})
            for i in product(*[l for l in ax_iters]):
                lt.update()
                x = {rd: n for rd, n in zip(remaining_dims,i)}
                a_sl = a.sel(x).values
                a_shift_sl = a_shift.sel(x).values
                corr = nan_correlate(a_sl, a_shift_sl)
                x['lag'] = lag
                end_result.loc[x] = corr
        end_result.to_netcdf(savename)
            

def get_dataset(var):
    if var in ['EIS', 'SST', 'RH_700', 'sfc_div', 'div_700', 'WSPD_10M', 'LTS']:
        MERRA_data = xr.open_dataset(r'/home/disk/eos4/jkcm/Data/MERRA/measures/MERRA_unified_subset_SEP.mon_anom.nc')
        dataset = MERRA_data[var].isel(time=slice(None,None,8))
    elif var == 'vapor':
        amsr_data = xr.open_dataset(r'/home/disk/eos9/jkcm/Data/amsr/rss/all/amsr_unified.subset.mon_anom.nc')
        dataset = amsr_data[var].isel(orbit_segment=0, latitude=slice(None,None,4), longitude=slice(None,None,4))
    elif var in ['net_cre', 'cldarea_low_1h']:
        ceres_data = xr.open_dataset(r'/home/disk/eos9/jkcm/Data/ceres/proc/CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.subset.mon_anom.nc')
        dataset = ceres_data[var].isel(time=slice(19,None,24))
    elif var == 'ascat_div':
        raise NotImplementedError()
#         ascat_data =xr.open_dataset(r'/home/disk/eos9/jkcm/Data/ascat/rss/proc/ascat_unified.anomfromyseas.nc')
    else:
        raise ValueError('variable not recognized')        
    return dataset
    

if __name__ == '__main__':
    
    if not len(sys.argv) >= 2:
        raise ValueError('hey bro gimme a variable')
    var = sys.argv[1]    
    if len(sys.argv) == 3:
        dims=sys.argv[2].split(',')
    else:
        dims=None
    dataset = get_dataset(var)
    process_data(dataset, var, dims) # daily starting at Noon


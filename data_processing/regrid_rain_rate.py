import sys
import os
import glob
import numpy as np
import re
import xarray as xr
import datetime as dt
from itertools import product
from multiprocessing import Pool, cpu_count, current_process
from classified_cset.utils import LoopTimer 


class Groupby:
    # note: adapted from https://github.com/esantorella/hdfe. MIT license required upon publication
    def __init__(self, keys):
        self.keys, self.keys_as_int = np.unique(keys, return_inverse = True)
        self.n_keys = max(self.keys_as_int) + 1
        self.set_indices()
        
    def set_indices(self):
        self.indices = [[] for i in range(self.n_keys)]
        for i, k in enumerate(self.keys_as_int):
            self.indices[k].append(i)
        self.indices = [np.array(elt) for elt in self.indices]
        
    def apply(self, function, vector, broadcast=False):
        if broadcast:
            result = np.zeros(len(vector))
            for idx in self.indices:
                result[idx] = function(vector[idx])
        else:
            result = np.zeros(self.n_keys)
            for k, idx in enumerate(self.indices):
                result[k] = function(vector[idx])
        return result
    

def read_file(f):
    data = xr.open_dataset(f)
    year = data.time_vars.isel(yr_day_utc=0).values
    day = data.time_vars.isel(yr_day_utc=1).values
    utc = data.time_vars.isel(yr_day_utc=2).values
    total_secs = (utc*3600)
    secs = total_secs//1
    msecs = 1000*total_secs%1
    dtime = np.datetime64(f'{np.median(year):0.0f}-01-01')+np.timedelta64(1, 'D')*(day-1)+np.timedelta64(1, 's')*(secs)+np.timedelta64(1, 'ms')*(msecs)
    data['datetime']= (data.longitude.dims, np.broadcast_to(dtime, data.longitude.shape))
    data = data.drop(labels=['time_vars'])
    data['longitude'] = data['longitude']%360
    return data

def load_test_data():
    testfile = '/home/disk/eos5/rmeast/rain_rates_89/2015/AMSR2_89GHz_pcp_est_2015_206_day.nc'
    data = read_file(testfile)
    return data

def make_gridded_dataset(data, res=0.25):
    """
    Big ugly function to make a lat/lon gridded netcdf out L2 AMSR precip retrievals.
    In lieu of proper docstrings, because if you're reading this I forgot to polish this before sharing,
    I'll explain the gist of what's happening. 
    
    Real simple, we take our data, smoosh it so that each obs falls at the nearest lat/lon point on our regular grid,
    group the data by which grid box it falls in, and calculate the relevant stats of the distribution of obs in 
    each grid box. Stats are then returned as an xarray dataset. 
    """ 
    def round_nearest(arr, res):
        nans = np.isnan(arr)
        ret = (((arr+res/2)/res)//1)*res
        ret[nans] = np.nan
        return ret
    
    def reshape_incomplete_array(complete_idx, incomplete_idx, vals, shape):
        new_vals = np.full_like(complete_idx, fill_value=np.nan)
        for idx, val in zip(incomplete_idx, vals):
            new_vals[idx] = val
        return new_vals.reshape(shape)
    
    rain_stats_dict = {0: {'name': 'rain_prob',
                       'long_name': 'Probability of Rain',
                       'standard_name': 'rain_probability',
                       'units': '0-1'},
                   1: {'name': 'rain_rate',
                       'long_name': 'Rain Rate',
                       'standard_name': 'rain_rate',
                       'units': 'mm hr^-1'},
                   2: {'name': 'rain_rwr',
                       'long_name': 'Rain Rate While Raining',
                       'standard_name': 'conditional_rain_rate',
                       'units': 'mm hr^-1'},
                   3: {'name': 'rain_max',
                       'long_name': 'Max Rain Rate',
                       'standard_name': 'max_rain_rate',
                       'units': 'mm hr^-1'}}
    
    func_dict = {'mean': np.nanmean,
             'median': np.nanmedian,
             '25_pctile': lambda x: np.nanpercentile(x, 25),
             '75_pctile': lambda x: np.nanpercentile(x, 75),
             'min': np.nanmin,
             'max': np.nanmax}
    
    
    
    if not 1/res == int(1/res):
        raise ValueError("I haven't gone through to test whether this will work for any resolution that's not a unit fraction.")
    
    #setting up new grid and gridbox index
    grid_lats = np.arange(-90, 90, res)
    grid_lons = np.arange(0, 360, res)
    grid_coords = np.array(list(product(grid_lats, grid_lons)))
    full_grid_lats = grid_coords[:,0]
    full_grid_lons = grid_coords[:,1]
    grid_coords_lats_idx = (full_grid_lats+90)/res
    grid_coords_lons_idx = full_grid_lons/res
    grid_combined_idx = (360/res)*grid_coords_lats_idx + grid_coords_lons_idx
    assert(len(np.unique(grid_combined_idx)) == len(grid_combined_idx))

    #setting up old data unique index
    old_lats = data.latitude.values.flatten()
    old_lons = data.longitude.values.flatten()
    good_filt = np.logical_and(~np.isnan(old_lats), ~np.isnan(old_lons))
    old_lats, old_lons = old_lats[good_filt], old_lons[good_filt]
    lats_regrid = round_nearest(old_lats, res)
    lons_regrid = round_nearest(old_lons, res)%360
    lats_regrid_idx = (lats_regrid+90)/res
    lons_regrid_idx = lons_regrid/res
    unique_combined_idx = (360/res)*lats_regrid_idx + lons_regrid_idx
    assert(set(unique_combined_idx).issubset(grid_combined_idx))
    
    #grouping old data by box
    grouped = Groupby(unique_combined_idx.astype(int))
    
    def new_reshape(vals):
        """Reshapes value from groupby operation to an unfilled lat/lon grid"""
        return reshape_incomplete_array(grid_combined_idx, grouped.keys, vals, shape=(len(grid_lats), len(grid_lons)))
    
    ds = xr.Dataset()
    ds['latitude'] = grid_lats
    ds['longitude'] = grid_lons
    
    ds.attrs['comments'] = "gridded netcdf created by jkcm@uw.edu, adapted from R Eastman AMSR 89 GHz retrievals. " +\
                           "https://doi.org/10.1175/JTECH-D-18-0185.1"
    ds.attrs['creation date'] = str(dt.datetime.utcnow())
    ds.attrs['resolution'] = f'{str(res)} deg'
    
    ds['obs_count'] = (('latitude', 'longitude'), new_reshape(grouped.apply(len, np.empty_like(unique_combined_idx))))
    ds['not_nan_count'] = (('latitude', 'longitude'), new_reshape(grouped.apply(
        lambda x: sum(~np.isnan(x)), np.empty_like(unique_combined_idx))))
    ds['time'] = (('latitude', 'longitude'), new_reshape(grouped.apply(
        lambda x: np.nanmean(x.astype('int64')).astype('datetime64[ns]'), data['datetime'].values.flatten()[good_filt])))
    
    for k, v in rain_stats_dict.items():
        print('working on '+v['name'])
        sys.stdout.flush()
        old_data = data.rain_stats.isel(prob_rate_rwr_max=k).values.flatten()[good_filt]
        for func_name, func in func_dict.items():
            new_vals = new_reshape(grouped.apply(func, old_data))
            new_dict = {'long_name': f"{v['long_name']}_{func_name}",
                        'standard_name': f"{v['standard_name']}_{func_name}",
                        'units': v['units']}
            ds[f"{v['name']}_{func_name}"] = (('latitude', 'longitude'), new_vals, new_dict)
#             print(f"{v['name']}_{func_name}")
            sys.stdout.flush()
    
    print('finishing one')
    sys.stdout.flush()
    return ds


def process_file(f):
    print(os.path.basename(f))
    date = dt.datetime.strptime(os.path.basename(f)[20:28], '%Y_%j')
    data = read_file(f)
    ds = make_gridded_dataset(data, res=0.25)
    ds = ds.expand_dims({'date': [date]})
    save_name = os.path.join(r'/home/disk/eos9/jkcm/Data/rain/2016', os.path.basename(f)[:-3]+'_gridded.nc')
    print(f'saving {save_name}...')
    sys.stdout.flush()
    comp = dict(zlib=True, complevel=2)
    ds.to_netcdf(save_name, engine='h5netcdf', encoding={var: comp for var in ds.data_vars})

if __name__ == "__main__":
    files_2014 = glob.glob('/home/disk/eos5/rmeast/rain_rates_89/2014/AMSR2_89GHz_pcp_est_2014_*_day.nc')
    files_2015 = glob.glob('/home/disk/eos5/rmeast/rain_rates_89/2015/AMSR2_89GHz_pcp_est_2015_*_day.nc')
    files_2016 = glob.glob('/home/disk/eos5/rmeast/rain_rates_89/2016/AMSR2_89GHz_pcp_est_2016_*_day.nc')
    # done_2014 = [os.path.basename(i)[:-11] for i in glob.glob('/home/disk/eos9/jkcm/Data/rain/2014/AMSR2_89*.nc')]
    # done_2015 = [os.path.basename(i)[:-11] for i in glob.glob('/home/disk/eos9/jkcm/Data/rain/2015/AMSR2_89*.nc')]
    # not_done_2014 = [i for i in files_2014 if os.path.basename(i)[:-3] not in done_2014]
    # not_done_2015 = [i for i in files_2015 if os.path.basename(i)[:-3] not in done_2015]
    
    with Pool(16) as p:
        p.map(process_file, files_2016)
        #doing 2015
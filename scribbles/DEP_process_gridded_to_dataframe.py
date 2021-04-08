# Theirs
import numpy as np
import pickle
import sys
from functools import partial
import time
import os
import pandas as pd
import xarray as xr
from multiprocessing import Pool, cpu_count, current_process

# Mine
import utils

"""Script for processing the large gridded datasets (MERRA, ASCAT, CERES, AMSR) into the dataframe of classifications."""


def minimal_add_scat_wind_divergence(scat_data, lat, lon, time, realtime):
    xy = scat_data.sel(latitude=slice(lat-0.5, lat+0.5), longitude=slice(lon-0.5, lon+0.5))
    if not xy.windspd.ndim == 2:
        raise ValueError('hey boss I thought this was gonna only be 2-dimensional.')
    times = ((xy.time.values+(xy.mingmt.values*60*1e3).astype('timedelta64[ms]'))-realtime.to_datetime64())/np.timedelta64(1, 'h')
    return np.nanmean(xy.div.values), np.nanstd(xy.div.values), np.nanmean(times)
    
    
def add_amsr_cwv(amsr_data, lat, lon, time):
    xy = amsr_data.sel(latitude=slice(lat-0.5, lat+0.5), longitude=slice(lon-0.5, lon+0.5))
    big = amsr_data.sel(latitude=slice(lat-5, lat+5), longitude=slice(lon-5, lon+5))
    if xy.ndim > 2:
        raise ValueError('hey boss I thought this was gonna only be 2-dimensional.')        
    return np.nanmean(xy.values), np.nanmean(big.values)

def add_amsr_cwv_climo(amsr_data, lat, lon, time):
    xy = amsr_data.sel(latitude=slice(lat-0.5, lat+0.5), longitude=slice(lon-0.5, lon+0.5))
#     big = amsr_data.sel(latitude=slice(lat-5, lat+5), longitude=slice(lon-5, lon+5))
    if xy.ndim > 2:
        raise ValueError('hey boss I thought this was gonna only be 2-dimensional.')        
    return np.nanmean(xy.values)


    
def add_ceres(ceres_data, lat, lon, time):
    #TODO insert check between ceres data and time for max mismatch
    xy = ceres_data.sel(lat=slice(lat-0.5, lat+0.5), lon=slice(lon-0.5, lon+0.5))
    lw_crf = xy.adj_atmos_lw_up_clr_toa_1h - xy.adj_atmos_lw_up_all_toa_1h
    sw_crf = xy.adj_atmos_sw_up_clr_toa_1h - xy.adj_atmos_sw_up_all_toa_1h
    net_crf = lw_crf+sw_crf
    return np.nanmean(lw_crf), np.nanmean(sw_crf), np.nanmean(net_crf)


'lw_cre = adj_atmos_lw_up_clr_toa_1h-adj_atmos_lw_up_all_toa_1h;sw_cre = adj_atmos_sw_up_clr_toa_1h-adj_atmos_sw_up_all_toa_1h;net_cre = lw_crf+sw_cre;'
    
    
def add_MERRA_var(MERRA_data, varname, lat, lon, time, lev=None, pm=0.5, seas=False):
    xy = MERRA_data.sel(lat=slice(lat-pm, lat+pm), lon=slice(lon-pm, lon+pm))
    if lev:
        xy = xy.sel(lev=lev)
    xy = xy.sel(time=time, method='nearest')
    if (xy.time-time.to_datetime64())/np.timedelta64(1, 'h') > 6 and not seas:
        print(xy.time, time)
        raise ValueError('bad time bro')
    if not xy[varname].ndim == 2:
        raise ValueError('hey boss I thought this was gonna only be 2-dimensional.')
#     try:
#         if sum(~np.isnan(xy[varname].values))==0:
#             print(lat, lon, time, varname)
#             print(xy[varname].shape)
#     except ValueError as e:
#         print(xy[varname].values.shape)
    return np.nanmean(xy[varname].values) #merra_var
    
    
def add_MERRA_wspd(MERRA_data, lat, lon, time, pm=0.5, seas=False):
    xy = MERRA_data.sel(lat=slice(lat-pm, lat+pm), lon=slice(lon-pm, lon+pm))
    xy = xy.sel(time=time, method='nearest')
    if (xy.time-time.to_datetime64())/np.timedelta64(1, 'h') > 6 and not seas:
        print(xy.time, time)
        raise ValueError('bad time bro')
    if not xy["U10M"].ndim == 2:
        raise ValueError('hey boss I thought this was gonna only be 2-dimensional.')
    u10 = xy["U10M"].values
    v10 = xy["V10M"].values
    return np.nanmean(np.sqrt(u10**2+v10**2))
    
    
def add_divergence(df, dayshift=0, MERRA=False, AMSR=True, CERES=False):
    #add in ASCAT divergence to the dataframe
       #"descending" branch, segment 0, = "nighttime", but it's the 9:30am one. About 4-5 hours off MODIS time
    #lats and lons are centered in middle of scene, so slice +-1/2 degree
    if not len(np.unique(df.day)) == 1:
        raise ValueError('hey boss I thought we were gonna take it one day at a time')
    dates = sorted(df.datetime)
    date = dates[len(dates)//2]
    ts0 = time.time()
    ASCAT=True
    if ASCAT:
        print(f'{current_process().name}: working on {date}')
        sys.stdout.flush()
        print(f'{current_process().name}: working on ASCAT')
        ts = time.time()
        sys.stdout.flush()
        date_adj = date-np.timedelta64(5, 'h')+np.timedelta64(dayshift, 'D') #diff between MODIS and ASCAT overpass times
        all_scat_data = xr.open_dataset(
            f'/home/disk/eos9/jkcm/Data/ascat/rss/all/ascat_unified_{date_adj.year}-{date_adj.month:02}.nc')
        scat_data = all_scat_data.sel(time=date_adj.date(), method='nearest', orbit_segment=0)
        scat_data = utils.get_ascat_divergence(scat_data)
        df['ascat_div_all'] = df.apply(lambda x: minimal_add_scat_wind_divergence(
            scat_data, x['lat'], x['lon'], x['datetime'], realtime=date), axis=1)
        if dayshift == 0:
            df['ascat_div'] = df.apply(lambda x: x['ascat_div_all'][0], axis=1)
            df['ascat_div_std'] = df.apply(lambda x: x['ascat_div_all'][1], axis=1)
            df['ascat_time_offset'] = df.apply(lambda x: x['ascat_div_all'][2], axis=1)
        else:
            df[f'ascat_div_{dayshift}'] = df.apply(lambda x: x['ascat_div_all'][0], axis=1)
            df[f'ascat_time_offset_{dayshift}'] = df.apply(lambda x: x['ascat_div_all'][2], axis=1)
        df = df.drop(columns='ascat_div_all')
        ascat_climo = xr.open_dataset(
            r'/home/disk/eos9/jkcm/Data/ascat/rss/proc/ascat_unified_seasmean.nc', lock=False)
        ascat_climo = ascat_climo.sel(time=date, method='nearest', orbit_segment=0)  # TODO check
        ascat_climo = utils.get_ascat_divergence(ascat_climo)
        df['ascat_div_clim'] = df.apply(lambda x: minimal_add_scat_wind_divergence(
            ascat_climo, x['lat'], x['lon'], x['datetime'], realtime=date)[0], axis=1)

        tf = time.time()
        tp = f'{(tf-ts)//60}m, {(tf-ts)%60:0.0f}s' 
        print(f'{current_process().name}: done with ASCAT, {tp}')
        sys.stdout.flush() 
    if MERRA:
        print(f'{current_process().name}: working on MERRA')
        ts = time.time()
        sys.stdout.flush()
        if MERRA == 'CSET':
            MERRA_data = xr.open_dataset(
                r'/home/disk/eos4/jkcm/Data/CSET/MERRA/measures/MERRA_unified_subset.nc', lock=False)
        elif MERRA == 'SEP':
            MERRA_data = xr.open_dataset(
                r'/home/disk/eos4/jkcm/Data/MERRA/measures/split/MERRA_unified_subset_SEP.'+\
                f'{date.year}-{date.month:02}.nc',
                lock=False)
        else: 
            raise ValueError('MERRA specification not known, please specify "CSET" or "SEP"')
        df['MERRA_div_sfc'] = df.apply(lambda x: add_MERRA_var(
            MERRA_data, 'sfc_div', x['lat'], x['lon'], x['datetime']), axis=1)
        df['MERRA_SST'] = df.apply(lambda x: add_MERRA_var(
            MERRA_data, 'SST', x['lat'], x['lon'], x['datetime']), axis=1)
        df['MERRA_EIS'] = df.apply(lambda x: add_MERRA_var(
            MERRA_data, 'EIS', x['lat'], x['lon'], x['datetime']), axis=1)
        df['MERRA_LTS'] = df.apply(lambda x: add_MERRA_var(
            MERRA_data, 'LTS', x['lat'], x['lon'], x['datetime']), axis=1)
        df['MERRA_subs_850_s'] = df.apply(lambda x: add_MERRA_var(
            MERRA_data, 'dzdt', x['lat'], x['lon'], x['datetime'], lev=850, pm=0.5), axis=1)
        df['MERRA_z_850_s'] = df.apply(lambda x: add_MERRA_var(
            MERRA_data, 'H', x['lat'], x['lon'], x['datetime'], lev=850, pm=0.5), axis=1)
        df['MERRA_div_850_ss'] = df['MERRA_subs_850_s']/df['MERRA_z_850_s']
        df['MERRA_subs_700'] = df.apply(lambda x: add_MERRA_var(
            MERRA_data, 'dzdt', x['lat'], x['lon'], x['datetime'], lev=700), axis=1)
        df['MERRA_z_700'] = df.apply(lambda x: add_MERRA_var(
            MERRA_data, 'H', x['lat'], x['lon'], x['datetime'], lev=700), axis=1)
        df['MERRA_div_ls'] = df['MERRA_subs_700']/df['MERRA_z_700']
        df['MERRA_subs_700_s'] = df.apply(lambda x: add_MERRA_var(
            MERRA_data, 'dzdt', x['lat'], x['lon'], x['datetime'], lev=700, pm=0.5), axis=1)
        df['MERRA_z_700_s'] = df.apply(lambda x: add_MERRA_var(
            MERRA_data, 'H', x['lat'], x['lon'], x['datetime'], lev=700, pm=0.5), axis=1)
        df['MERRA_div_ss'] = df['MERRA_subs_700_s']/df['MERRA_z_700_s']
#         df['MERRA_ascat_sfc_anomaly'] = df['ascat_div'] - df['MERRA_div_ls']
#         df['MERRA_MERRA_sfc_anomaly'] = df['MERRA_div_sfc'] - df['MERRA_div_ls']        
        MERRA_climo = xr.open_dataset(
        r'/home/disk/eos4/jkcm/Data/MERRA/measures/split/MERRA_unified_subset_SEP.seasmean.nc', lock=False)
        df['MERRA_div_sfc_climo'] = df.apply(lambda x: add_MERRA_var(
            MERRA_climo, 'sfc_div', x['lat'], x['lon'], x['datetime'], seas=True), axis=1)     
        df['MERRA_subs_700_climo'] = df.apply(lambda x: add_MERRA_var(
            MERRA_climo, 'dzdt', x['lat'], x['lon'], x['datetime'], lev=700, seas=True), axis=1)
        df['MERRA_z_700_climo'] = df.apply(lambda x: add_MERRA_var(
            MERRA_climo, 'H', x['lat'], x['lon'], x['datetime'], lev=700, seas=True), axis=1)
        df['MERRA_div_ls_climo'] = df['MERRA_subs_700_climo']/df['MERRA_z_700_climo'] 
        tf = time.time()
        tp = f'{(tf-ts)//60}m, {(tf-ts)%60:0.0f}s' 
        print(f'{current_process().name}: done with MERRA, {tp}')
        sys.stdout.flush()
    if AMSR:
        print(f'{current_process().name}: working on AMSR')
        ts = time.time()
        sys.stdout.flush()
        if AMSR == 'CSET':
            amsr_file = r'/home/disk/eos4/jkcm/Data/CSET/amsr/AMSR2_CWV_CSET_fixed.nc'
            amsr_data = xr.open_dataset(amsr_file, lock=False)        
            amsr_cwv = amsr_data.CWV.sel(time=date_adj.date(), method='nearest')
        elif AMSR == 'SEP':
            amsr_file = f'/home/disk/eos9/jkcm/Data/amsr/rss/all/amsr_unified_{date_adj.year}-{date_adj.month:02}.nc'
            amsr_data = xr.open_dataset(amsr_file, lock=False)        
            amsr_cwv = amsr_data.vapor.sel(
                time=date_adj.date(), orbit_segment=0, method='nearest')
        df['amsr_all'] = df.apply(lambda x: add_amsr_cwv(amsr_cwv, x['lat'], x['lon'], x['datetime']), axis=1)   
        df['amsr_cwv'] = df.apply(lambda x: x['amsr_all'][0], axis=1)
        df['amsr_cwv_region'] = df.apply(lambda x: x['amsr_all'][1], axis=1)
        df['amsr_cwv_anom'] = df['amsr_cwv'] - df['amsr_cwv_region']
        df = df.drop(columns='amsr_all')
        tf = time.time()
        tp = f'{(tf-ts)//60}m, {(tf-ts)%60:0.0f}s' 
        print(f'{current_process().name}: done with AMSR, {tp}')
        sys.stdout.flush()
    if CERES:
        print(f'{current_process().name}: working on CERES')
        ts = time.time()
        sys.stdout.flush()
        ceres_file = os.path.join('/home/disk/eos9/jkcm/Data/ceres/proc/',
                                  f'CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.{date_adj.year}-{date_adj.month:02}.nc')
        ceres_data = xr.open_dataset(ceres_file, lock=False)
        ceres_data = ceres_data.sel(time=date, method='nearest', tolerance=np.timedelta64(1, 'h'))
        df['ceres_all'] = df.apply(lambda x: add_ceres(ceres_data, x['lat'], x['lon'], x['datetime']), axis=1)   
        df['ceres_lw_crf'] = df.apply(lambda x: x['ceres_all'][0], axis=1)
        df['ceres_sw_crf'] = df.apply(lambda x: x['ceres_all'][1], axis=1)
        df['ceres_net_crf'] = df.apply(lambda x: x['ceres_all'][2], axis=1)
        df = df.drop(columns='ceres_all')
        tf = time.time()
        tp = f'{(tf-ts)//60}m, {(tf-ts)%60:0.0f}s' 
        print(f'{current_process().name}: done with CERES, {tp}')
        sys.stdout.flush()

    tf0 = time.time()
    tp = f'{(tf0-ts0)//60}m, {(tf0-ts0)%60:0.0f}s' 
    print(f'{current_process().name}: done with {date}, {tp}')
    sys.stdout.flush()
    return df



def add_some_stuff(df):
    if not len(np.unique(df.day)) == 1:
        raise ValueError('hey boss I thought we were gonna take it one day at a time')
    dates = sorted(df.datetime)
    date = dates[len(dates)//2]
    ts0 = time.time()
    MERRA='SEP'
    if MERRA:
        print(f'{current_process().name}: working on MERRA')
        ts = time.time()
        sys.stdout.flush()
        if MERRA == 'CSET':
            MERRA_data = xr.open_dataset(
                r'/home/disk/eos4/jkcm/Data/CSET/MERRA/measures/MERRA_unified_subset.nc', lock=False)
        elif MERRA == 'SEP':
            MERRA_data = xr.open_dataset(
                r'/home/disk/eos4/jkcm/Data/MERRA/measures/split/MERRA_unified_subset_SEP.'+\
                f'{date.year}-{date.month:02}.nc',
                lock=False)
        else: 
            raise ValueError('MERRA specification not known, please specify "CSET" or "SEP"')
        df['MERRA_div_sfc_region'] = df.apply(lambda x: add_MERRA_var(
            MERRA_data, 'sfc_div', x['lat'], x['lon'], x['datetime'], pm=5), axis=1)
        df['MERRA_SST_region'] = df.apply(lambda x: add_MERRA_var(
            MERRA_data, 'SST', x['lat'], x['lon'], x['datetime'], pm=5), axis=1)
        df['MERRA_EIS_region'] = df.apply(lambda x: add_MERRA_var(
            MERRA_data, 'EIS', x['lat'], x['lon'], x['datetime'], pm=5), axis=1)
        tf = time.time()
        tp = f'{(tf-ts)//60}m, {(tf-ts)%60:0.0f}s' 
        print(f'{current_process().name}: done with MERRA, {tp}')
        sys.stdout.flush()
    
    tf0 = time.time()
    tp = f'{(tf0-ts0)//60}m, {(tf0-ts0)%60:0.0f}s' 
    print(f'{current_process().name}: done with {date}, {tp}')
    sys.stdout.flush()
    return df


def add_more_stuff(df):
    if not len(np.unique(df.day)) == 1:
        raise ValueError('hey boss I thought we were gonna take it one day at a time')
    dates = sorted(df.datetime)
    date = dates[len(dates)//2]
    ts0 = time.time()
    MERRA='SEP'
    if MERRA:
        print(f'{current_process().name}: working on MERRA')
        ts = time.time()
        sys.stdout.flush()
        if MERRA == 'CSET':
            MERRA_data = xr.open_dataset(
                r'/home/disk/eos4/jkcm/Data/CSET/MERRA/measures/MERRA_unified_subset.nc', lock=False)
        elif MERRA == 'SEP':
            MERRA_data = xr.open_dataset(
                r'/home/disk/eos4/jkcm/Data/MERRA/measures/split/MERRA_unified_subset_SEP.'+\
                f'{date.year}-{date.month:02}.nc',
                lock=False)
            MERRA_climo = xr.open_dataset(
                r'/home/disk/eos4/jkcm/Data/MERRA/measures/split/MERRA_unified_subset_SEP.seasmean.nc', lock=False)
        else: 
            raise ValueError('MERRA specification not known, please specify "CSET" or "SEP"')
            
        
#         df['MERRA_div_sfc_region'] = df.apply(lambda x: add_MERRA_var(
#             MERRA_data, 'sfc_div', x['lat'], x['lon'], x['datetime'], pm=5), axis=1)
#         df['MERRA_SST_region'] = df.apply(lambda x: add_MERRA_var(
#             MERRA_data, 'SST', x['lat'], x['lon'], x['datetime'], pm=5), axis=1)
#         df['MERRA_EIS_region'] = df.apply(lambda x: add_MERRA_var(
#             MERRA_data, 'EIS', x['lat'], x['lon'], x['datetime'], pm=5), axis=1)
        
        df['MERRA_wpsd'] = df.apply(lambda x: add_MERRA_wspd(
            MERRA_data, x['lat'], x['lon'], x['datetime']), axis=1)
        df['MERRA_wpsd_region'] =  df.apply(lambda x: add_MERRA_wspd(
            MERRA_data, x['lat'], x['lon'], x['datetime'], pm=5), axis=1)
        
        df['MERRA_EIS_climo'] = df.apply(lambda x: add_MERRA_var(
            MERRA_climo, 'EIS', x['lat'], x['lon'], x['datetime'], seas=True), axis=1)     
        df['MERRA_SST_climo'] = df.apply(lambda x: add_MERRA_var(
            MERRA_climo, 'SST', x['lat'], x['lon'], x['datetime'], seas=True), axis=1)     
        df['MERRA_wpsd_climo'] = df.apply(lambda x: add_MERRA_wspd(
            MERRA_climo, x['lat'], x['lon'], x['datetime'], seas=True), axis=1)     
    
        tf = time.time()
        tp = f'{(tf-ts)//60}m, {(tf-ts)%60:0.0f}s' 
        print(f'{current_process().name}: done with MERRA, {tp}')
        sys.stdout.flush()
    
    tf0 = time.time()
    tp = f'{(tf0-ts0)//60}m, {(tf0-ts0)%60:0.0f}s' 
    print(f'{current_process().name}: done with {date}, {tp}')
    sys.stdout.flush()
    return df
    
    

    
    
    
    
def add_even_more_stuff(df):
    if not len(np.unique(df.day)) == 1:
        raise ValueError('hey boss I thought we were gonna take it one day at a time')
    dates = sorted(df.datetime)
    date = dates[len(dates)//2]
    ts0 = time.time()
    MERRA='SEP'
    if MERRA:
        print(f'{current_process().name}: working on MERRA')
        ts = time.time()
        sys.stdout.flush()
        if MERRA == 'CSET':
            pass
        elif MERRA == 'SEP':
            MERRA_data = xr.open_dataset(
                r'/home/disk/eos4/jkcm/Data/MERRA/measures/split/MERRA_unified_subset_SEP.'+\
                f'{date.year}-{date.month:02}.nc',
                lock=False)
            MERRA_climo = xr.open_dataset(
                r'/home/disk/eos4/jkcm/Data/MERRA/measures/split/MERRA_unified_subset_SEP.seasmean.nc', lock=False)
        else: 
            raise ValueError('MERRA specification not known, please specify "CSET" or "SEP"')

        MERRA_z_700_region = df.apply(lambda x: add_MERRA_var(
            MERRA_data, 'H', x['lat'], x['lon'], x['datetime'], lev=700, pm=5), axis=1)
        MERRA_subs_700_region = df.apply(lambda x: add_MERRA_var(
            MERRA_data, 'dzdt', x['lat'], x['lon'], x['datetime'], lev=700, pm=5), axis=1)

        df['MERRA_div_ls_region'] = MERRA_subs_700_region/MERRA_z_700_region
        df['MERRA_RH700'] = df.apply(lambda x: add_MERRA_var(
            MERRA_data, 'RH_700', x['lat'], x['lon'], x['datetime']), axis=1)
        df['MERRA_RH700_region'] = df.apply(lambda x: add_MERRA_var(
            MERRA_data, 'RH_700', x['lat'], x['lon'], x['datetime'], pm=5), axis=1)
        df['MERRA_RH700_climo'] = df.apply(lambda x: add_MERRA_var(
            MERRA_climo, 'RH_700', x['lat'], x['lon'], x['datetime'], seas=True), axis=1)

        tf = time.time()
        tp = f'{(tf-ts)//60}m, {(tf-ts)%60:0.0f}s' 
        print(f'{current_process().name}: done with MERRA, {tp}')
        sys.stdout.flush()
    
    
#     AMSR='SEP'
#     if AMSR:
#         print(f'{current_process().name}: working on AMSR')
#         ts = time.time()
#         sys.stdout.flush()
#         if AMSR == 'CSET':
#             pass
# #             amsr_file = r'/home/disk/eos4/jkcm/Data/CSET/amsr/AMSR2_CWV_CSET_fixed.nc'
# #             amsr_data = xr.open_dataset(amsr_file, lock=False)        
# #             amsr_cwv = amsr_data.CWV.sel(time=date_adj.date(), method='nearest')
#         elif AMSR == 'SEP':
# #             amsr_file = f'/home/disk/eos9/jkcm/Data/amsr/rss/all/amsr_unified_{date_adj.year}-{date_adj.month:02}.nc'
# #             amsr_data = xr.open_dataset(amsr_file, lock=False)        
# #             amsr_cwv = amsr_data.vapor.sel(time=date_adj.date(), orbit_segment=0, method='nearest')
            
#             amsr_climo_file = f'/home/disk/eos9/jkcm/Data/amsr/rss/all/amsr_unified.seasmean.nc'
#             amsr_climo_data = xr.open_dataset(amsr_climo_file, lock=False)        
#             amsr_cwv_climo = amsr_climo_data.vapor.sel(
#                 time=date_adj.date(), orbit_segment=0, method='nearest')
            
            
#         df['amsr_cwv_climo'] = df.apply(lambda x: add_amsr_cwv_climo(amsr_cwv_climo, x['lat'], x['lon'], x['datetime']), axis=1)   
#         tf = time.time()
#         tp = f'{(tf-ts)//60}m, {(tf-ts)%60:0.0f}s' 
#         print(f'{current_process().name}: done with AMSR, {tp}')
#         sys.stdout.flush()
    
    
    
    
    
    
    
    
    
    tf0 = time.time()
    tp = f'{(tf0-ts0)//60}m, {(tf0-ts0)%60:0.0f}s' 
    print(f'{current_process().name}: done with {date}, {tp}')
    sys.stdout.flush()
    return df

    
def add_final_fucking_set_of_stuff(df):
    if not len(np.unique(df.day)) == 1:
        raise ValueError('hey boss I thought we were gonna take it one day at a time')
    dates = sorted(df.datetime)
    date = dates[len(dates)//2]
    ts0 = time.time()
    MERRA='SEP'
    if MERRA:
        print(f'{current_process().name}: working on MERRA')
        ts = time.time()
        sys.stdout.flush()
        if MERRA == 'CSET':
            pass
        elif MERRA == 'SEP':
            MERRA_data = xr.open_dataset(
                r'/home/disk/eos4/jkcm/Data/MERRA/measures/split/MERRA_unified_subset_SEP.'+\
                f'{date.year}-{date.month:02}.nc',
                lock=False)
            MERRA_climo = xr.open_dataset(
                r'/home/disk/eos4/jkcm/Data/MERRA/measures/split/MERRA_unified_subset_SEP.seasmean.nc', lock=False)
        else: 
            raise ValueError('MERRA specification not known, please specify "CSET" or "SEP"')

        MERRA_z_700_region = df.apply(lambda x: add_MERRA_var(
            MERRA_data, 'H', x['lat'], x['lon'], x['datetime'], lev=700, pm=5), axis=1)
        MERRA_subs_700_region = df.apply(lambda x: add_MERRA_var(
            MERRA_data, 'dzdt', x['lat'], x['lon'], x['datetime'], lev=700, pm=5), axis=1)

        df['MERRA_div_ls_region'] = MERRA_subs_700_region/MERRA_z_700_region
        df['MERRA_RH700'] = df.apply(lambda x: add_MERRA_var(
            MERRA_data, 'RH_700', x['lat'], x['lon'], x['datetime']), axis=1)
        df['MERRA_RH700_region'] = df.apply(lambda x: add_MERRA_var(
            MERRA_data, 'RH_700', x['lat'], x['lon'], x['datetime'], pm=5), axis=1)
        df['MERRA_RH700_climo'] = df.apply(lambda x: add_MERRA_var(
            MERRA_climo, 'RH_700', x['lat'], x['lon'], x['datetime'], seas=True), axis=1)

        tf = time.time()
        tp = f'{(tf-ts)//60}m, {(tf-ts)%60:0.0f}s' 
        print(f'{current_process().name}: done with MERRA, {tp}')
        sys.stdout.flush()
    
    
#     AMSR='SEP'
#     if AMSR:
#         print(f'{current_process().name}: working on AMSR')
#         ts = time.time()
#         sys.stdout.flush()
#         if AMSR == 'CSET':
#             pass
# #             amsr_file = r'/home/disk/eos4/jkcm/Data/CSET/amsr/AMSR2_CWV_CSET_fixed.nc'
# #             amsr_data = xr.open_dataset(amsr_file, lock=False)        
# #             amsr_cwv = amsr_data.CWV.sel(time=date_adj.date(), method='nearest')
#         elif AMSR == 'SEP':
# #             amsr_file = f'/home/disk/eos9/jkcm/Data/amsr/rss/all/amsr_unified_{date_adj.year}-{date_adj.month:02}.nc'
# #             amsr_data = xr.open_dataset(amsr_file, lock=False)        
# #             amsr_cwv = amsr_data.vapor.sel(time=date_adj.date(), orbit_segment=0, method='nearest')
            
#             amsr_climo_file = f'/home/disk/eos9/jkcm/Data/amsr/rss/all/amsr_unified.seasmean.nc'
#             amsr_climo_data = xr.open_dataset(amsr_climo_file, lock=False)        
#             amsr_cwv_climo = amsr_climo_data.vapor.sel(
#                 time=date_adj.date(), orbit_segment=0, method='nearest')
            
            
#         df['amsr_cwv_climo'] = df.apply(lambda x: add_amsr_cwv_climo(amsr_cwv_climo, x['lat'], x['lon'], x['datetime']), axis=1)   
#         tf = time.time()
#         tp = f'{(tf-ts)//60}m, {(tf-ts)%60:0.0f}s' 
#         print(f'{current_process().name}: done with AMSR, {tp}')
#         sys.stdout.flush()
        
    

def applyParallel(dfGrouped, func):
    #take a grouped pandas dataframe and apply a function over its groups
    with Pool(cpu_count()) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])
    return pd.concat(ret_list)

def applyPartial(dfGrouped, func, args):
    with Pool(min(2, cpu_count())) as p:
        ret_list = p.map(partial(func, **args), [group for name, group in dfGrouped])
    return pd.concat(ret_list)

def applySingle(dfGrouped, func, args):
    ret_list = map(partial(func, **args), [group for name, group in dfGrouped])
    return pd.concat(ret_list)

def process_month(year, month):
    all_class_df = utils.load_class_data('all')
    sep_class_df = all_class_df[all_class_df['loc']=='SEP']


if __name__ == "__main__":
#     class_df = utils.load_class_data('all')
#     class_df = class_df[np.logical_and(class_df.date>'2014-01-03', class_df.date<'2016-12-29')]
#     by_day = class_df.groupby('day')
#     new_df = applyPartial(dfGrouped=by_day, func=add_divergence, args={'dayshift': 0})
#     savefile = f'/home/disk/eos4/jkcm/Data/MEASURES/beta_data/all_with_clim.pickle'
#     pickle.dump(new_df, open(savefile, "wb" ))
    
#     new_df = applyParallel(dfGrouped=by_day, func=add_divergence)
#     savefile = r'/home/disk/eos4/jkcm/Data/MEASURES/beta_data/all_with_div_2.pickle'
#     pickle.dump(new_df, open(savefile, "wb" ) )
#     for nday in [-2, -1, 0, 1, 2]:
#         new_df = applyPartial(dfGrouped=by_day, func=add_divergence, args={'dayshift': nday})
#         savefile = f'/home/disk/eos4/jkcm/Data/MEASURES/beta_data/all_with_div_shift_{nday}.pickle'
#         pickle.dump(new_df, open(savefile, "wb" ) )

#     class_df = utils.load_class_data('cset')
#     by_day = class_df.groupby('day')
#     new_df = applyPartial(dfGrouped=by_day, func=add_divergence, args={'dayshift': 0, 'MERRA': "CSET"})
#     savefile = f'/home/disk/eos4/jkcm/Data/MEASURES/beta_data/cset_with_clim_MERRA.pickle'
#     pickle.dump(new_df, open(savefile, "wb" ) )



    if not len(sys.argv) == 2:
        raise ValueError('no year')
    year = float(sys.argv[1])    
    loadfile = f'/home/disk/eos4/jkcm/Data/MEASURES/beta_data/sep_with_clim_ceres_merra_{int(year)}_modified2.pickle'
    savefile = f'/home/disk/eos4/jkcm/Data/MEASURES/beta_data/sep_with_clim_ceres_merra_{int(year)}_modified3.pickle'

    df = pickle.load(open(loadfile, 'rb'))
    by_day = df.groupby('day')
    print('applying function')
    new_df = applySingle(dfGrouped=by_day, func=add_even_more_stuff, 
                         args={})
    print(savefile)
    pickle.dump(new_df, open(savefile, "wb" ) )   


#     if len(sys.argv) not in [2,3]:
#         raise ValueError('please provide year and month, or just a year')
#     year = float(sys.argv[1])
#     all_class_df = utils.load_class_data('all')
#     sep_class_df = all_class_df[all_class_df['loc']=='SEP']
#     if year not in np.unique(sep_class_df.year):
#         raise ValueError("incorrect input year")
#     year_class_df = sep_class_df[sep_class_df['year']==year]
#     df_to_use = year_class_df
#     savefile = f'/home/disk/eos4/jkcm/Data/MEASURES/beta_data/sep_with_clim_ceres_merra_{int(year)}.pickle'
#     if len(sys.argv) == 3:
#         month = int(sys.argv[2])
#         if month not in np.unique(year_class_df.month):
#             raise ValueErro('incorrect input month')
#         month_class_df = year_class_df[year_class_df['month']==month]
#         df_to_use = month_class_df
#         savefile = f'/home/disk/eos4/jkcm/Data/MEASURES/beta_data/sep_with_clim_{int(year)}_{month:02}.pickle'
#     by_day = df_to_use.groupby('day')
#     print('applying function')
#     new_df = applySingle(dfGrouped=by_day, func=add_divergence, 
#                          args={'dayshift': 0, 'MERRA': "SEP", 'AMSR': "SEP", "CERES": True})
#     print(savefile)
#     pickle.dump(new_df, open(savefile, "wb" ) )
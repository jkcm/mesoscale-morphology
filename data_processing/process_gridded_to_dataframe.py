#!/usr/bin/env conda run -n classified-cset python
# -*- coding: utf-8 -*-
"""Script for processing the large gridded datasets (MERRA, ASCAT, CERES, AMSR) into the dataframe of classifications.
    Takes a while to run...
    Created by Johannes Mohrmann, March 6 2020"""
    

# Theirs
import numpy as np
import pickle
import sys
from functools import partial
import time
import os
import pandas as pd
#Specials
import xarray as xr
from multiprocessing import Pool, cpu_count, current_process
# Mine
from classified_cset import utils
from tools.decorators import timed

####SETUP####
# pandarallel.initialize()


def slice_data(dataset, lat, lon, size):
    if 'lat' in dataset.dims:
        lat_n, lon_n = 'lat', 'lon'
    elif 'latitude' in dataset.dims:
        lat_n, lon_n = 'latitude', 'longitude'
    xy = dataset.sel({lat_n:slice(lat-size, lat+size), lon_n:slice(lon-size, lon+size)})
    if xy.ndim > 2:
        raise ValueError(f'Too many dimensions in data slice. Shape is {xy.shape}, dims are {xy.dims}')
    return xy

def precip_slice(dataset, lat, lon, size):
    ds_lats = dataset.where(np.logical_and(dataset.latitude>(lat-size), dataset.latitude<(lat+size)), drop=True)
    ds_lons = ds_lats.where(np.logical_and((ds_lats.longitude%360)>(lon-size), (ds_lats.longitude%360)<(lon+size)), drop=True)
    #     if ds_lons.ndim > 2:
#         raise ValueError(f'Too many dimensions in data slice. Shape is {ds_lons.shape}, dims are {ds_lons.dims}')
    return ds_lons
    


@timed
def add_CERES_to_df(df, date, bounds):
    ceres_file = os.path.join('/home/disk/eos9/jkcm/Data/ceres/proc/split',
                                f'CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.{date.year}-{date.month:02}.nc')
    ceres_data = xr.open_dataset(ceres_file, lock=False)
    ceres_data = ceres_data.sel(time=date, method='nearest', tolerance=np.timedelta64(2, 'h'))  
    ceres_climo = xr.open_dataset(r'/home/disk/eos9/jkcm/Data/ceres/proc/CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.merged_complete.seasmean.nc', lock=False)
    ceres_climo = ceres_climo.sel(time=date, method='nearest', tolerance=np.timedelta64(2, 'M'))
    
    df['ceres_net_cre'] = df.apply(lambda x: np.nanmean(slice_data(ceres_data.net_cre, x['lat'], x['lon'], size=0.5)), axis=1)
    df['ceres_net_cre_region'] = df.apply(lambda x: np.nanmean(slice_data(ceres_data.net_cre, x['lat'], x['lon'], size=5)), axis=1)
    df['ceres_net_cre_climo'] = df.apply(lambda x: np.nanmean(slice_data(ceres_climo.net_cre, x['lat'], x['lon'], size=0.5)), axis=1)

    df['ceres_low_cf'] = df.apply(lambda x: np.nanmean(slice_data(ceres_data.cldarea_low_1h, x['lat'], x['lon'], size=0.5)), axis=1)
    df['ceres_low_cf_region'] = df.apply(lambda x: np.nanmean(slice_data(ceres_data.cldarea_low_1h, x['lat'], x['lon'], size=5)), axis=1)
    df['ceres_low_cf_climo'] = df.apply(lambda x: np.nanmean(slice_data(ceres_climo.cldarea_low_1h, x['lat'], x['lon'], size=0.5)), axis=1)
    return df


@timed
def add_more_CERES_to_df(df, date):
    ceres_file = os.path.join('/home/disk/eos9/jkcm/Data/ceres/proc/split',
                                f'CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.{date.year}-{date.month:02}.nc')
    ceres_data = xr.open_dataset(ceres_file, lock=False)
    ceres_data = ceres_data.sel(time=date, method='nearest', tolerance=np.timedelta64(2, 'h'))  
    ceres_climo = xr.open_dataset(r'/home/disk/eos9/jkcm/Data/ceres/proc/CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.merged_complete.seasmean.nc', lock=False)
    ceres_climo = ceres_climo.sel(time=date, method='nearest', tolerance=np.timedelta64(2, 'M'))
    
    df['ceres_sw_cre'] = df.apply(lambda x: np.nanmean(slice_data(ceres_data.sw_cre, x['lat'], x['lon'], size=0.5)), axis=1)
    df['ceres_sw_cre_region'] = df.apply(lambda x: np.nanmean(slice_data(ceres_data.sw_cre, x['lat'], x['lon'], size=5)), axis=1)
    df['ceres_sw_cre_climo'] = df.apply(lambda x: np.nanmean(slice_data(ceres_climo.sw_cre, x['lat'], x['lon'], size=0.5)), axis=1)
        
    df['ceres_lw_cre'] = df.apply(lambda x: np.nanmean(slice_data(ceres_data.lw_cre, x['lat'], x['lon'], size=0.5)), axis=1)
    df['ceres_lw_cre_region'] = df.apply(lambda x: np.nanmean(slice_data(ceres_data.lw_cre, x['lat'], x['lon'], size=5)), axis=1)
    df['ceres_lw_cre_climo'] = df.apply(lambda x: np.nanmean(slice_data(ceres_climo.lw_cre, x['lat'], x['lon'], size=0.5)), axis=1)
    return df

    
@timed
def add_precip_to_df(df, date):
#     
    
    try:
        precip_data = xr.open_dataset(
            f'/home/disk/eos9/jkcm/Data/rain/{date.year}/AMSR2_89GHz_pcp_est_{date.year}_{date.dayofyear:03}_day_gridded.nc')
        precip_data = precip_data.sel(date=date, method='nearest', tolerance=np.timedelta64(24, 'h'))
        precip_climo = xr.open_dataset(
            f'/home/disk/eos9/jkcm/Data/rain/AMSR2_89GHz_pcp_est.seasmean.nc', lock=False)
        precip_climo = precip_climo.sel(date=date, method='nearest', tolerance=np.timedelta64(2, 'M'))
    
        df['amsr_tb_rate'] = df.apply(lambda x: np.nanmean(slice_data(precip_data.rain_rate_mean, x['lat'], x['lon'], size=0.5)), axis=1)
        df['amsr_tb_rwr'] = df.apply(lambda x: np.nanmean(slice_data(precip_data.rain_rwr_mean, x['lat'], x['lon'], size=0.5)), axis=1)
        df['amsr_tb_prob'] = df.apply(lambda x: np.nanmean(slice_data(precip_data.rain_prob_mean, x['lat'], x['lon'], size=0.5)), axis=1)

        df['amsr_tb_rate_region'] = df.apply(lambda x: np.nanmean(slice_data(precip_data.rain_rate_mean, x['lat'], x['lon'], size=5)), axis=1)
        df['amsr_tb_rwr_region'] = df.apply(lambda x: np.nanmean(slice_data(precip_data.rain_rwr_mean, x['lat'], x['lon'], size=5)), axis=1)
        df['amsr_tb_prob_region'] = df.apply(lambda x: np.nanmean(slice_data(precip_data.rain_prob_mean, x['lat'], x['lon'], size=5)), axis=1)
        
        df['amsr_tb_rate_climo'] = df.apply(lambda x: np.nanmean(slice_data(precip_climo.rain_rate_mean, x['lat'], x['lon'], size=0.5)), axis=1)
        df['amsr_tb_rwr_climo'] = df.apply(lambda x: np.nanmean(slice_data(precip_climo.rain_rwr_mean, x['lat'], x['lon'], size=0.5)), axis=1)
        df['amsr_tb_prob_climo'] = df.apply(lambda x: np.nanmean(slice_data(precip_climo.rain_prob_mean, x['lat'], x['lon'], size=0.5)), axis=1)
    
    except (FileNotFoundError, AttributeError) as e:
        if isinstance(e, FileNotFoundError):
            print('could not find precip file for date, continuing:' + str(date))
            print(e)
        elif isinstance(e, AttributeError):
            print('attribute error, likely no data found' + str(date))
            print(e)
            
        df['amsr_tb_rate'] = np.nan
        df['amsr_tb_rwr'] = np.nan
        df['amsr_tb_prob'] = np.nan
        
        df['amsr_tb_rate_region'] = np.nan
        df['amsr_tb_rwr_region'] = np.nan
        df['amsr_tb_prob_region'] = np.nan

        df['amsr_tb_rate_climo'] = np.nan
        df['amsr_tb_rwr_climo'] = np.nan
        df['amsr_tb_prob_climo'] = np.nan
    except KeyError as e:
        print(precip_data.date)
        print(date)
        print(e)
    
    return df



@timed
def add_AMSR_to_df(df, date, bounds, region='SEP'):
    if region == 'CSET':
        amsr_data = xr.open_dataset(r'/home/disk/eos4/jkcm/Data/CSET/amsr/AMSR2_CWV_CSET_fixed.nc', lock=False)        
        amsr_cwv = amsr_data.CWV.sel(time=date.date(), method='nearest', tolerance=np.timedelta64(12, 'h'))
        # TODO add CSET climo
    elif region == 'SEP':
        amsr_file = f'/home/disk/eos9/jkcm/Data/amsr/rss/all/amsr_unified_{date.year}-{date.month:02}.nc'
        amsr_data = xr.open_dataset(amsr_file, lock=False)        
        amsr_cwv = amsr_data.vapor.sel(time=date.date(), method='nearest', tolerance=np.timedelta64(12, 'h')).sel(orbit_segment=0)
        amsr_climo_data = xr.open_dataset(f'/home/disk/eos9/jkcm/Data/amsr/rss/all/amsr_unified.seasmean.nc', lock=False)        
        amsr_cwv_climo = amsr_climo_data.vapor.sel(time=date.date(), method='nearest', tolerance=np.timedelta64(2, 'M')).sel(orbit_segment=0)
    else: 
        raise ValueError('MERRA region specification not known, please specify "CSET" or "SEP"')  
    
    df['amsr_cwv'] = df.apply(lambda x: np.nanmean(slice_data(amsr_cwv, x['lat'], x['lon'], size=0.5)), axis=1)
    df['amsr_cwv_region'] = df.apply(lambda x: np.nanmean(slice_data(amsr_cwv, x['lat'], x['lon'], size=5)), axis=1)
    df['amsr_cwv_climo'] = df.apply(lambda x: np.nanmean(slice_data(amsr_cwv_climo, x['lat'], x['lon'], size=0.5)), axis=1)
    return df
    
    
@timed
def add_ASCAT_to_df(df, date, bounds):
    """add in ASCAT divergence to the dataframe
        'descending' branch, segment 0, = "nighttime", but it's the 9:30am one. About 4-5 hours off MODIS time
        lats and lons are centered in middle of scene, so slice +-1/2 degree"""
    
    date_adj = date-np.timedelta64(5, 'h')#diff between MODIS and ASCAT overpass times
    ascat_data = xr.open_dataset(f'/home/disk/eos9/jkcm/Data/ascat/rss/all/ascat_unified_{date_adj.year}-{date_adj.month:02}.nc', lock=False)
    try:
        ascat_data = ascat_data.sel(time=date_adj.date(), method='nearest', tolerance=np.timedelta64(12, 'h')).sel(orbit_segment=0, 
                                      latitude=slice(*bounds['lat']), longitude=slice(*bounds['lon']))
        try:
            ascat_data = utils.get_ascat_divergence(ascat_data)
        except ValueError as e:
            print(ascat_data)
            raise e
        ascat_climo = xr.open_dataset(r'/home/disk/eos9/jkcm/Data/ascat/rss/proc/ascat_unified.seasmean.nc', lock=False)
        ascat_climo = ascat_climo.sel(time=date, method='nearest', tolerance=np.timedelta64(2, 'M')).sel(orbit_segment=0, 
                                      latitude=slice(*bounds['lat']), longitude=slice(*bounds['lon']))  # TODO check
        ascat_climo = utils.get_ascat_divergence(ascat_climo)

        df['ascat_div'] = df.apply(lambda x: np.nanmean(slice_data(ascat_data.div, x['lat'], x['lon'], size=0.5)), axis=1)
        df['ascat_div_std'] = df.apply(lambda x: np.nanstd(slice_data(ascat_data.div, x['lat'], x['lon'], size=0.5)), axis=1)
        df['ascat_div_region'] = df.apply(lambda x: np.nanmean(slice_data(ascat_data.div, x['lat'], x['lon'], size=5)), axis=1)
        df['ascat_div_climo'] = df.apply(lambda x: np.nanmean(slice_data(ascat_climo.div, x['lat'], x['lon'], size=0.5)), axis=1)
    except KeyError as e:
        print(f'KeyError caught in add_ASCAT_to_df, for date {date}. filling with nans and moving on')
        print(e)
        df['ascat_div'] = np.nan
        df['ascat_div_std'] = np.nan
        df['ascat_div_region'] = np.nan
        df['ascat_div_climo'] = np.nan
    return df

    
@timed
def add_MERRA_to_df(df, date, bounds, region='SEP'):
    if region == 'CSET':
        MERRA_data = xr.open_dataset(r'/home/disk/eos4/jkcm/Data/CSET/MERRA/measures/MERRA_unified_subset.nc', lock=False)
        # MERRA_climo = TODO ADD CSET MERRA CLIMO
    elif region == 'SEP':
        MERRA_data = xr.open_dataset(
            f'/home/disk/eos4/jkcm/Data/MERRA/measures/split/MERRA_unified_subset.{date.year}-{date.month:02}.nc', lock=False)
        MERRA_data = MERRA_data.sel(time=date, method='nearest', tolerance=np.timedelta64(3, 'h'))
        MERRA_climo = xr.open_dataset(
            r'/home/disk/eos4/jkcm/Data/MERRA/measures/MERRA_unified_subset_SEP.seasmean.nc', lock=False)
        MERRA_climo = MERRA_climo.sel(time=date, method='nearest', tolerance=np.timedelta64(2, 'M'))
    else: 
        raise ValueError('MERRA region specification not known, please specify "CSET" or "SEP"')  

    for var in ['sfc_div', 'div_700', 'SST', 'EIS', 'LTS', 'RH_700', 'WSPD_10M']:
        df['MERRA_'+var] = df.apply(lambda x: np.nanmean(slice_data(MERRA_data[var], x['lat'], x['lon'], size=0.5)), axis=1)
        df['MERRA_'+var+'_region'] = df.apply(lambda x: np.nanmean(slice_data(MERRA_data[var], x['lat'], x['lon'], size=5)), axis=1)
        df['MERRA_'+var+'_climo'] = df.apply(lambda x: np.nanmean(slice_data(MERRA_climo[var], x['lat'], x['lon'], size=0.5)), axis=1)
    return df


@timed
def redo_add_MERRA_to_df(df, date, region='SEP'):
    if region == 'CSET':
        raise NotImplementedError()
#         MERRA_data = xr.open_dataset(r'/home/disk/eos4/jkcm/Data/CSET/MERRA/measures/MERRA_unified_subset.nc', lock=False)
#         # MERRA_climo = TODO ADD CSET MERRA CLIMO
    elif region == 'SEP':
        MERRA_data = xr.open_dataset(
            f'/home/disk/eos4/jkcm/Data/MERRA/measures/subset/2/MERRA2.unified_subset.{date.year}{date.month:02}.nc', lock=False)
        MERRA_data = MERRA_data.sel(time=date, method='nearest', tolerance=np.timedelta64(3, 'h'))
        MERRA_climo = xr.open_dataset(
            r'/home/disk/eos4/jkcm/Data/MERRA/measures/subset/2/MERRA2.unified_subset.seasmean.nc', lock=False)
        MERRA_climo = MERRA_climo.sel(time=date, method='nearest', tolerance=np.timedelta64(2, 'M'))
    else: 
        raise ValueError('MERRA region specification not known, please specify "CSET" or "SEP"')  

    for var in ['sfc_div', 'div_700', 'SST', 'EIS', 'LTS', 'RH_700', 'WSPD_10M', 'PS', 'TQV', 'TQL', 'T2M', 'M', 'T_adv', 'T_700',
               'ISCCPALB', 'MDSCLDFRCLO']:
        df['MERRA2_'+var] = df.apply(lambda x: np.nanmean(slice_data(MERRA_data[var], x['lat'], x['lon'], size=0.5)), axis=1)
        df['MERRA2_'+var+'_region'] = df.apply(lambda x: np.nanmean(slice_data(MERRA_data[var], x['lat'], x['lon'], size=5)), axis=1)
        df['MERRA2_'+var+'_climo'] = df.apply(lambda x: np.nanmean(slice_data(MERRA_climo[var], x['lat'], x['lon'], size=0.5)), axis=1)
    return df

@timed
def add_all_to_df(df, region='SEP', MERRA=False, AMSR=False, CERES=False, ASCAT=False):
    if not len(np.unique(df.day)) == 1:
        raise ValueError('hey boss I thought we were gonna take it one day at a time')
    dates = sorted(df.datetime)
    date = dates[len(dates)//2]
    try:
        bounds = {'lat':(np.nanmin(df.lat)-0.5, np.nanmax(df.lat)+0.5), 'lon':(np.nanmin(df.lon)-0.5, np.nanmax(df.lon)+0.5)}
    except TypeError as e:
        print({'lat':(np.nanmin(df.lat), np.nanmax(df.lat)), 'lon':(np.nanmin(df.lon), np.nanmax(df.lon))})
        raise e
    
    if MERRA:
        df = add_MERRA_to_df(df, date, bounds, region=region)
    if ASCAT:
        df = add_ASCAT_to_df(df, date, bounds)
    if AMSR:
        df = add_AMSR_to_df(df, date, bounds, region=region)
    if CERES:
        df = add_CERES_to_df(df, date, bounds)
    return df

def add_a_bit_to_df(df):
    if not len(np.unique(df.day)) == 1:
        raise ValueError('hey boss I thought we were gonna take it one day at a time')
    dates = sorted(df.datetime)
    date = dates[len(dates)//2]
    print(f'adding a bit, apples: {date}')
    sys.stdout.flush()
    df = add_precip_to_df(df, date)
    return df

def add_some_more_MERRA_to_df(df):
    if not len(np.unique(df.day)) == 1:
        raise ValueError('hey boss I thought we were gonna take it one day at a time')
    dates = sorted(df.datetime)
    date = dates[len(dates)//2]
    print(f'adding a bit, apples: {date}')
    sys.stdout.flush()
#     df = redo_add_MERRA_to_df(df, date)
#     df = add_precip_to_df(df, date)
    df = add_more_CERES_to_df(df, date)
    return df


def applyParallel(dfGrouped, func):
    #take a grouped pandas dataframe and apply a function over its groups
    with Pool(cpu_count()//2) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])
    return pd.concat(ret_list)

def applyPartial(dfGrouped, func, args):
    with Pool(12) as p:
        ret_list = p.map(partial(func, **args), [group for name, group in dfGrouped])
    return pd.concat(ret_list)

def applySingle(dfGrouped, func, args):
    ret_list = map(partial(func, **args), [group for name, group in dfGrouped])
    return pd.concat(ret_list)

def process_month(year, month):
    all_class_df = utils.load_class_data('all')
    sep_class_df = all_class_df[all_class_df['loc']=='SEP']


if __name__ == "__main__":
    region = sys.argv[1]
    year = sys.argv[2]
    loadfile = f'/home/disk/eos4/jkcm/Data/MEASURES/beta_data/classified_df_final.{region}_{year}_4.pickle'
    savefile = f'/home/disk/eos4/jkcm/Data/MEASURES/beta_data/classified_df_final.{region}_{year}_5.pickle'
    df = pickle.load(open(loadfile, 'rb'))
    by_granule = df.groupby('granule')

    new_df = applyPartial(dfGrouped=by_granule, func=add_some_more_MERRA_to_df, 
                          args=dict())
    print(savefile)
    pickle.dump(new_df, open(savefile, "wb" ))
    

    
    
    
    
#     n_args = len(sys.argv)
#     if n_args < 2 or n_args > 3:
#         raise ValueError('gimme a region (CSET or SEP), and if you give me SEP, give me a year (2014, 2015, or 2016). thanks pal.')
#     region = sys.argv[1].upper()
#     if not ((n_args == 2 and region == 'CSET') or (n_args == 3 and region == 'SEP')):
#         raise ValueError('gimme a region (CSET or SEP), and if you give me SEP, give me a year (2014, 2015, or 2016). thanks pal.')
#     if region == 'CSET':
#         raise NotImplementedError('hey uhhh we still need to work on CSET functionality!')
#         #         class_df = utils.load_class_data('CSET')
#     elif region == 'SEP':
#         year = int(sys.argv[2])
#         if not year in [2014, 2015, 2016]:
#             raise ValueError('baaaad year pal')
#         all_class_df = utils.load_class_data('all')
#         df = all_class_df[all_class_df['loc']=='SEP']
#         df = df[df['year'] == year]
#     else:
#         raise ValueError('dunno what do do with this input, friend.')
#     by_day = df.groupby('day')
#     by_granule = df.groupby('granule')
#     savefile = f'/home/disk/eos4/jkcm/Data/MEASURES/beta_data/classified_df_final.{region}_{year}_2.pickle'

#     new_df = applyPartial(dfGrouped=by_granule, func=add_all_to_df, 
#                           args=dict(region=region, MERRA=True, AMSR=True, CERES=True, ASCAT=True))
#     print(savefile)
#     pickle.dump(new_df, open(savefile, "wb" ))
    
    
    
    
    
    
#below is probably trash    
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

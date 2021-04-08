import os
os.environ['PROJ_LIB'] = '/home/disk/p/jkcm/anaconda3/envs/classified-cset/share/proj'
import sys
sys.path.insert(0, '/home/disk/p/jkcm/Code')
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import xarray as xr
import pickle
import matplotlib.pyplot as plt
import datetime as dt
# from Lagrangian_CSET import met_utils as mu
import multiprocessing as mp
from tools.LoopTimer import LoopTimer
import collections
# from Lagrangian_CSET.utils import as_datetime
from scipy.stats import sem
import random
import pytz
from mpl_toolkits.basemap import Basemap



"""
UTILS WRITTEN:
get_div_from_x_y: divergence values from u, v, winds
get_ascat_divergence: add divergence to ASCAT dataset
load_class_data: load all classifications for a given region name
plot_MERRA2_var_dists: plot standard histogram of gridded data
plot_dataframe_by_cat: plot standard histogram of tabular data
"""

labels={0: 'Closed-cellular MCC', 1: 'Clustered cumulus', 2: 'Disorganized MCC',
        3: 'Open-cellular MCC', 4: 'Solid Stratus', 5: 'Suppressed Cu'}
short_labels = {0: 'Closed MCC ', 1: 'Clust. Cu', 2: 'Disorg. MCC',
        3: 'Open MCC', 4: 'Solid St', 5: 'Supp. Cu', 6: 'All scenes'}
colors = [mpl.cm.get_cmap('viridis')(i) for i in np.linspace(0,1,6)]
ordering = [4, 0, 2, 3, 1, 5]

true_sample_size = pickle.load(open('/home/disk/eos4/jkcm/Data/MEASURES/true_sample_sizes.pickle', 'rb'))

name_map = {'MERRA_EIS': 'EIS',
            'MERRA_LTS': 'LTS',
            'MERRA_SST': 'SST',
            'MERRA_div_ls': 'div_700',
            'MERRA_div_sfc': 'sfc_div',
            'MERRA_wpsd': 'WSPD_10M',
            'MERRA_RH700': 'RH_700',
            'amsr_cwv': 'vapor',
            'ceres_net_crf': 'net_cre',
            'ceres_cld_area_low_1h': 'cldarea_low_1h'}


def get_div_from_u_v(u, v, lat, lon):
    lat_dim = np.nonzero(np.array(u.shape) == lat.shape)[0][0]
    lon_dim = np.nonzero(np.array(v.shape) == lon.shape)[0][0]    
    dudi = np.gradient(u)[lon_dim]
    dvdj = np.gradient(v)[lat_dim]
    dlatdy = 360/4.000786e7  # degrees lat per meter y
    def get_dlondx(lat) : return(360/(np.cos(np.deg2rad(lat))*4.0075017e7))
    dlondx = get_dlondx(lat)
    lat_spaces = np.diff(lat)
    lon_spaces = np.diff(lon)
    assert(np.allclose(lat_spaces, lat_spaces[0], atol=0.01) and np.allclose(lon_spaces, lon_spaces[0], atol=0.05))
    dlondi = np.mean(lon_spaces)
    dlatdj = np.mean(lat_spaces)
    try:
        dudx = dudi/dlondi*dlondx[:, None]
    except IndexError as e:
        print(dudi.shape, dlondi.shape, dlondx.shape)
        dudx = dudi/dlondi*dlondx
    dvdy = dvdj/dlatdj*dlatdy
    div = dudx + dvdy
    return div


def get_ascat_divergence(ds):
    rho = ds.windspd.values
    phi = ds.winddir.values
#     print('here4')
    sys.stdout.flush()
    u = rho*4
#     print('here51')
    sys.stdout.flush()   
    u = rho*np.cos(np.deg2rad((-phi+90)%360))
    v = rho*np.sin(np.deg2rad((-phi+90)%360))
    lat = ds.coords['latitude'].values
    lon = ds.coords['longitude'].values
    try:
        div = get_div_from_u_v(u, v, lat, lon)
    except ValueError as e:
        print(f'ValueError in get_div_from_u_v: {np.nanmean(lat)}, {np.nanmean(lon)}, {len(ds)}')
        print(ds)
        div = np.full_like(ds.windspd.values, fill_value=np.nan)
    ds['div'] = (ds.windspd.dims, div)
    ds['div'] = ds['div'].assign_attrs(
                    {"long_name": "scatterometer divergence",
                     "units": "s**-1",
                     "_FillValue": "NaN"})
#     print('here8')
    sys.stdout.flush()
    return ds

    
def as_datetime(date, timezone=pytz.UTC):
    "Converts all datetimes types to datetime.datetime with TZ = UTC"
    def to_dt(d, timezone):
        """does all the heavy lifting
        """
        supported_types = (np.datetime64, dt.datetime)
        if not isinstance(d, supported_types):
            raise TypeError('type not supported: {}'.format(type(d)))
        if isinstance(d, np.datetime64):
            # TODO: add timezoneawareness here
            ts = (d - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
            d = dt.datetime.utcfromtimestamp(ts)
        if isinstance(d, pd.Timestamp):
            d = d.to_datetime()
        if isinstance(d, dt.datetime):
            if d.tzinfo is None:
                return(d.replace(tzinfo=timezone))
            else:
                return(d.astimezone(timezone))

    if isinstance(date, (collections.Sequence, np.ndarray)):
        return np.array([to_dt(x, timezone) for x in date])
    return to_dt(date, timezone)
    

def load_class_data(dataset_name):
    if dataset_name == 'cset':
        classification_file=r'/home/disk/eos4/jkcm/Data/MEASURES/beta_data/all_unified/unified_table_NEP_2015.csv'
    elif dataset_name == 'all':
        classification_file=r'/home/disk/eos4/jkcm/Data/MEASURES/beta_data/unified_table_all.csv'
    else:
        raise ValueError('unknown dataset name')
    class_df = pd.read_csv(classification_file, index_col=0).drop(columns=['refl_img', 'context_img'])
    class_df['datetime'] = [dt.datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in class_df['date']]
    class_df['granule'] = [i[:22] for i in class_df.name.values]
    class_df['lat'] = pd.to_numeric(class_df['lat'], errors='coerce')
    class_df['lon'] = pd.to_numeric(class_df['lon'], errors='coerce')%360
    class_df['day'] = [i[:10] for i in class_df.date]
    class_df['month'] = [int(i[5:7]) for i in class_df.date]

    if dataset_name in ['all']:
        class_df['locyear'] = class_df['loc']+'_' + class_df['year'].values.astype(int).astype(str)
        class_df['season'] = [['DJF','MAM','JJA','SON'][np.floor((i.month)%12/3).astype(int)] for i in class_df.datetime]
    
    return class_df


def plot_MERRA2_var_dists(MERRA2_data, varname, lev=None, xlims=None, xlabel=None, ax=None, scale=1, savename=None, verbose=False, filt=None):
    if not ax:
        fig, ax = plt.subplots(figsize=(10,6))
    else:
        fig = ax.figure
    colors = [mpl.cm.get_cmap('viridis')(i) for i in np.linspace(0,1,6)]
    ordering = [4, 0, 2, 3, 1, 5]
    for i, name in enumerate(ordering):
        if filt:
            if name not in filt:
                continue
        if lev:
            lev_data = MERRA2_data.sel(lev=lev)
            all_vals = lev_data[varname].where(MERRA2_data['cat']==name).values.flatten()*scale
        else:    
            all_vals = MERRA2_data[varname].where(MERRA2_data['cat']==name).values.flatten()*scale
        if verbose:
                print(short_labels[name]+':', np.sum(MERRA2_data['cat']==name).item())
                print(sum(~np.isnan(all_vals)))
#             print(f'{short_labels[name]}:, total:{np.sum(MERRA2_data['cat']==name).item()}, usable:{sum(~np.isnan(all_vals))}')

        sns.distplot(all_vals, hist = False, kde = True,
                     kde_kws = {'shade': True, 'linewidth': 3},
                     label = short_labels[name], color=colors[i], ax=ax)
    if xlims:
        ax.set_xlim(xlims)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel("normed density")
    if savename:
        fig.savefig(savename, bbox_inches='tight')
            
            
def plot_dataframe_by_cat(dataframe, varname, scale_factor=1, xlims=None, ylabel=None, ax=None, savename=None,
                          cert_thresh=None, verbose=False, hist=False, label_pct=False, ylims=None, normed=True):
    if not ax:
        fig, ax = plt.subplots(figsize=(10,6))
    else:
        fig = ax.figure
    
    grouped = dataframe.groupby('cat')
    colors = [mpl.cm.get_cmap('viridis')(i) for i in np.linspace(0,1,6)]
    ordering = [4, 0, 2, 3, 1, 5]
    for i, name in enumerate(ordering):
        group = grouped.get_group(name)
        vals = group[varname].values*scale_factor
        if cert_thresh:
            vals = vals[group.cert>cert_thresh]
        if verbose:
            print(f'{short_labels[name]}:, total:{len(vals)}, usable:{sum(~np.isnan(vals))/len(vals):0.0%}')
        label_addon = f' ({sum(~np.isnan(vals))/sum(~np.isnan(dataframe[varname])):0.0%})' if label_pct else ''
        sns.distplot(vals[~np.isnan(vals)], hist=hist, kde=(not hist), norm_hist=normed,# bins=100,
                     kde_kws={'shade': True, 'linewidth': 3},
                     hist_kws={'histtype': "step", 'linewidth': 3, "alpha": 1, "edgecolor":colors[i]},
                     label=short_labels[name]+label_addon, color=colors[i], ax=ax)
    sns.distplot(dataframe[varname][~np.isnan(dataframe[varname])], hist=hist, kde=(not hist), norm_hist=normed,# bins=100,
                     kde_kws={'shade': False, 'linewidth': 1, "linestyle": "dashed"},
                     hist_kws={'histtype': "step", 'linewidth': 1, "linestyle": "dashed", "alpha": 0.5, "edgecolor":'k'},
                     label="all scenes", color="k", ax=ax)
    if xlims:
        ax.set_xlim(xlims)
    if ylims:
        ax.set_ylim(ylims)
    if ylabel:
        ax.set_xlabel(xlabel)
    if (hist and not normed):
        ax.set_ylabel("count")
    else:
        ax.set_ylabel("normed density")
    if savename:
        fig.savefig(savename, bbox_inches='tight') 
    plt.legend()
    return fig, ax
    
# def plot_cat_vals(vals_dict, ax=None):
#     if not ax:
#         fig, ax = plt.subplots(figsize=(10,6))
#     else:
#         fig = ax.figure

#     if 6 in vals_dict:
#         colors.append('k')
#         ordering.append(6)

#     for i, name in enumerate(ordering):
#         ax.errorbar(i, mean, yerr=stderr, c=colors[i], fmt='o')
        
#                     stderr = sem(vals[~np.isnan(vals)])
#             mean = np.nanmean(vals)
        
def plot_mean_by_cat_b(dataframe, varname, ax, scale_factor=1, offset=0, **plargs):
    fig = ax.figure
    yeargroup = dataframe.groupby('year')
    for year in yeargroup.groups.keys():
        year_df = yeargroup.get_group(year)
        grouped = year_df.groupby('cat')
        colors = [mpl.cm.get_cmap('viridis')(i) for i in np.linspace(0,1,6)] + ['k']
        ordering = [4, 0, 2, 3, 1, 5, 6]
        for i, name in enumerate(ordering):
            if not name==6:
                group = grouped.get_group(name)
                vals = group[varname].values*scale_factor
            else:
                vals = year_df[varname].values*scale_factor
            mean = np.nanmean(vals)
            if 'clim' in varname or 'region' in varname:
                lookup = '_'.join(varname.split('_')[:-1])
            else:
                lookup = varname
            if lookup in name_map.keys(): # can do effective sample size adjustment
                if name == 6:
                    name = 'all'
                sample_frac = true_sample_size[name_map[lookup]][name]['frac']
                ss = sum(~np.isnan(vals))*sample_frac
#                 print(sample_frac)
                bar_width = np.nanstd(vals)/(ss**(1/2))
            else:
#                 print(varname)
                bar_width = sem(vals[~np.isnan(vals)])

    #         bar_width = np.nanstd(vals)
            ax.errorbar(i+offset, mean, yerr=bar_width, c=colors[i], **plargs)

    #         ax.boxplot(vals[~np.isnan(vals)], positions=[i+ofs], sym='')
    ax.set_xticks(sorted(ordering))
    ax.set_xticklabels([short_labels[i] for i in ordering], rotation=70)
    ylims = ax.get_ylim()
    if 0>ylims[0] and 0<ylims[1]:
        ax.axhline(y=0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], c='k')
    return fig, ax


def plot_mean_by_cat(dataframe, varname, scale_factor=1, ylabel=None, ax=None, savename=None,
                          cert_thresh=None, verbose=False, hist=False, bars='true_stderr', total=True):
    if not ax:
        fig, ax = plt.subplots(figsize=(10,6))
    else:
        fig = ax.figure
    
    grouped = dataframe.groupby('cat')
    colors = [mpl.cm.get_cmap('viridis')(i) for i in np.linspace(0,1,6)]
    ordering = [4, 0, 2, 3, 1, 5]
    if total:
        colors.append('k')
        ordering.append(6)
    for i, name in enumerate(ordering):
        if not name==6:
            group = grouped.get_group(name)
            vals = group[varname].values*scale_factor
        else:
            vals = dataframe[varname].values*scale_factor
        if cert_thresh:
            vals = vals[group.cert>cert_thresh]
        if bars=='bootstrap':
            res = []
            for r in range(100):
#                 res.append(np.nanmean(random.choices(vals[~np.isnan(vals)], k=min(sum(~np.isnan(vals)), 1000))))
                res.append(np.nanmean(random.choices(vals[~np.isnan(vals)], k=min(sum(~np.isnan(vals))//100, 1000))))
                
            bar_width = np.nanstd(res)
            mean = np.nanmean(res)
        elif bars=='true_stderr':
            print(varname)
            mean = np.nanmean(vals)
            ss = true_sample_size[varname][name]
            bar_width = np.nanstd(vals)/np.sqrt(ss)
        elif bars=='stderr':
            mean = np.nanmean(vals)
            bar_width = sem(vals[~np.isnan(vals)])
        elif bars=='stddev':
            mean = np.nanmean(vals)
            bar_width = np.nanstd(vals)
        ax.errorbar(i, mean, yerr=bar_width, c=colors[i], fmt='o')
        if verbose:
            print(f'{short_labels[name]}:, total:{len(vals)}, usable:{sum(~np.isnan(vals))/len(vals):0.0%}')
            print(f'       mean:{mean}, stderr:{stderr}')
        
    ax.set_xticks(sorted(ordering))
    ax.set_xticklabels([short_labels[i] for i in ordering], rotation=70)
    if ylabel:
        ax.set_ylabel(ylabel)
    ylims = ax.get_ylim()
    if 0>ylims[0] and 0<ylims[1]:
        ax.axhline(y=0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], c='k')
#     ax.set_ylim(ylims)
    if savename:
        fig.savefig(savename, bbox_inches='tight')
        
    return fig, ax


def nan_correlate(x,y):
    idx = np.logical_and(~np.isnan(x), ~np.isnan(y))
    return np.corrcoef(x[idx],y[idx])[0][1]



#added lev, removed dataframe, 
def plot_mean_by_cat_3d(MERRA2_data, varname, lev=None, scale_factor=1, ylabel=None, ax=None, savename=None,
                          cert_thresh=None, verbose=False, hist=False, bars=None):
    
    if not ax:
        fig, ax = plt.subplots(figsize=(10,6))
    else:
        fig = ax.figure
    colors = [mpl.cm.get_cmap('viridis')(i) for i in np.linspace(0,1,6)]
    ordering = [4, 0, 2, 3, 1, 5]
    if lev:
        MERRA2_data = MERRA2_data.sel(lev=lev)
    for i, name in enumerate(ordering):
        vals = MERRA2_data[varname].values[MERRA2_data['cat']==name]*scale_factor

        if cert_thresh:
            vals = vals[group.cert>cert_thresh]
        
        if bars=='bootstrap':
            res = []
            for r in range(100):
#                 res.append(np.nanmean(random.choices(vals[~np.isnan(vals)], k=min(sum(~np.isnan(vals)), 1000))))
                res.append(np.nanmean(random.choices(vals[~np.isnan(vals)], k=sum(~np.isnan(vals)/100))))
            bar_width = np.nanstd(res)
            mean = np.nanmean(res)
        elif bars=='stderr':
            mean = np.nanmean(vals)
            bar_width = sem(vals[~np.isnan(vals)])
        elif bars=='stddev':
            mean = np.nanmean(vals)
            bar_width = np.nanstd(vals)
        ax.errorbar(i, mean, yerr=bar_width, c=colors[i], fmt='o')
        if verbose:
            print(f'{short_labels[name]}:, total:{len(vals)}, usable:{sum(~np.isnan(vals))}')
            print(f'       mean:{mean}, stderr:{stderr}')
        
        
    ax.set_xticks(sorted(ordering))
    ax.set_xticklabels([short_labels[i] for i in ordering], rotation=45)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ax.get_ylim()[0]<0 and ax.get_ylim()[1]>0:
        ax.axhline(y=0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1])
    if savename:
        fig.savefig(savename, bbox_inches='tight')
        
    return fig, ax


def bmap(ax=None, drawlines=True, llr=None, par_labs=[1, 1, 0, 0], mer_labs=[0, 0, 1, 1], 
         merspace=15, parspace=15, **kwargs):

    if ax is None:
        fig, ax = plt.subplots()

    if llr is None:
        lat_range = latlon_range['lat']
        lon_range = latlon_range['lon']
    else:
        lat_range = llr['lat']
        lon_range = llr['lon']
    if 'projection' not in kwargs.keys():
        kwargs['projection'] = 'cyl'
        kwargs['rsphere'] =(6378137.00, 6356752.3142)
    m = Basemap(llcrnrlon=lon_range[0], llcrnrlat=lat_range[0],
                urcrnrlon=lon_range[1],  urcrnrlat=lat_range[1],
                ax=ax, resolution='l', **kwargs)
    if drawlines:
        m.drawparallels(np.arange(-90., 90., parspace), labels=par_labs, fontsize=14)
        m.drawmeridians(np.arange(-180., 180., merspace), labels=mer_labs, fontsize=14)
    m.drawcoastlines()
    m.fillcontinents(color="white", lake_color="white")

    return m


def add_all_ASCAT_divergence():
#     files = glob.glob(r'/home/disk/eos9/jkcm/Data/ascat/rss/all/ascat_unified_20*.nc')
    lt = LoopTimer(len(files))
    for f in files:
        lt.update()
        savefile = os.path.join(os.path.dirname(f), os.path.basename(f)[:-2]+'div.nc')
        ascat_data = xr.open_dataset(f)
        ascat_data = utils.get_ascat_divergence(ascat_data)
        ascat_data.to_netcdf(savefile)
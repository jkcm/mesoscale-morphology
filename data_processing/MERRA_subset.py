from Lagrangian_CSET import met_utils as mu
import numpy as np
import glob
import xarray as xr
import time

for year in [2014, 2015, 2016]:
    for month in np.arange(1,13):
        
    


        #MERRA_data = xr.open_mfdataset(sorted(glob.glob(r'/home/disk/eos4/jkcm/Data/CSET/MERRA/unified_2/*.unified*.nc4')), combine='by_coords')
        # MERRA2_data =  xr.open_mfdataset(sorted(glob.glob(r'/home/disk/eos4/jkcm/Data/MERRA/pressure/*.inst3_3d_asm_Np*.nc')), combine='by_coords')
#         MERRA2_data = xr.open_dataset(r'/home/disk/eos4/jkcm/Data/MERRA/measures/MERRA2_400.inst3_3d_asm_Np.SEP.nc')
        MERRA2_data = xr.open_mfdataset(f'/home/disk/eos4/jkcm/Data/MERRA/measures/lev/MERRA2_400.inst3_3d_asm_Np.{str(year)}{month:02}*.SUB.nc', combine='by_coords')
        MERRA2_data['lon'] = MERRA2_data.lon%360
        
#         MERRA2_csp = xr.open_dataset(r'/home/disk/eos4/jkcm/Data/MERRA/measures/MERRA2_400.tavg1_2d_csp_Nx.SEP.nc')
        MERRA2_csp = xr.open_mfdataset(f'/home/disk/eos4/jkcm/Data/MERRA/measures/cloud/MERRA2_400.tavg1_2d_csp_Nx.{str(year)}{month:02}*.nc4.nc4', combine='by_coords')
        MERRA2_csp['lon'] = MERRA2_csp.lon%360

        # MERRA2_sfc = xr.open_mfdataset(sorted(glob.glob(r'/home/disk/eos4/jkcm/Data/MERRA/sfc/*.inst*.nc4')), combine='by_coords')
#         MERRA2_sfc = xr.open_dataset(r'/home/disk/eos4/jkcm/Data/MERRA/measures/MERRA2_400.instU_2d_asm_Nx.SEP.nc')
        MERRA2_sfc = xr.open_mfdataset(f'/home/disk/eos4/jkcm/Data/MERRA/measures/sfc/MERRA2_400.inst1_2d_asm_Nx.{str(year)}{month:02}*.nc4.nc4', combine='by_coords')
        MERRA2_sfc['lon'] = MERRA2_sfc.lon%360
        
        #add some sfc stuffs
        time_arr = [True if i in MERRA2_data.time.values else False for i in MERRA2_sfc.time.values]
        MERRA2_sfc_subs = MERRA2_sfc.isel(time=time_arr)
        MERRA2_data['SST'] = MERRA2_sfc_subs['TS']
        MERRA2_data['U10M'] = MERRA2_sfc_subs['U10M']
        MERRA2_data['V10M'] = MERRA2_sfc_subs['V10M']
        MERRA2_data['T2M'] = MERRA2_sfc_subs['T2M']
        MERRA2_data['TQL'] = MERRA2_sfc_subs['TQL']
        MERRA2_data['TQV'] = MERRA2_sfc_subs['TQV']
        wspd = np.sqrt(MERRA2_sfc_subs['U10M'].values**2 + MERRA2_sfc_subs['V10M'].values**2)
        MERRA2_sfc_subs['WSPD_10M'] = (('time', 'lat', 'lon'), wspd)
        MERRA2_sfc_subs['WSPD_10M'] = MERRA2_sfc_subs['WSPD_10M'].assign_attrs(
                        {"long_name": "wind speed 10m",
                         "units": "m s**-1"})
        MERRA2_data['WSPD_10M'] = MERRA2_sfc_subs['WSPD_10M']


        #add some COSP stuffs
        time_arr = np.full(MERRA2_csp.time.values.shape, False)
        for i, t in enumerate(MERRA2_data.time.values):
            time_arr[np.argmin(np.abs(MERRA2_csp.time.values-t))] = True        
        MERRA2_csp_subs = MERRA2_csp.isel(time=time_arr)
        MERRA2_csp_subs = MERRA2_csp_subs.assign_coords(time=MERRA2_data.time)
        MERRA2_data['ISCCPALB'] = MERRA2_csp_subs['ISCCPALB']
        MERRA2_data['MDSCLDFRCLO'] = MERRA2_csp_subs['MDSCLDFRCLO']
        MERRA2_data['MDSH2OPATH'] = MERRA2_csp_subs['MDSH2OPATH']
        MERRA2_data['MDSH2OPATH'] = MERRA2_csp_subs['MDSH2OPATH']
        MERRA2_data['MDSCLDSZH20'] = MERRA2_csp_subs['MDSCLDSZH20']
        MERRA2_data['MDSOPTHCKH2O'] = MERRA2_csp_subs['MDSOPTHCKH2O']


        #some spare levs
        MERRA2_data["RH_700"] = MERRA2_data.RH.sel(lev=700)
        MERRA2_data["T_700"] = MERRA2_data.T.sel(lev=700)


        # #ADD SOME MORE MERRA VARS
        t_1000 = MERRA2_data.T.sel(lev=1000)
        theta_700 = mu.theta_from_p_T(p=700, T=MERRA2_data.T.sel(lev=700))
        LTS = theta_700-t_1000
        t_dew = t_1000-(100-100*MERRA2_data.RH.sel(lev=1000))/5
        lcl = mu.get_LCL(t=t_1000, t_dew=t_dew, z=MERRA2_data.H.sel(lev=1000))
        z_700 = MERRA2_data.H.sel(lev=700)
        gamma_850 = mu.get_moist_adiabatic_lapse_rate(MERRA2_data.T.sel(lev=850), 850)
        EIS = LTS - gamma_850*(z_700-lcl)
        MERRA2_data['LTS'] = LTS
        MERRA2_data['EIS'] = EIS

        t_v = mu.tvir_from_T_w(MERRA2_data.T, MERRA2_data.QV)
        rho = mu.density_from_p_Tv(MERRA2_data.lev*100, t_v)
        MERRA2_data['dzdt'] = -MERRA2_data['OMEGA']/(9.81*rho)
        MERRA2_data['div_700'] = MERRA2_data['dzdt'].sel(lev=700)/MERRA2_data['H'].sel(lev=700)


        theta_sst = mu.theta_from_p_T(p=MERRA2_data.PS/100, T=MERRA2_data.SST)
        theta_800 = mu.theta_from_p_T(p=800, T=MERRA2_data.T.sel(lev=800))

        MERRA2_data['M'] = theta_sst - theta_800


        [_, dtdi, dtdj] = np.gradient(MERRA2_data.SST)
        #di is dlat, dj is dlon
        lat_spaces = np.diff(MERRA2_data.coords['lon'].values)
        lon_spaces = np.diff(MERRA2_data.coords['lon'].values)
        assert(np.allclose(lat_spaces, 0.625, atol=0.01) and np.allclose(lon_spaces, 0.625, atol=0.05))
        # PREVIOUSLY HAD NEGATIVE LAT_SPACES
        dlondj = np.mean(lon_spaces)
        dlatdi = np.mean(lat_spaces)
        def get_dlondx(lat) : return(360/(np.cos(np.deg2rad(lat))*4.0075017e7))
        dlondx = get_dlondx(MERRA2_data.coords['lat'].values)
        dlatdy = 360/4.000786e7  # degrees lat per meter y
        dtdx = dtdj/dlondj*dlondx[None,:,None]
        dtdy = dtdi/dlatdi*dlatdy

        T_adv = -(MERRA2_data.U10M.values*dtdx + MERRA2_data.V10M.values*dtdy)
        MERRA2_data['T_adv'] = (('time', 'lat', 'lon'), T_adv, {'units': "K s**-1", 
                          'long_name': "temperature_advection"})
        


        #ADDING SOME MERRA DIVERGENCE STUFFS     
        dudi = np.gradient(MERRA2_data.U10M)[2]
        dvdj = np.gradient(MERRA2_data.V10M)[1]

        dlatdy = 360/4.000786e7  # degrees lat per meter y
        def get_dlondx(lat) : return(360/(np.cos(np.deg2rad(lat))*4.0075017e7))
        dlondx = get_dlondx(MERRA2_data.coords['lat'].values)
        lat_spaces = np.diff(MERRA2_data.coords['lat'].values)
        lon_spaces = np.diff(MERRA2_data.coords['lon'].values)    
        assert(np.allclose(lat_spaces, 0.5, atol=0.01) and np.allclose(lon_spaces, 0.625, atol=0.05))
        dlondi = np.mean(lon_spaces)
        dlatdj = np.mean(lat_spaces)

        dudx = dudi/dlondi*dlondx[None, :, None]
        dvdy = dvdj/dlatdj*dlatdy

        div = dudx + dvdy
        MERRA2_data['sfc_div'] = (('time', 'lat', 'lon'), div)
        MERRA2_data['sfc_div'] = MERRA2_data['sfc_div'].assign_attrs(
                        {"long_name": "wind divergence 10m",
                         "units": "s**-1"})

        MERRA_subset = MERRA2_data[['H', 'PS', 'SST', 'U10M', 'V10M', 'LTS', 'EIS', 'dzdt', 'sfc_div', 'RH_700', 'T_700', 'WSPD_10M', 'div_700',
                                   'TQV', 'TQL', 'T2M', 'M',  'T_adv',
                                   'ISCCPALB', 'MDSCLDFRCLO', 'MDSH2OPATH', 'MDSCLDSZH20', 'MDSOPTHCKH2O']]
        
        
        for key, var in MERRA_subset.data_vars.items():
            print(key)
            print(var.dtype)
        
        comp = dict(zlib=True, complevel=2)
        MERRA_subset.to_netcdf(f'/home/disk/eos4/jkcm/Data/MERRA/measures/subset/2/MERRA2.unified_subset.{str(year)}{month:02}.nc', engine='h5netcdf', encoding={var: comp for var in MERRA_subset.data_vars})
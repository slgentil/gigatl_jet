#!/usr/bin/env python
# coding: utf-8
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib import cm
import matplotlib
import numpy as np
import xarray as xr
import pandas as pd
from xgcm import Grid
import dask.array as da

from definition.defPrincipal2 import *


lat_r = -35.

# Initialisation of pathway to data and liste
path = '/ccc/store/cont003/gen12051/gulaj/GIGATL3/GIGATL3_1h/'
filenames = path + pd.read_csv('liste3',header=None) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
filenames = filenames.values.flatten().tolist()


ds = xr.open_dataset(filenames[0], chunks={'time_counter': 1},
                     drop_variables=['time', 'sustr', 'svstr','salt','temp','w','bvf',
                                      'hc','h','Vtransform','sc_r','sc_w','Cs_r','Cs_w','angle','mask_rho',
                                      'pm','pn','Tcline','theta_s','theta_b','f',
                                      'lon_rho','lat_rho','lon_u','lat_u','lon_v','lat_v',
                                      'time_instant','time_instant_bounds'])
ds, grid = addGrille(ds)
lat, lon = findLatLonIndex(ds.isel(time_counter = 0), lat_r, 0) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


datasets = []
for f in filenames :
    ds = xr.open_dataset(f, chunks={'time_counter': 1},
                     drop_variables=['time', 'sustr', 'svstr','salt','temp','w','bvf',
                                      'hc','h','Vtransform','sc_r','sc_w','Cs_r','Cs_w','angle','mask_rho',
                                      'pm','pn','Tcline','theta_s','theta_b','f',
                                      'lon_rho','lat_rho','lon_u','lat_u','lon_v','lat_v',
                                      'time_instant','time_instant_bounds'])
    ds, grid = addGrille(ds)
    ds = ds.isel(x_u = slice(lon-1,lon+2,1),y_v = slice(lat-1,lat+2,1),
            y_rho = slice(lat-2,lat+2,1), x_rho = slice(lon-2,lon+2,1))
    datasets.append(ds)
ds = xr.concat(datasets, dim='time_counter', coords='minimal', compat='override')    

################################################################################
ds = ds.isel(time_counter = slice(0,None,8))
################################################################################
# Compute depth at rho point
ds2 = ds.isel(time_counter = 0)

z = get_z(ds2,zeta=ds2['zeta'],hgrid='r').compute()

# get u at rho point
[urot,vrot] = rotuv(ds)
lat, lon = findLatLonIndex(ds.isel(time_counter = 0), lat_r, 0) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
vrot = vrot.isel(y_rho=lon)
vrot = vrot.isel(x_rho=lat)
urot = urot.isel(y_rho=lon)
urot = urot.isel(x_rho=lat)
ds2 = vrot.to_dataset(name = 'v')
ds2['u'] = urot

ds2 = ds2.assign_coords(z=("s_rho",z.isel(y_rho= 2, x_rho= 2 )))
ds2

del ds2['eta_rho'],ds2['xi_rho'],ds2['xi_w'],ds2['eta_w']
    
##############################################################
ds2 = ds2.load()
valuePlot = ds2.u*100
matplotlib.rcParams.update({'font.size' :22})
plt.figure(figsize=(16,12), dpi=80)
valuePlot.plot.contourf(x = 'time_counter', y = 'z',
              vmax = 20,vmin = -20, levels = 41,cmap='RdBu_r')

plt.xlabel("Date")
plt.ylabel("Depth")
plt.title("u")
plt.savefig("Hov_u_"+str(abs(int(lat_r)))+".png")
##############################################################
valuePlot = ds2.v*100
plt.figure(figsize=(16,12), dpi=80)
valuePlot.plot.contourf(x = 'time_counter', y = 'z',
              vmax = 20,vmin = -20, levels = 41, cmap='RdBu_r')

plt.xlabel("Date")
plt.ylabel("Depth")
plt.title("v")
plt.savefig("Hov_v_"+str(abs(int(lat_r)))+".png")

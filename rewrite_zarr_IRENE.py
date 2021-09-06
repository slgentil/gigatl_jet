#!/usr/bin/env python
# coding: utf-8
import os, sys
import numpy as np

import xarray as xr
import pandas as pd
from xgcm import Grid
from matplotlib import pyplot as plt

# Initialisation of pathway to data and liste
path = '/ccc/store/cont003/gen12051/gulaj/GIGATL3/GIGATL3_1h/'
filenames = path + pd.read_csv('liste3',header=None) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
filenames = filenames.values.flatten().tolist()

datasets = []
for f in filenames:
    ds = xr.open_dataset(f, chunks={'time_counter': 1, 's_rho': 1},
                         drop_variables=['time', 'ubar', 'vbar', 'sustr', 'svstr','w','u','v',
                                         'temp','salt', 'hc','theta_s','theta_b',
                                         'Tcline','Vtransform','pm','pn','h','f','angle',
                                         'mask_rho','Cs_r','sc_r','Cs_w','sc_w','lon_rho',
                                         'lat_rho','lon_u','lat_u','lon_v','lat_v',
                                        'time_instant','time_instant_bounds'])
    datasets.append(ds)
ds = xr.concat(datasets, dim='time_counter', coords='minimal', compat='override')
ds = ds.mean(dim="time_counter")

# keep all variables but sustr,svstr,bvf,
#ds = ds.drop(['time', 'ubar', 'vbar', 'sustr', 'svstr', 'bvf'])
L = ds.dims['x_rho']
M = ds.dims['y_rho']
N = ds.dims['s_rho']

zarr_dir = '/ccc/scratch/cont003/gen12051/durandya/bvf/zarr/'
V = ['bvf','zeta']
for v in V:
    dv = ds[v].to_dataset()

    file_out = zarr_dir+'%s.zarr'%(v)
    print(file_out)
    print(dv)
    try:
        dv.to_zarr(file_out, mode='w')                    
    except:
        print('Failure')
        
    dsize = os.path.getsize(file_out)
    print('   data is %.1fMB ' %(dsize/1e6))


#!/usr/bin/python3 python
# coding: utf-8

import sys
import os
import numpy as np
import xarray as xr
import pandas as pd
from xgcm import Grid
import dask.array as da
import pyinterp
import datetime

import gc

from definition.defPrincipal import *
from definition.specificReduce import *


from dask.distributed import Client, LocalCluster
#
# Initialisation d'un cluster de 32 coeurs
cluster = LocalCluster(processes=False, n_workers=1, threads_per_worker=56)
client = Client(cluster)
client

# Initialisation of pathway to data and liste
path = '/home/datawork-lops-megatl/GIGATL6/GIGATL6_1h/HIS/'
filenames = path + pd.read_csv('liste',header=None) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
filenames = filenames.values.flatten().tolist()


# Initialize new cartesian grid for interpolation, Date to take in account
times = pd.date_range('2007-03-31 00:00:00', freq='4D', periods=1600) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print("times : ")
print(times)

# Depth and values of geographical aera coordinates in degree, resolution
depth_Ana=[00,-10,-50,-100,-200,-300,-400,-500,-600,-700,-800,-900,-1000,-1100,-1200,-1300,-1400,-1500,-2000,-2500,-3000,-4000,-5000]
xmax, xmin, ymax, ymin = (14.5,-75.5,24.5, -25.5)
km = 20.		# resolution

dx = km*0.01/1.11	# km*degree/km

x = np.arange(xmin, xmax, dx)
y = np.arange(ymin, ymax, dx)

#Open one file to get the Z-level
ds = xr.open_dataset(filenames[0], chunks={'time_counter': 1,'s_rho': 1},
                     drop_variables=['time', 'sustr', 'svstr','salt','temp','w','bvf',
                                     'hc','h','Vtransform','sc_r','sc_w','Cs_r','Cs_w','angle','mask_rho',
                                      'pm','pn','Tcline','theta_s','theta_b','f',
                                      'lon_rho','lat_rho','lon_u','lat_u','lon_v','lat_v'])

ds, grid = addGrille(ds)
ds = ds.isel(time_counter=0)
ds = ds.load()
z_r = get_z(ds,zeta=ds['zeta'],hgrid='r')

datasets2 = []
# For memory using, the output of a netCDF file is done all of 100 timestep
for d in depth_Ana : 
	month=1
	nDay=0
	datasets_V = []
	datasets_U = []
	datasets = []
	for f in filenames :
		ds = xr.open_dataset(f, chunks={'time_counter': 1,'s_rho': 1},
				         drop_variables=['time', 'sustr', 'svstr','salt','temp','w','bvf',
				                          'hc','h','Vtransform','sc_r','sc_w','Cs_r','Cs_w','angle','mask_rho',
				                          'pm','pn','Tcline','theta_s','theta_b','f',
				                          'lon_rho','lat_rho','lon_u','lat_u','lon_v','lat_v',
				                          'time_instant','time_instant_bounds'])

		ds, grid = addGrille(ds)
		
        
		for i in range(0,len(ds.time_counter)):
			print('nDay : ',ds.time_counter[i],'   times nDay',times[nDay])
			if times[nDay] == np.datetime64('2007-05-02 00:00:00') :   #traitement de la date manquante
				v_cart_2, u_cart_2 = v_cart, u_cart
				if d == 0 :
					v_cart, u_cart = cylTOcartSFC(ds,i,x,y)
				else :
					# z_r = get_z(ds[i],zeta=ds['zeta'][i],hgrid='r')
					v_cart, u_cart = cylTOcart(ds,i,d,z_r,x,y)
				v_cart, u_cart = (v_cart+v_cart_2)/2, (u_cart+u_cart_2)/2
				datasets_V.append(v_cart) 
				datasets_U.append(u_cart)
				nDay = nDay+1
			if ds.time_counter[i] == times[nDay] :
				if d == 0 :
					v_cart, u_cart = cylTOcartSFC(ds,i,x,y)
				else :
					# z_r = get_z(ds[i],zeta=ds['zeta'][i],hgrid='r')
					v_cart, u_cart = cylTOcart(ds,i,d,z_r,x,y)
				v_cart = v_cart.assign_coords(time=times[nDay])
				u_cart = u_cart.assign_coords(time=times[nDay])
				datasets_V.append(v_cart) 
				datasets_U.append(u_cart)

				nDay=nDay+1

	ds2 = stockage_cart(datasets_V,datasets_U,ymax,ymin,d)
	datasets2.append(ds2)

ds = xr.concat(datasets2, dim='z_rho', coords='minimal', compat='override')
ds = ds.transpose("time","z_rho","y_rho","x_rho")
ds.to_netcdf('explarge_final.nc')

cluster.close()



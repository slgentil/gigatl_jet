import numpy as np
import xarray as xr
import pandas as pd
import dask.array as da

import time

from definition.defPrincipal import *

deg2m = 111000

def stockage_cart(ds,ds2,ymax,ymin,d):
	ds = xr.concat(ds, dim='time')
	ds2 = xr.concat(ds2, dim='time')
	ds = ds.to_dataset(name='vCart')
	ds2 = ds2.to_dataset(name='uCart')

	ds3 = xr.merge([ds,ds2])
	# ds3 = ds3.rename_dims({'x_rho':'x','y_rho':'y'})
	# ds3.time.encoding['calendar'] = "gregorian"

	# xxx=(ds3.coords['x_rho']).astype('float32')
	# yyy=(ds3.coords['y_rho']).astype('float32')

	# ds3.coords['x_rho']=xxx
	# ds3.coords['y_rho']=yyy
	ds3.coords['z_rho']=d

	# yLevelMin=find_nearest(ds3.y_rho, ymin)
	# yLevelMax=find_nearest(ds3.y_rho, ymax)
	# ds3=ds3.isel(y=slice(yLevelMin,yLevelMax)) 
	return ds3

def cylTOcart(ds,i,d,z_r,x,y):
	ds2 = ds.isel(time_counter=i)
	[urot,vrot] = rotuv(ds2,'r')
	ds2['vrot'] = vrot
	ds2['urot'] = urot

	# Initialise grid at specific depth
	ds2 = ds2.load()
	depth = d
	vslice = slice2(ds2, ds2.vrot, z_r, depth=depth)
	uslice = slice2(ds2, ds2.urot, z_r, depth=depth)

	# Compute interpolation on new grid
	v_cart = rtree_xr(ds2, vslice, x, y)
	u_cart = rtree_xr(ds2, uslice, x, y)

	# v_cart = v_cart.fillna(0)
	# u_cart = u_cart.fillna(0)	
	return v_cart, u_cart

def cylTOcartSFC(ds,i,x,y):
	ds2 = ds.isel(time_counter=i)
	[urot,vrot] = rotuv(ds2,'r')
	ds2['vrot'] = vrot
	ds2['urot'] = urot

	ds2 = ds2.isel(s_rho=-1).load()

	# Initialise grid at specific depth
	vslice = ds2['vrot']
	uslice = ds2['urot'] 

	# Compute interpolation on new grid
	v_cart = rtree_xr(ds2, vslice, x, y)
	u_cart = rtree_xr(ds2, uslice, x, y)

	# v_cart = v_cart.fillna(0)
	# u_cart = u_cart.fillna(0)
	
	return v_cart, u_cart

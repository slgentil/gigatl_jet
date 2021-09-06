#!/usr/bin/env python
# coding: utf-8

from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib

import numpy as np
import xarray as xr
import pandas as pd
from xgcm import Grid
import dask.array as da

from definition.defPrincipal2 import *

# Initialisation of pathway to data and liste
path = '/home/datawork-lops-megatl/GIGATL6/GIGATL6_1h/HIS/'
path = '/ccc/store/cont003/gen12051/gulaj/GIGATL3/GIGATL3_1h/'

filenames = path + pd.read_csv('liste3',header=None) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
filenames = filenames.values.flatten().tolist()

matplotlib.rcParams.update({'font.size' :22})

datasets = []
ds = xr.open_dataset(filenames[0], chunks={'time_counter': 1,'s_rho': 1},
                     drop_variables=['time', 'sustr', 'svstr','salt','temp','w','bvf',
                                      'hc','h','Vtransform','sc_r','sc_w','Cs_r','Cs_w','angle','mask_rho',
                                      'pm','pn','Tcline','theta_s','theta_b','f',
                                      'lon_rho','lat_rho','lon_u','lat_u','lon_v','lat_v',
                                      'time_instant','time_instant_bounds'])
ds, grid = addGrille(ds)

ds = ds.sel(time_counter = np.datetime64('2010-07-01'), method = "nearest")

# Compute depth at rho point
z = get_z(ds,zeta=ds['zeta'],hgrid='r').compute()

# get u at rho point
[urot,vrot] = rotuv(ds)
u=urot.compute()

rho=ds.rho+1027.4-1000
rho=rho.compute()

x= np.arange(-7,7.1,1)
matplotlib.rcParams.update({'font.size' :22})
longitude = -35.
vnew= slice2(ds,u,z,longitude=longitude)*100
vnew2= slice2(ds,rho,z,longitude=longitude)
plt.figure(figsize=(16,12), dpi=80)

cmap = colors.ListedColormap(['black', 'midnightblue','darkblue', 'royalblue','cornflowerblue','lightblue', 'lightgreen', 'beige', 'yellow', 'orange','tomato','tomato','red'])
boundaries = np.linspace(-0.3, 0.3, 13)*100
norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

vnew.plot.contourf(x=[s for s in vnew.coords if s in ["eta_rho","eta_u","eta_v"]][0],
          y=[s for s in vnew.coords if "z" in s][0],
          xlim=[-7.,7.], 
          ylim=[-5000.,0.],
          #vmin=0.,
          cmap=cmap, norm=norm,
          cbar_kwargs=dict(orientation='vertical',
                      pad=0.15, shrink=1, label='$cm.s^{-1}$')
         )

vnew.plot.contour(x=[s for s in vnew.coords if s in ["eta_rho","eta_u","eta_v"]][0],
          y=[s for s in vnew.coords if "z" in s][0],levels=boundaries,colors='grey')


contour=vnew2.plot.contour(x=[s for s in vnew.coords if s in ["eta_rho","eta_u","eta_v"]][0],
          y=[s for s in vnew.coords if "z" in s][0],levels=np.linspace(25,27.4,7),colors='black')

manual_locations = [(0, -1.4), (0, -0.7), (0, -100), (0, -200), (0., -500), (0., -1000.)]

plt.clabel(contour, fontsize=12,fmt = '%2.1f',inline=True,manual=manual_locations)
ax = plt.gca()
ax.set_facecolor('darkkhaki')

axisx=np.linspace(-7, 7, 15)
ax.xaxis.set_ticks(axisx)
plt.axvline(x=-0.,color='black',linewidth=0.5)
plt.grid()
plt.text(4., -4250, str(int(abs(longitude)))+' ° W',fontweight = 'bold', fontsize = 20)
plt.text(-6., -4250, np.datetime_as_string(vnew.time_counter.values, unit='D'),fontweight = 'bold', fontsize = 20)
# plt.text(9.25, -2400, '$m.s^{-1}$', fontsize = 10, rotation='vertical')

plt.title("")
plt.xlabel("Latitude")
plt.ylabel("Depth")

plt.savefig("test1.png")


longitude = -23.
vnew= slice2(ds,u,z,longitude=longitude)*100
vnew2= slice2(ds,rho,z,longitude=longitude)
plt.figure(figsize=(16,12), dpi=80)

cmap = colors.ListedColormap(['black', 'midnightblue','darkblue', 'royalblue','cornflowerblue','lightblue', 'lightgreen', 'beige', 'yellow', 'orange','tomato','tomato','red'])
boundaries = np.linspace(-0.3, 0.3, 13)*100
norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

vnew.plot.contourf(x=[s for s in vnew.coords if s in ["eta_rho","eta_u","eta_v"]][0],
          y=[s for s in vnew.coords if "z" in s][0],
          xlim=[-7,7], 
          ylim=[-5000.,0.],
          #vmin=0.,
          cmap=cmap, norm=norm,
          cbar_kwargs=dict(orientation='vertical',
                      pad=0.15, shrink=1, label='$cm.s^{-1}$')
         )

vnew.plot.contour(x=[s for s in vnew.coords if s in ["eta_rho","eta_u","eta_v"]][0],
          y=[s for s in vnew.coords if "z" in s][0],levels=boundaries,colors='grey')


contour=vnew2.plot.contour(x=[s for s in vnew.coords if s in ["eta_rho","eta_u","eta_v"]][0],
          y=[s for s in vnew.coords if "z" in s][0],levels=np.linspace(25,27.4,7),colors='black')

manual_locations = [(0, -1.4), (0, -0.7), (0, -100), (0, -200), (0., -500), (0., -1000.)]

plt.clabel(contour, fontsize=12,fmt = '%2.1f',inline=True,manual=manual_locations)
ax = plt.gca()
ax.set_facecolor('darkkhaki')

axisx=np.linspace(-7, 7, 15)
ax.xaxis.set_ticks(axisx)
plt.axvline(x=-0.,color='black',linewidth=0.5)
plt.grid()
plt.text(4., -4250, str(int(abs(longitude)))+' ° W',fontweight = 'bold', fontsize = 20)
plt.text(-6., -4250, np.datetime_as_string(vnew.time_counter.values, unit='D'),fontweight = 'bold', fontsize = 20)
# plt.text(9.25, -2400, '$m.s^{-1}$', fontsize = 10, rotation='vertical')

plt.title("")
plt.xlabel("Latitude")
plt.ylabel("Depth")

plt.savefig("test2.png")

longitude = -10.
vnew= slice2(ds,u,z,longitude=longitude)*100
vnew2= slice2(ds,rho,z,longitude=longitude)
plt.figure(figsize=(16,12), dpi=80)

cmap = colors.ListedColormap(['black', 'midnightblue','darkblue', 'royalblue','cornflowerblue','lightblue', 'lightgreen', 'beige', 'yellow', 'orange','tomato','tomato','red'])
boundaries = np.linspace(-0.3, 0.3, 13)*100
norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

vnew.plot.contourf(x=[s for s in vnew.coords if s in ["eta_rho","eta_u","eta_v"]][0],
          y=[s for s in vnew.coords if "z" in s][0],
          xlim=[-7.,7.], 
          ylim=[-5000.,0.],
          #vmin=0.,
          cmap=cmap, norm=norm,
          cbar_kwargs=dict(orientation='vertical',
                      pad=0.15, shrink=1, label='$cm.s^{-1}$')
         )

vnew.plot.contour(x=[s for s in vnew.coords if s in ["eta_rho","eta_u","eta_v"]][0],
          y=[s for s in vnew.coords if "z" in s][0],levels=boundaries,colors='grey')


contour=vnew2.plot.contour(x=[s for s in vnew.coords if s in ["eta_rho","eta_u","eta_v"]][0],
          y=[s for s in vnew.coords if "z" in s][0],levels=np.linspace(25,27.4,7),colors='black')

manual_locations = [(0, -1.4), (0, -0.7), (0, -100), (0, -200), (0., -500), (0., -1000.)]

plt.clabel(contour, fontsize=12,fmt = '%2.1f',inline=True,manual=manual_locations)
ax = plt.gca()
ax.set_facecolor('darkkhaki')

axisx=np.linspace(-7, 7, 15)
ax.xaxis.set_ticks(axisx)
plt.axvline(x=-0.,color='black',linewidth=0.5)
plt.grid()
plt.text(4., -4250, str(int(abs(longitude)))+' ° W',fontweight = 'bold', fontsize = 20)
plt.text(-6., -4250, np.datetime_as_string(vnew.time_counter.values, unit='D'),fontweight = 'bold', fontsize = 20)
# plt.text(9.25, -2400, '$m.s^{-1}$', fontsize = 10, rotation='vertical')

plt.title("")
plt.xlabel("Latitude")
plt.ylabel("Depth")

plt.savefig("test3.png")

cluster.close()

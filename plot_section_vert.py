from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib
matplotlib.use('AGG')


import numpy as np
import xarray as xr
import pandas as pd
from xgcm import Grid
import dask.array as da

import gridop as gop


#---------------------- Parameters -----------------------
# Initialisation of pathway to data and liste
path = '/home/datawork-lops-megatl/GIGATL6/GIGATL6_1h/HIS/'
path = '/ccc/store/cont003/gen12051/gulaj/GIGATL3/GIGATL3_1h/'

#filenames = path + pd.read_csv('liste3',header=None) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#filenames = filenames.values.flatten().tolist()
filename = path+'nomfichier.nc'
gridname = 'path/grid_file.nc'

longitudes = [-35, -23, -10]
date = '2010-07-01'

#---------------------- End parameters -----------------------


ds = xr.open_dataset(filename, chunks={'time_counter': 1,'s_rho': 1},
                     drop_variables=['time', 'sustr', 'svstr','salt','temp','w','bvf',
                                     'hc','h','Vtransform','sc_r','sc_w',
                                     'Cs_r','Cs_w','angle','mask_rho',
                                     'pm','pn','Tcline','theta_s','theta_b','f',
                                     'lon_rho','lat_rho','lon_u','lat_u','lon_v','lat_v',
                                     'time_instant','time_instant_bounds'])
ds, grid = addGrid(ds, gridname)

ds = ds.sel(time_counter = np.datetime64(date), method = "nearest")

for longitude in longitudes:
    # compute u_EW and v_SN at rho point
    [urot,vrot] = rotuv(ds)

    rho=ds.rho+1027.4-1000

    x= np.arange(-7,7.1,1)
    uplot= slices(ds,urot,urot.z_r,longitude=longitude)
    rhoplot= slices(ds,rho,rho.z_r,longitude=longitude)
    plt.figure(figsize=(16,12), dpi=80)

    cmap = colors.ListedColormap(['black', 'midnightblue','darkblue', 
                                  'royalblue','cornflowerblue','lightblue', 'lightgreen', 
                                  'beige', 'yellow', 'orange','tomato','tomato','red'])
    boundaries = np.linspace(-0.3, 0.3, 13)
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    coords = gop.get_spatial_coords(uplot)
    uplot.plot.contourf(x=coords['lat'],
              y=coords['z'],
              xlim=[-7.,7.], 
              ylim=[-5000.,0.],
              #vmin=0.,
              cmap=cmap, norm=norm,
              cbar_kwargs=dict(orientation='vertical',
                          pad=0.15, shrink=1, label='$cm.s^{-1}$')
             )

    uplot.plot.contour(x=coords['lat'], y=coords['z'],levels=boundaries,colors='grey')

    coords = gop.get_spatial_coords(rhoplot)
    contour=rhoplot.plot.contour(x=coords['lat'], y=coords['z'],
                                 levels=np.linspace(25,27.4,7),colors='black')

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

    EW='W' if longitude<0 else 'E'
    namefig = "u_"+str(abs(longitude))+EW+'_'+date+'.png'
    plt.savefig(namefig)




cluster.close()
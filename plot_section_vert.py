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
#path = '/home/datawork-lops-megatl/GIGATL6/GIGATL6_1h/HIS/'
path = '/ccc/store/cont003/gen12051/gulaj/GIGATL3/GIGATL3_1h/'
#path = '/ccc/store/cont003/gen12051/gulaj/GIGATL3/GIGATL3_1h_tides/'

#filenames = path + pd.read_csv('liste3',header=None) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#filenames = filenames.values.flatten().tolist()

#filename = path+'GIGATL3_12h_inst_2012-08-25-2012-08-29.nc'
filename = path+'GIGATL3_5d_aver_2012-07-21-2012-07-25.nc'

#filename = path+'GIGATL3_5d_aver_2011-07-02-2011-07-06.nc'  # pb champ que pour les plus hautes latitudes?
#filename = path+'GIGATL3_5d_aver_2009-07-02-2009-07-06.nc'  # idem?
#filename = path+'GIGATL3_3h_inst_2011-07-17-2011-07-21.nc'


gridname = '/ccc/store/cont003/gen12051/gulaj/GIGATL3/GIGATL3_1h_UP5/gigatl3_grid.nc'

longitudes = [-35, -23, -10]
date = '2012-07-21'

#---------------------- End parameters -----------------------

if __name__ == "__main__":
    
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster()  # these may not be hardcoded
    client = Client(cluster)
        
    ds = xr.open_dataset(filename, chunks={'time_counter': 1,'s_rho': 1},
                         drop_variables=['time', 'sustr', 'svstr','salt','temp','w','bvf',
                                         'hc','h','Vtransform','sc_r','sc_w',
                                         'Cs_r','Cs_w','angle','mask_rho',
                                         'pm','pn','Tcline','theta_s','theta_b','f',
                                         'lon_rho','lat_rho','lon_u','lat_u','lon_v','lat_v',
                                         'time_instant','time_instant_bounds'])

    ds, grid = gop.addGrid(ds, gridname, grid_metrics=1)

#     ds = ds.sel(time_counter = np.datetime64(date), method = "nearest")
    ds = ds.sel(time_counter = date, method = "nearest")

    z_r = gop.get_z(ds, zeta=ds.zeta, xgrid=grid, hgrid='r').fillna(0.)

    for longitude in longitudes:

        # compute slice of u at rho point
        uslice = gop.x2x(ds, ds.u, grid,'r')
        uslice = uslice.chunk({'x_rho':-1})
        uslice= gop.slices(ds,uslice,z_r,longitude=longitude)

        # compute slice of v at rho point
        vslice = gop.x2x(ds, ds.v, grid,'r')
        vslice = vslice.chunk({'x_rho':-1})
        vslice= gop.slices(ds,vslice,z_r,longitude=longitude)

        # compute slice of angle
        aslice= gop.slices(ds,ds.angle,z_r,longitude=longitude)

        # compute u_EW and v_NS at rho point
        [uplot,vplot] = rotuv(ds, u=uslice, v=vslice, angle=aslice)

        # compute slice of rho
        rho=ds.rho+1027.4-1000
        rhoplot= gop.slices(ds,rho,z_r,longitude=longitude)

        rho=ds.rho+1027.4-1000

        x= np.arange(-7,7.1,1)
        plt.figure(figsize=(16,12), dpi=80)
    
        cmap = colors.ListedColormap(['black', 'darkblue','mediumblue', 
                                      'royalblue','cornflowerblue','lightblue', 'lightgreen', 
                                      'beige', 'yellow', 'orange','tomato','red','firebrick','maroon'])
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
        plt.text(-6., -4250, np.datetime_as_string(uplot.time_counter.values, unit='D'),fontweight = 'bold', fontsize = 20)
        # plt.text(9.25, -2400, '$m.s^{-1}$', fontsize = 10, rotation='vertical')

        plt.title("")
        plt.xlabel("Latitude")
        plt.ylabel("Depth")

        EW='W' if longitude<0 else 'E'
        namefig = "u_"+str(abs(longitude))+EW+'_'+date+'.png'
        plt.savefig(namefig)



    cluster.close()

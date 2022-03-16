#Audrey D.
# March 2018

import numpy as np 

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

import matplotlib.colors as colors



def map_contourf_pacific_tropical(latitudes,longitudes,MAT,**kwargs):

    options = {
        'cmap'   : plt.cm.jet,
        'vmin'   : np.nanmin(MAT),
        'vmax'   : np.nanmax(MAT),
        'vstep'  : (np.nanmax(MAT)-np.nanmin(MAT))/10,
        'cbar_title': '',
        'title': '',
        'hold_on':False,
        'display':True,
        'save' : False,
        'save_path' : './out.png',
        }

    options.update(kwargs)
    
    # if not(options['hold_on']):
    fig = plt.figure(figsize=(18*2,4*2))

    # llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
    # are the lat/lon values of the lower left and upper right corners
    # of the map.
    # resolution = 'i' means use intermediate resolution coastlines.
    # lon_0, lat_0 are the central longitude and latitude of the projection.
    m = Basemap(llcrnrlon=120,llcrnrlat=-20.1,urcrnrlon=290,urcrnrlat=20.1,
                resolution='i',projection='merc',lat_ts=0,)#lon_0=220,lat_0=0)
    # design colors
    m.drawcoastlines()
    m.fillcontinents(color='grey',lake_color='white')
    m.drawmapboundary(fill_color='white')
    # draw parallels and meridians. labels = [left,right,top,bottom]
    m.drawparallels(np.arange(-60,60.,5.),labels=[True,False,False,False],fontsize=12)
    m.drawmeridians(np.arange(-180.,180.,10.),labels=[False,False,False,True],fontsize=12)

    # compute native map projection coordinates of lat/lon grid.

    LON,LAT=np.meshgrid(longitudes,latitudes)
    X,Y=m(LON,LAT)
    
    #m.contourf(X,Y,MAT,levels=np.arange(0,options['vmax'],options['vstep']),cmap=options['cmap'],extend='max')

    #cbar=plt.colorbar(ticks=np.arange(0,options['vmax']+10*options['vstep'],10*options['vstep']))

    m.contourf(X,Y,MAT, range(-16,16,9),cmap=options['cmap'] )#vmin= options['vmin'],vmax=options['vmax'], levels=9,cmap=options['cmap'])

    # cbar=plt.colorbar(ticks=np.arange(options['vmin'],options['vmax'],options['vstep']))


    cbar.ax.set_ylabel(options['cbar_title'],size=13)

    plt.title(options['title'],size=15)

    if options['save']:
        plt.savefig(options['save_path'])

    if options['display']:
        plt.show()






def map_contourf_atlantic_tropical(latitudes,longitudes,MAT,**kwargs):

    options = {
        'cmap'   : plt.cm.jet,
        'vmin'   : np.nanmin(MAT),
        'vmax'   : np.nanmax(MAT),
        'vstep'  : (np.nanmax(MAT)-np.nanmin(MAT))/10,
        'cbar_title': '',
        'title': '',
        'hold_on':False,
        'display':True,
        'outfile':'./out.png',
        }

    options.update(kwargs)

    #if not(options['hold_on']):
    fig = plt.figure(figsize=(12*2,4*2))

    # llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
    # are the lat/lon values of the lower left and upper right corners
    # of the map.
    # resolution = 'i' means use intermediate resolution coastlines.
    # lon_0, lat_0 are the central longitude and latitude of the projection.
    m = Basemap(llcrnrlon=-70,llcrnrlat=-25,urcrnrlon=15,urcrnrlat=25.1,
                resolution='i',projection='merc',lat_ts=0,)#lon_0=220,lat_0=0)
    # design colors
    m.drawcoastlines()
    m.fillcontinents(color='grey',lake_color='white')
    m.drawmapboundary(fill_color='white')
    # draw parallels and meridians. labels = [left,right,top,bottom]
    m.drawparallels(np.arange(-60,60.,5.),labels=[True,False,False,False])#,fontsize=12)
    m.drawmeridians(np.arange(-180.,180.,10.),labels=[False,False,False,True])#,fontsize=12)

    # compute native map projection coordinates of lat/lon grid.

    LON,LAT=np.meshgrid(longitudes,latitudes)
    X,Y=m(LON,LAT)
    
    m.contourf(X,Y,MAT,levels=np.arange(options['vmin'],options['vmax'],options['vstep']),cmap=options['cmap'],extend='both')
    levels_cont=np.linspace(0, 40, 9)

    
    cbar=plt.colorbar(ticks=np.arange(options['vmin'],options['vmax'],options['vstep']*2))
    m.contour(X,Y,MAT,levels=levels_cont,colors='k')


    cbar.ax.set_title(options['cbar_title'])#,size=13)

    plt.title(options['title'],size=15)

    if options['display']:
        plt.savefig(options['outfile'])
        plt.show()





def map_contourf_indian_tropical(latitudes,longitudes,MAT,**kwargs):

    options = {
        'cmap'   : plt.cm.jet,
        'vmin'   : np.nanmin(MAT),
        'vmax'   : np.nanmax(MAT),
        'vstep'  : (np.nanmax(MAT)-np.nanmin(MAT))/10,
        'cbar_title': '',
        'title': '',
        'hold_on':False,
        'display':True,
        'outfile':'./out.png'
        }

    options.update(kwargs)

    if not(options['hold_on']):
        fig = plt.figure(figsize=(12,4))

    # llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
    # are the lat/lon values of the lower left and upper right corners
    # of the map.
    # resolution = 'i' means use intermediate resolution coastlines.
    # lon_0, lat_0 are the central longitude and latitude of the projection.
    m = Basemap(llcrnrlon=35,llcrnrlat=-20,urcrnrlon=120,urcrnrlat=20,
                resolution='i',projection='merc',lat_ts=0,)#lon_0=220,lat_0=0)
    # design colors
    m.drawcoastlines()
    m.fillcontinents(color='grey',lake_color='white')
    m.drawmapboundary(fill_color='white')
    # draw parallels and meridians. labels = [left,right,top,bottom]
    m.drawparallels(np.arange(-60,60.,5.),labels=[True,False,False,False],fontsize=12)
    m.drawmeridians(np.arange(-180.,180.,10.),labels=[False,False,False,True],fontsize=12)

    # compute native map projection coordinates of lat/lon grid.

    LON,LAT=np.meshgrid(longitudes,latitudes)
    X,Y=m(LON,LAT)
    
    m.contourf(X,Y,MAT,levels=np.arange(options['vmin']-10*options['vstep'],options['vmax']+10*options['vstep'],options['vstep']),cmap=options['cmap'],extend='both')

    cbar=plt.colorbar(ticks=np.arange(options['vmin'],options['vmax']+20*options['vstep'],10*options['vstep']))

    cbar.ax.set_title(options['cbar_title'],size=13)

    plt.title(options['title'],size=15)

    if options['display']:
        plt.savefig(options['outfile'])
        plt.show()







def map_contours_pacific_tropical(latitudes,longitudes,MAT,**kwargs):

    options = {
        'cmap'   : plt.cm.jet,
        'vmin'   : np.nanmin(MAT),
        'vmax'   : np.nanmax(MAT),
        'vstep'  : (np.nanmax(MAT)-np.nanmin(MAT))/10,
        'cbar_title': '',
        'title': '',
        'hold_on':False,
        'display':True,
        'outfile':'./out.png',
        }

    options.update(kwargs)

    if not(options['hold_on']):
        fig = plt.figure(figsize=(18,4))

    # llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
    # are the lat/lon values of the lower left and upper right corners
    # of the map.
    # resolution = 'i' means use intermediate resolution coastlines.
    # lon_0, lat_0 are the central longitude and latitude of the projection.
    m = Basemap(llcrnrlon=120,llcrnrlat=-20,urcrnrlon=290,urcrnrlat=20,
                resolution='i',projection='merc',lat_ts=0,)#lon_0=220,lat_0=0)
    # design colors
    m.drawcoastlines()
    m.fillcontinents(color='grey',lake_color='white')
    m.drawmapboundary(fill_color='white')
    # draw parallels and meridians. labels = [left,right,top,bottom]
    m.drawparallels(np.arange(-60,60.,5.),labels=[True,False,False,False],fontsize=12)
    m.drawmeridians(np.arange(-180.,180.,10.),labels=[False,False,False,True],fontsize=12)

    # compute native map projection coordinates of lat/lon grid.

    LON,LAT=np.meshgrid(longitudes,latitudes)
    X,Y=m(LON,LAT)

    plt.contour(X,Y,MAT,levels=np.arange(options['vmin']-50,options['vmax']+50,5*options['vstep']),colors='k')

    m.contourf(X,Y,MAT,levels=np.arange(options['vmin']-10*options['vstep'],options['vmax']+10*options['vstep'],options['vstep']),cmap=options['cmap'],extend='both')

    cbar=plt.colorbar(ticks=np.arange(options['vmin'],options['vmax']+20*options['vstep'],10*options['vstep']))

    cbar.ax.tick_params(labelsize=13)
 
    cbar.ax.set_title(options['cbar_title'],size=13)

    plt.title(options['title'],size=15)

    if options['display']:
        plt.savefig(options['outfile'])
        plt.show()




def map_contours_pacific(latitudes,longitudes,MAT,**kwargs):

    options = {
        'lon_min': 120,
        'lon_max': 290,
        'lat_min': -50,
        'lat_max': 60,
        'cmap'   : plt.cm.jet,
        'vmin'   : np.nanmin(MAT),
        'vmax'   : np.nanmax(MAT),
        'vstep'  : (np.nanmax(MAT)-np.nanmin(MAT))/10,
        'cbar_title': '',
        'title': '',
        'hold_on':False,
        'display':True,
        'outfile':'./out.png',
        'quiver':False,
        'Xq':longitudes,
        'Yq':latitudes,
        'U' :MAT,
        'V' :MAT,
        }

    options.update(kwargs)

    if not(options['hold_on']):
        fig = plt.figure(figsize=(12,6))

    # llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
    # are the lat/lon values of the lower left and upper right corners
    # of the map.
    # resolution = 'i' means use intermediate resolution coastlines.
    # lon_0, lat_0 are the central longitude and latitude of the projection.
    m = Basemap(llcrnrlon=options['lon_min'],llcrnrlat=options['lat_min'],urcrnrlon=options['lon_max'],urcrnrlat=options['lat_max'],
                resolution='i',projection='merc',lat_ts=0,)#lon_0=220,lat_0=0)
    # design colors
    m.drawcoastlines()
    m.fillcontinents(color='lightgrey',lake_color='white')
    m.drawmapboundary(fill_color='white')
    # draw parallels and meridians. labels = [left,right,top,bottom]
    m.drawparallels(np.arange(-60,80.,20.),labels=[True,False,False,False],fontsize=12)
    m.drawmeridians(np.arange(-180.,180.,40.),labels=[False,False,False,True],fontsize=12)

    # compute native map projection coordinates of lat/lon grid.

    LON,LAT=np.meshgrid(longitudes,latitudes)
    X,Y=m(LON,LAT)

    #plt.contour(X,Y,MAT,levels=np.arange(options['vmin']-50,options['vmax']+50,2*options['vstep']),colors='k')

    m.contourf(X,Y,MAT,levels=np.arange(options['vmin']-10*options['vstep'],options['vmax']+10*options['vstep'],options['vstep']),cmap=options['cmap'],extend='both')
    if options['quiver']:
        m.quiver(options['Xq'],options['Yq'],options['U'],options['V'])

    cbar=plt.colorbar(ticks=np.arange(options['vmin'],options['vmax']+20*options['vstep'],10*options['vstep']))

    cbar.ax.tick_params(labelsize=13)
 
    cbar.ax.set_title(options['cbar_title'],size=13)

    plt.title(options['title'],size=15)

    if options['display']:
        plt.savefig(options['outfile'])
        plt.show()





def map_contourf_global(latitudes,longitudes,MAT,**kwargs):

    options = {
        'cmap'   : plt.cm.jet,
        'vmin'   : np.nanmin(MAT),
        'vmax'   : np.nanmax(MAT),
        'vstep'  : (np.nanmax(MAT)-np.nanmin(MAT))/10,
        'cbar_title': '',
        'title': '',
        'hold_on':False,
        'display':True,
        'save' : False,
        'save_path' : './out.png',
        }

    options.update(kwargs)

    if not(options['hold_on']):
        fig = plt.figure(figsize=(18,4))

    # llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
    # are the lat/lon values of the lower left and upper right corners
    # of the map.
    # resolution = 'i' means use intermediate resolution coastlines.
    # lon_0, lat_0 are the central longitude and latitude of the projection.
    m = Basemap(llcrnrlon=20,llcrnrlat=-40.1,urcrnrlon=380,urcrnrlat=40.1,
                resolution='c',projection='merc',lat_ts=0,)#lon_0=220,lat_0=0)
    # design colors
    m.drawcoastlines()
    m.fillcontinents(color='gray',lake_color='white')
    m.drawmapboundary(fill_color='white')
    # draw parallels and meridians. labels = [left,right,top,bottom]
    m.drawparallels(np.arange(-60,60.,20.),labels=[True,False,False,False],fontsize=10)
    m.drawmeridians(np.arange(-180.,180.,40.),labels=[False,False,False,True],fontsize=10)

    # compute native map projection coordinates of lat/lon grid.

    LON,LAT=np.meshgrid(longitudes,latitudes)
    X,Y=m(LON,LAT)
    
    #m.contourf(X,Y,MAT,levels=np.arange(0,options['vmax'],options['vstep']),cmap=options['cmap'],extend='max')

    #cbar=plt.colorbar(ticks=np.arange(0,options['vmax']+10*options['vstep'],10*options['vstep']))
    
    cmap = options['cmap']
    cmap.set_bad('lightgrey')

    m.contourf(X,Y,MAT,levels=np.arange(options['vmin'],options['vmax']+options['vstep'],options['vstep']),cmap=options['cmap'],extend='both')
    #m.contourf(X,Y,MAT,levels=np.arange(options['vmin'],options['vmax']+options['vstep'],options['vstep']),cmap=cmap,norm=colors.LogNorm(vmin=options['vmin']+0.2,vmax=options['vmax']),extend='both')

    cbar=plt.colorbar(ticks=np.arange(options['vmin'],options['vmax']+options['vstep'],2*options['vstep']))


    cbar.ax.set_ylabel(options['cbar_title'],size=13)

    plt.title(options['title'],size=15)

    if options['save']:
        plt.savefig(options['save_path'])

    if options['display']:
        plt.show()




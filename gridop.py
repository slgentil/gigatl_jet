from matplotlib import pyplot as plt
import matplotlib.colors as colors

import numpy as np
import xarray as xr
import pandas as pd
from xgcm import Grid
import dask.array as da
import pyinterp

import time
from collections import OrderedDict

from math import radians, cos, sin, asin, sqrt

def addGrid(ds, gridname, grid_metrics=0):
    gd = xr.open_dataset(gridname, chunks={'s_rho': 1})
    # hc in history file
    try: 
        ds['hc'] = gd.hc
    except:
        ds['hc'] = gd.attrs['hc']
    ds['h'] = gd.h
    # ds['Vtransform'] = gd.Vtransform
    ds['pm']   = gd.pm
    ds['pn']   = gd.pn
    ds['sc_r'] = gd.sc_r
    ds['sc_w'] = gd.sc_w
    ds['Cs_r'] = gd.Cs_r
    ds['Cs_w'] = gd.Cs_w
    ds['lon_rho'] = gd.lon_rho
    ds['lat_rho'] = gd.lat_rho
    ds['angle'] = gd.angle
    ds['mask_rho'] = gd.mask_rho

    # On modifie des dimensions et des coordonnées, on crée la grille xgc
    ds = adjust_grid(ds)

    # On crée la grille xgcm
    ds, grid = xgcm_grid(ds, grid_metrics=grid_metrics)
    
    return ds, grid


def xgcm_grid(ds,grid_metrics=0):
        
        # Create xgcm grid without metrics
        coords={'xi': {'center':'x_rho', 'inner':'x_u'}, 
                'eta': {'center':'y_rho', 'inner':'y_v'}, 
                's': {'center':'s_rho', 'outer':'s_w'}}
        grid = Grid(ds, 
                  coords=coords,
                  periodic=False,
                  boundary='extend')
        
        if grid_metrics==0:           
            ds.attrs['xgcm-Grid'] = grid
            return ds, grid
        
        # compute horizontal coordinates
        ds['lon_u'] = grid.interp(ds.lon_rho,'xi')
        ds['lat_u'] = grid.interp(ds.lat_rho,'xi')
        ds['lon_v'] = grid.interp(ds.lon_rho,'eta')
        ds['lat_v'] = grid.interp(ds.lat_rho,'eta')
        ds['lon_psi'] = grid.interp(ds.lon_v,'xi')
        ds['lat_psi'] = grid.interp(ds.lat_u,'eta')
        # set as coordinates in the dataset
        _coords = ['lon_u','lat_u',
                   'lon_v','lat_v',
                   'lon_psi','lat_psi',
                  ]
        ds = ds.set_coords(_coords)
        
        
        # add horizontal metrics for u, v and psi point
        if 'pm' in ds and 'pn' in ds:
            ds['dx_r'] = 1/ds['pm']
            ds['dy_r'] = 1/ds['pn']
        else: # backward compatibility, hack
            dlon = grid.interp(grid.diff(ds.lon_rho,'xi'),'xi')
            dlat =  grid.interp(grid.diff(ds.lat_rho,'eta'),'eta')
            ds['dx_r'], ds['dy_r'] = dll_dist(dlon, dlat, ds.lon_rho, ds.lat_rho)
        dlon = grid.interp(grid.diff(ds.lon_u,'xi'),'xi')
        dlat = grid.interp(grid.diff(ds.lat_u,'eta'),'eta')
        ds['dx_u'], ds['dy_u'] = dll_dist(dlon, dlat, ds.lon_u, ds.lat_u)
        dlon = grid.interp(grid.diff(ds.lon_v,'xi'),'xi')
        dlat = grid.interp(grid.diff(ds.lat_v,'eta'),'eta')
        ds['dx_v'], ds['dy_v'] = dll_dist(dlon, dlat, ds.lon_v, ds.lat_v)
        dlon = grid.interp(grid.diff(ds.lon_psi,'xi'),'xi')
        dlat = grid.interp(grid.diff(ds.lat_psi,'eta'),'eta')
        ds['dx_psi'], ds['dy_psi'] = dll_dist(dlon, dlat, ds.lon_psi, ds.lat_psi)

        # add areas metrics for rho,u,v and psi points
        ds['rAr'] = ds.dx_psi * ds.dy_psi
        ds['rAu'] = ds.dx_v * ds.dy_v
        ds['rAv'] = ds.dx_u * ds.dy_u
        ds['rAf'] = ds.dx_r * ds.dy_r

        
        # create new xgcmgrid with vertical metrics
        coords={'xi': {'center':'x_rho', 'inner':'x_u'}, 
                'eta': {'center':'y_rho', 'inner':'y_v'}}
        
        metrics = {
               ('xi',): ['dx_r', 'dx_u', 'dx_v', 'dx_psi'], # X distances
               ('eta',): ['dy_r', 'dy_u', 'dy_v', 'dy_psi'], # Y distances
               ('xi', 'eta'): ['rAr', 'rAu', 'rAv', 'rAf'] # Areas
              }
        
        if grid_metrics==1:
            # generate xgcm grid
            grid = Grid(ds,
                        coords=coords,
                        periodic=False,
                        metrics=metrics,
                        boundary='extend')
            ds.attrs['xgcm-Grid'] = grid
            return ds, grid
        
        # compute z coordinate at rho/w points
        if 'zeta' in [v for v in ds.data_vars] and \
           's_rho' in [d for d in ds.dims.keys()] and \
            ds['s_rho'].size>1:
            z_r = get_z(ds, zeta=ds.zeta, xgrid=grid).fillna(0.)
            z_w = get_z(ds, zeta=ds.zeta, xgrid=grid, vgrid='w').fillna(0.)
            ds['z_r'] = z_r
            ds['z_w'] = z_w
            ds['z_u'] = grid.interp(z_r,'xi')
            ds['z_v'] = grid.interp(z_r,'eta')
            ds['z_psi'] = grid.interp(ds.z_u,'eta')
            # set as coordinates in the dataset
            _coords = ['z_r','z_w','z_u','z_v','z_psi']
            ds = ds.set_coords(_coords)

        # add vertical metrics for u, v, rho and psi points
        if 'z_r' in [v for v in ds.coords]:
            ds['dz_r'] = grid.diff(ds.z_r,'s')
            ds['dz_w'] = grid.diff(ds.z_w,'s')
            ds['dz_u'] = grid.diff(ds.z_u,'s')
            ds['dz_v'] = grid.diff(ds.z_v,'s')
            ds['dz_psi'] = grid.diff(ds.z_psi,'s')
            
        if 'z_r' in ds:
            coords.update({'s': {'center':'s_rho', 'outer':'s_w'}})
        if 'z_r' in ds:
            metrics.update({('s',): ['dz_r', 'dz_u', 'dz_v', 'dz_psi', 'dz_w']}), # Z distances
        
        # generate xgcm grid
        grid = Grid(ds,
                    coords=coords,
                    periodic=False,
                    metrics=metrics,
                    boundary='extend')

        ds.attrs['xgcm-Grid'] = grid

        return ds, grid

def dll_dist(dlon, dlat, lon, lat):
    """
    Converts lat/lon differentials into distances in meters
    PARAMETERS
    ----------
    dlon : xarray.DataArray longitude differentials 
    dlat : xarray.DataArray latitude differentials 
    lon : xarray.DataArray longitude values
    lat : xarray.DataArray latitude values
    RETURNS
    -------
    dx : xarray.DataArray distance inferred from dlon 
    dy : xarray.DataArray distance inferred from dlat 
    """
    distance_1deg_equator = 111000.0
    dx = dlon * xr.ufuncs.cos(xr.ufuncs.deg2rad(lat)) * distance_1deg_equator 
    dy = ((lon * 0) + 1) * dlat * distance_1deg_equator
    return dx, dy



def adjust_grid(ds):
    # relevant to regular/analytical grid for now
    #
    #ds = ds.reset_coords([c for c in ds.coords if 'nav' in c])
    
    # rename redundant dimensions
    _dims = (d for d in ['x', 'y','x_v', 'y_u', 'x_w', 'y_w'] if d in ds.dims)
    for d in _dims:
        ds = ds.rename({d: d[0]+'_rho'})
        
    # change nav variables to coordinates        
    #_coords = [d for d in [d for d in ds.data_vars.keys()] if "nav_" in d]
    #ds = ds.set_coords(_coords) 
    _coords = [d for d in ds.data_vars.keys() if "lat_" in d]
    ds = ds.set_coords(_coords) 
    _coords = [d for d in ds.data_vars.keys() if "lon_" in d]
    ds = ds.set_coords(_coords) 
    
    # rename nav_lat/lon coordinates to lat/lon   
    #_coords = [c for c in ds.coords.keys() if "nav_" in c]   
    #for c in _coords:
    #    ds = ds.rename({c: c.replace('nav_','')})
    
    return ds
    


# Ajout des coordonnées au DataArray
def add_coords(ds, var, coords):
    for co in coords:
        var.coords[co] = ds.coords[co]



def get_spatial_dims(v):
    """ Return an ordered dict of spatial dimensions in the s/z, y, x order
    """
    dims = OrderedDict( (d, next((x for x in v.dims if x[0]==d), None))
                        for d in ['s','y','x'] )
    return dims

def get_spatial_coords(v):
    """ Return an ordered dict of spatial dimensions in the s/z, y, x order
    """
    coords = OrderedDict( (d, next((x for x in v.coords if x.startswith(d)), None))
                       for d in ['z','lat','y','lon','x'] )
    return coords



def x2rho(ds,v, grid):
    """ Interpolate from any grid to rho grid
    """
    dims = get_spatial_dims(v)
    coords = get_spatial_coords(v)
    vout = v.copy()
    if coords['z']: zout = v[coords['z']].copy()
    if dims['x'] == 'x_u':
        vout = grid.interp(vout, 'xi')
        if coords['z']: zout = grid.interp(zout, 'xi')
    if dims['y'] == 'y_v':
        vout = grid.interp(vout, 'eta')
        if coords['z']: zout = grid.interp(zout, 'eta')
    if dims['s'] == 's_w':
        vout = grid.interp(vout, 's')
        if coords['z']: zout = grid.interp(zout, 's')
    # assign coordinates
    vout = vout.assign_coords(coords={'lon_rho':ds.lon_rho})
    vout = vout.assign_coords(coords={'lat_rho':ds.lat_rho})
    if coords['z']:vout = vout.assign_coords(coords={'z_r':zout})

    return vout

def x2u(ds,v, grid):
    """ Interpolate from any grid to u grid
    """
    dims = get_spatial_dims(v)
    coords = get_spatial_coords(v)
    vout = v.copy()
    if coords['z']: zout = v[coords['z']].copy()
    if dims['x'] == 'x_rho':
        vout = grid.interp(vout, 'xi')
        if coords['z']: zout = grid.interp(zout, 'xi')
    if dims['y'] == 'y_v':
        vout = grid.interp(vout, 'eta')
        if coords['z']: zout = grid.interp(zout, 'eta')
    if dims['s'] == 's_w':
        vout = grid.interp(vout, 's')
        if coords['z']: zout = grid.interp(zout, 's')
    # assign coordinates
    vout = vout.assign_coords(coords={'lon_u':ds.lon_u})
    vout = vout.assign_coords(coords={'lat_u':ds.lat_u})
    if coords['z']:vout = vout.assign_coords(coords={'z_u':zout})
    return vout

def x2v(ds,v, grid):
    """ Interpolate from any grid to v grid
    """
    dims = get_spatial_dims(v)
    coords = get_spatial_coords(v)
    vout = v.copy()
    if coords['z']: zout = v[coords['z']].copy()
    if dims['x'] == 'x_u':
        vout = grid.interp(vout, 'xi')
        if coords['z']: zout = grid.interp(zout, 'xi')
    if dims['y'] == 'y_rho':
        vout = grid.interp(vout, 'eta')
        if coords['z']: zout = grid.interp(zout, 'eta')
    if dims['s'] == 's_w':
        vout = grid.interp(vout, 's')
        if coords['z']: zout = grid.interp(zout, 's')
    # assign coordinates
    vout = vout.assign_coords(coords={'lon_v':ds.lon_v})
    vout = vout.assign_coords(coords={'lat_v':ds.lat_v})
    if coords['z']:vout = vout.assign_coords(coords={'z_v':zout})
    return vout

def x2w(ds,v, grid):
    """ Interpolate from any grid to w grid
    """
    dims = get_spatial_dims(v)
    coords = get_spatial_coords(v)
    vout = v.copy()
    if coords['z']: zout = v[coords['z']].copy()
    if dims['x'] == 'x_u':
        vout = grid.interp(vout, 'xi')
        if coords['z']: zout = grid.interp(zout, 'xi')
    if dims['y'] == 'y_v':
        vout = grid.interp(vout, 'eta')
        if coords['z']: zout = grid.interp(zout, 'eta')
    if dims['s'] == 's_rho':
        vout = grid.interp(vout, 's')
        if coords['z']: zout = grid.interp(zout, 's')
    # assign coordinates
    vout = vout.assign_coords(coords={'lon_rho':ds.lon_rho})
    vout = vout.assign_coords(coords={'lat_rho':ds.lat_rho})
    if coords['z']:vout = vout.assign_coords(coords={'z_w':zout})
    return vout

def x2psi(ds,v, grid):
    """ Interpolate from any grid to psi grid
    """
    dims = get_spatial_dims(v)
    coords = get_spatial_coords(v)
    vout = v.copy()
    if coords['z']: zout = v[coords['z']].copy()
    if dims['x'] == 'x_rho':
        vout = grid.interp(vout, 'xi')
        if coords['z']: zout = grid.interp(zout, 'xi')
    if dims['y'] == 'y_rho':
        vout = grid.interp(vout, 'eta')
        if coords['z']: zout = grid.interp(zout, 'eta')
    if dims['s'] == 's_w':
        vout = grid.interp(vout, 's')
        if coords['z']: zout = grid.interp(zout, 's')
    # assign coordinates
    vout = vout.assign_coords(coords={'lon_psi':ds.lon_psi})
    vout = vout.assign_coords(coords={'lat_psi':ds.lat_psi})
    if coords['z']:vout = vout.assign_coords(coords={'z_psi':zout})
    return vout

def x2x(ds,v, grid, target):
    if target in ['rho', 'r']:
        return x2rho(ds,v, grid)
    elif target == 'u':
        return x2u(ds,v, grid)
    elif target == 'v':
        return x2v(ds,v, grid)
    elif target == 'w':
        return x2w(ds,v, grid)
    elif target in ['psi', 'p']:
        return x2psi(ds,v, grid)





def get_z(ds, zeta=None, h=None, xgrid=None, vgrid='r',
          hgrid='r', vtransform=2):
    ''' Compute vertical coordinates
        Spatial dimensions are placed last, in the order: s_rho/s_w, y, x

        Parameters
        ----------
        ds: xarray dataset
        zeta: xarray.DataArray, optional
            Sea level data, default to 0 if not provided
            If you use slices, make sure singleton dimensions are kept, i.e do:
                zeta.isel(x_rho=[i])
            and not :
                zeta.isel(x_rho=i)
        h: xarray.DataArray, optional
            Water depth, searche depth in grid if not provided
        vgrid: str, optional
            Vertical grid, 'r'/'rho' or 'w'. Default is 'rho'
        hgrid: str, optional
            Any horizontal grid: 'r'/'rho', 'u', 'v'. Default is 'rho'
        vtransform: int, str, optional
            croco vertical transform employed in the simulation.
            1="old": z = z0 + (1+z0/_h) * _zeta  with  z0 = hc*sc + (_h-hc)*cs
            2="new": z = z0 * (_zeta + _h) + _zeta  with  z0 = (hc * sc + _h * cs) / (hc + _h)
    '''

    xgrid = ds.attrs['xgcm-Grid'] if xgrid is None else xgrid

    h = ds.h if h is None else h
    zeta = 0*ds.h if zeta is None else zeta

    # switch horizontal grid if needed
    if hgrid in ['u','v']:
        h = x2x(ds, h, xgrid, hgrid)
        zeta = x2x(ds, zeta, xgrid, hgrid)

    # align datasets (zeta may contain a slice along one dimension for example)
    h, zeta  = xr.align(h, zeta, join='inner')

    if vgrid in ['r', 'rho']:
        vgrid = 'rho'
        sc = ds['sc_r']
        cs = ds['Cs_r']
    else:
        sc = ds['sc_'+vgrid]
        cs = ds['Cs_'+vgrid]

    hc = ds['hc']

    if vtransform == 1:
        z0 = hc*sc + (h-hc)*cs
        z = z0 + (1+z0/h) * zeta
    elif vtransform == 2:
        z0 = (hc * sc + h * cs) / (hc + h)
        z = z0 * (zeta + h) + zeta

    # reorder spatial dimensions and place them last
    sdims = list(get_spatial_dims(z).values())
    sdims = tuple(filter(None,sdims)) # delete None values
    reordered_dims = tuple(d for d in z.dims if d not in sdims) + sdims
    z = z.transpose(*reordered_dims, transpose_coords=True)

    return z.rename('z_'+vgrid)



def rotuv(ds, u=None, v=None, angle=None):
    '''
    Rotate winds or u,v to lat,lon coord -> result on rho grid by default
    '''

    import timeit
    
    xgrid = ds.attrs['xgcm-Grid']
        
    if u is None:
        u = ds.u 
        u = x2x(ds, u, xgrid, 'r')
        #u = ds_hor_chunk(u, wanted_chunk=100)
        
    if v is None:
        v = ds.v
        v = x2x(ds, v, xgrid, 'r')
        #v = ds_hor_chunk(v, wanted_chunk=100)
        
    angle = ds.angle if angle is None else angle
    
    cosang = np.cos(angle)
    sinang = np.sin(angle)

    # All the program statements
    urot = (u*cosang - v*sinang)
    #urot = da.multiply(u, cosang) - da.multiply(v, sinang)
    
    #start = timeit.default_timer()
    vrot = (u*sinang + v*cosang)
    #vrot = da.multiply(u, sinang) + da.multiply(v, cosang)
    #stop = timeit.default_timer()
    #print("time vrot: "+str(stop - start))
     
    coords = get_spatial_coords(u)
    if coords['z'] is not None: urot = urot.assign_coords(coords={'z_r':u[coords['z']]})
    coords = get_spatial_coords(v)
    if coords['z'] is not None: vrot = vrot.assign_coords(coords={'z_r':v[coords['z']]}) 

    return [urot,vrot]


def find_nearest_above(my_array, target, axis=0):
    diff = target - my_array
    diff = diff.where(diff>0,np.inf)
    return xr.DataArray(diff.argmin(axis=axis))


def findLatLonIndex(ds, lonValue, latValue):
    ''' Find nearest  grid point of  click value '''
    a = abs(ds['xi_rho'] - lonValue) + \
        abs(ds['eta_rho'] - latValue)
    return np.unravel_index(a.argmin(), a.shape)

def findLonIndex(ds, lonValue):
    ''' Find nearest  grid point of  click value '''
    a = abs(ds['xi_rho'] - lonValue)
    return np.unravel_index(a.argmin(), a.shape)

def findLatLonIndex2(ds, lonValue, latValue):
    ''' Find nearest  grid point of  click value '''
    a = abs(ds['x_rho'] - lonValue) + \
        abs(ds['y_rho'] - latValue)
    return np.unravel_index(a.argmin(), a.shape)

def findDepthIndex(z, depth):
        ''' Find nearest  grid point'''
        a = abs(z - depth)
        return xr.DataArray(a.argmin(dim='s_rho'))
    
def find_nearest(array, value):

    """
    Find item in array which is the closest to 
    a given values.
    """

    array = np.asarray(array)

    idx = (np.abs(array - value)).argmin()

    return idx
    
def get_grid_point(var):
    dims = var.dims
    if "x_u" in dims:
        if "y_rho" in dims:
            return 'u'
        else:
            return 'psi'
    elif "y_v" in dims:
        return 'v'
    else:
        if 's_rho' in dims:
            return 'rho'
        else:
            return 'w'
        
def slices(ds, var, z, longitude=None, latitude=None, depth=None):
    """
    #
    #
    # This function interpolate a 3D variable on slices at constant depths/longitude/latitude
    # This function use xcgm transform method and needs xgcm.Grid to be defined over the 3 axes.
    # !!! For now, it works only with curvilinear coordinates !!!
    #
    # On Input:
    #
    #    ds      dataset to find the grid
    #    var     (dataArray) Variable to process (3D matrix).
    #    z       (dataArray) Depths at the same point than var (3D matrix).
    #    longitude   (scalar,list or ndarray) longitude of the slice (scalar meters, negative).
    #    latitude    (scalar,list or ndarray) latitude of the slice (scalar meters, negative).
    #    depth       (scalar,list or ndarray) depth of the slice (scalar meters, negative).
    #
    # On Output:
    #
    #    vnew    (dataArray) Horizontal slice
    #
    #
    """
    from matplotlib.cbook import flatten

    xgrid = ds.attrs['xgcm-Grid']
    if longitude is None and latitude is None and depth is None:
        "Longitude or latitude or depth must be defined"
        return None

    # check typ of longitude/latitude/depth
    longitude = longitude.tolist() if isinstance(longitude,np.ndarray) else longitude
    longitude = [longitude] if (isinstance(longitude,int) or isinstance(longitude,float)) else longitude

    latitude = latitude.tolist() if isinstance(latitude,np.ndarray) else latitude
    latitude = [latitude] if (isinstance(latitude,int) or isinstance(latitude,float)) else latitude

    depth = depth.tolist() if isinstance(depth,np.ndarray) else depth
    depth = [depth] if (isinstance(depth,int) or isinstance(depth,float)) else depth

     # Find dimensions and coordinates of the variable
    dims = get_spatial_dims(var)
    coords = get_spatial_coords(var)
    hgrid = get_grid_point(var)

    if longitude is not None:
        axe = 'xi'
        coord_ref = coords['x'] if coords['x'] else coords['lon']
        coord_x = coords['y'] if coords['y'] else coords['lat']
        if dims['s'] is not None:
            coord_y = coords['z'] if coords['z'] is not None else 'z_'+hgrid[0]
        slices_values = longitude
    elif latitude is not None:
        axe = 'eta'
        coord_ref = coords['y'] if coords['y'] else coords['lat']
        coord_x = coords['x'] if coords['x'] else coords['lon']
        if dims['s'] is not None:
            coord_y = coords['z'] if coords['z'] is not None else 'z_'+hgrid[0]
        slices_values = latitude
    else:
        axe = 's'
        coord_ref = coords['z']
        coord_x = coords['x'] if coords['x'] else coords['lon']
        coord_y = coords['y'] if coords['y'] else coords['lat']
        slices_values = depth

    # Recursively loop over time if needed
    if len(var.squeeze().dims) == 4:
        vnew = [slices(ds, var.isel(time_counter=t), z.isel(time_counter=t),
                      longitude=longitude, latitude=latitude, depth=depth)
                      for t in range(len(var.time_counter))]
        vnew = xr.concat(vnew, dim='time_counter')
    else:
        vnew = xgrid.transform(var, axe, slices_values,
                               target_data=var[coord_ref]).squeeze()
    # Do the linear interpolation
    if not depth:
        x = xgrid.transform(var[coord_x], axe, slices_values,
                                   target_data=var[coord_ref]).squeeze() #\
                     #.expand_dims({dims['s']: len(var[dims['s']])})            
        vnew = vnew.assign_coords(coords={coord_x:x})

        #y = xgrid.transform(var[coord_y], axe, slices_values,
        if dims['s'] is not None:
            y = xgrid.transform(z, axe, slices_values,
                                       target_data=var[coord_ref]).squeeze()
            # Add the coordinates to dataArray
            vnew = vnew.assign_coords(coords={coord_y:y})
    else:
        # Add the coordinates to dataArray
        vnew = vnew.assign_coords(coords={coord_x:var[coord_x]})
        vnew = vnew.assign_coords(coords={coord_y:var[coord_y]})

    return vnew.squeeze().fillna(0.)  #unify_chunks()



def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

    
# ----------------------------- grid rechunk -----------------------------

def ds_hor_chunk(ds, keep_complete_axe=None, wanted_chunk=100):
    """
    Rechunk Dataset or DataArray such as each partition size is about 100Mb
    Input:
        - ds : (Dataset or DataArray) object to rechunk
        - keep_complete_axe : (character) Horizontal axe to keep with no chunk (x or y)
        - wanted_chunk : (integer) size of each partition in Mb
    Output:
        - object rechunked
    """
    
    #check input parameters
    if not isinstance(ds, (xr.Dataset,xr.DataArray)):
        print('argument must be a xarray.DataArray or xarray.Dataset')
        return
    if keep_complete_axe and keep_complete_axe!= 'x' and keep_complete_axe!='y':
        print('keep_complete_axe must equal x or y')
        return
    
    # get horizontal dimensions of the Dataset/DataArray
    chunks = {}
    hor_dim_names = {}
    if isinstance(ds,xr.Dataset):
        s_dim = max([ds.dims[s] for s in ds.dims.keys() if 's_' in s])
        for key,value in ds.dims.items():
            if 'x' in key or 'y' in key:
                hor_dim_names[key] = value
    else:
        s_dim = max([len(ds[s]) for s in ds.dims if 's_' in s])
        for key in ds.dims:
            if 'x' in key or 'y' in key:
                hor_dim_names[key] = len(ds[key])
                           
    if keep_complete_axe:
        # get the maximum length of the dimensions along the argument axe
        # set the chunks of those dimensions to -1 (no chunk)
        h_dim=[]
        for s in hor_dim_names.keys():
            if keep_complete_axe in s:
                h_dim.append(hor_dim_names[s])
                hor_dim_names[s] = -1
        h_dim = max(h_dim) 
        # compute the chunk on the dimensions along the other horizontal axe
        size_chunk = int(np.ceil(wanted_chunk*1.e6 / 4. / s_dim / h_dim))
    else : 
        # compute the chunk on all horizontal dimensions
        size_chunk = int(np.ceil(np.sqrt(wanted_chunk*1.e6 / 4. / s_dim )))
    
    # Initialize the dctionnary of chunks
    for s in hor_dim_names.keys():
        chunks[s]=min([size_chunk,hor_dim_names[s]])
        
    return ds.chunk(chunks)



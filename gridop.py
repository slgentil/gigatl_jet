from matplotlib import pyplot as plt
import matplotlib.colors as colors

import numpy as np
import xarray as xr
import pandas as pd
from xgcm import Grid
import dask.array as da
import pyinterp

import time
import itertools
from collections import OrderedDict

from math import radians, cos, sin, asin, sqrt

def open_files(model, gridname, filenames, 
               grid_metrics=0,
               drop_variables=[], chunks={'t':1},
              ):
                  
    # convert filenames to list of strings if string
    filenames = filenames.tolist() if isinstance(filenames,str) else filenames
    # find time dimension name
    concat_dim = [k for k,v in model.rename_vars.items() if v == 't'][0]
    open_kwargs = {'concat_dim': concat_dim,
                   'combine': 'nested',
                   'coords': 'minimal',
                   'parallel': False,
                   'compat': 'override'
                  }
    try : 
        ds = xr.open_zarr(filenames[0], drop_variables=drop_variables)
    except:
        try : 
            ds = xr.open_mfdataset(filenames, drop_variables=drop_variables, **open_kwargs)  
        except :
            print('open_files: unknown format: only Netcdf or Zarr')
            return
        
    model.ds = adjust_grid(model, ds)
    
    ds, grid = add_grid(model, gridname, grid_metrics=grid_metrics)
    model.ds = ds.chunk(chunks=chunks)
    return model.ds, grid

def add_grid(model, gridname, grid_metrics=0):
    # open grid file
    try : 
        gd = xr.open_zarr(gridname).squeeze()
    except:
        try : 
            gd = xr.open_dataset(gridname).squeeze() 
        except :
            print('add_grid: unknown format for grid : only Netcdf or Zarr')
            
    # Rename variable according model
    gd = adjust_grid(model, gd)
    
    # hc in history file
    try: 
        model.ds['hc'] = gd.hc
    except:
        model.ds['hc'] = gd.attrs['hc']
    model.ds['h'] = gd.h
    # ds['Vtransform'] = gd.Vtransform
    model.ds['pm']   = gd.pm
    model.ds['pn']   = gd.pn
    model.ds['sc_r'] = gd.sc_r
    model.ds['sc_w'] = gd.sc_w
    model.ds['Cs_r'] = gd.Cs_r
    model.ds['Cs_w'] = gd.Cs_w
    model.ds['angle'] = gd.angle
    model.ds['mask'] = gd.mask
    model.ds['lon'] = gd.lon
    model.ds['lat'] = gd.lat

    # On crée la grille xgcm
    ds, grid = xgcm_grid(model, grid_metrics=grid_metrics)
    
    return ds, grid

def xgcm_grid(model,grid_metrics=0):
        
        # Create xgcm grid without metrics
        coords={'x': {'center':'x', 'inner':'x_u'}, 
                'y': {'center':'y', 'inner':'y_v'}, 
                'z': {'center':'s', 'outer':'s_w'}}
        grid = Grid(model.ds, 
                  coords=coords,
                  periodic=False,
                  boundary='extend')
        
        if grid_metrics==0:           
            model.xgrid = grid
            return model.ds, grid
        
        # compute horizontal coordinates

        ds = model.ds
        if 'lon_u' not in ds: ds['lon_u'] = grid.interp(ds.lon,'x')
        if 'lat_u' not in ds: ds['lat_u'] = grid.interp(ds.lat,'x')
        if 'lon_v' not in ds: ds['lon_v'] = grid.interp(ds.lon,'y')
        if 'lat_v' not in ds: ds['lat_v'] = grid.interp(ds.lat,'y')
        if 'lon_f' not in ds: ds['lon_f'] = grid.interp(ds.lon_v,'x')
        if 'lat_f' not in ds: ds['lat_f'] = grid.interp(ds.lat_u,'y')
        _coords = [d for d in ds.data_vars.keys() if d.startswith(tuple(['lon','lat']))]
        ds = ds.set_coords(_coords)
        
        
        # add horizontal metrics for u, v and psi point
        if 'pm' in ds and 'pn' in ds:
            ds['dx'] = 1/ds['pm']
            ds['dy'] = 1/ds['pn']
        else: # backward compatibility, hack
            dlon = grid.interp(grid.diff(ds.lon,'x'),'x')
            dlat =  grid.interp(grid.diff(ds.lat,'y'),'y')
            ds['dx'], ds['dy'] = dll_dist(dlon, dlat, ds.lon, ds.lat)
        dlon = grid.interp(grid.diff(ds.lon_u,'x'),'x')
        dlat = grid.interp(grid.diff(ds.lat_u,'y'),'y')
        ds['dx_u'], ds['dy_u'] = dll_dist(dlon, dlat, ds.lon_u, ds.lat_u)
        dlon = grid.interp(grid.diff(ds.lon_v,'x'),'x')
        dlat = grid.interp(grid.diff(ds.lat_v,'y'),'y')
        ds['dx_v'], ds['dy_v'] = dll_dist(dlon, dlat, ds.lon_v, ds.lat_v)
        dlon = grid.interp(grid.diff(ds.lon_f,'x'),'x')
        dlat = grid.interp(grid.diff(ds.lat_f,'y'),'y')
        ds['dx_psi'], ds['dy_psi'] = dll_dist(dlon, dlat, ds.lon_f, ds.lat_f)

        # add areas metrics for rho,u,v and psi points
        ds['rAr'] = ds.dx_psi * ds.dy_psi
        ds['rAu'] = ds.dx_v * ds.dy_v
        ds['rAv'] = ds.dx_u * ds.dy_u
        ds['rAf'] = ds.dx * ds.dy

        
        # create new xgcmgrid with vertical metrics
#         coords={'x': {'center':'x', 'inner':'x_u'}, 
#                 'y': {'center':'y', 'inner':'y_v'}, 
#                 'z': {'center':'s', 'outer':'s_w'}}
        
        metrics = {
               ('x',): ['dx', 'dx_u', 'dx_v', 'dx_psi'], # X distances
               ('y',): ['dy', 'dy_u', 'dy_v', 'dy_psi'], # Y distances
               ('x', 'y'): ['rAr', 'rAu', 'rAv', 'rAf'] # Areas
              }
        
        if grid_metrics==1:
            # generate xgcm grid
            grid = Grid(ds,
                        coords=coords,
                        periodic=False,
                        metrics=metrics,
                        boundary='extend')
            model.xgrid = grid
            model.ds = ds
            return ds, grid
        
        # compute z coordinate at rho/w points
        if 'z_sfc' in [v for v in ds.data_vars] and \
           's' in [d for d in ds.dims.keys()] and \
            ds['s'].size>1:
            z = get_z(model, z_sfc=ds.z_sfc, xgrid=grid).fillna(0.)
            z_w = get_z(model, z_sfc=ds.z_sfc, xgrid=grid, vgrid='w').fillna(0.)
            ds['z'] = z
            ds['z_w'] = z_w
            ds['z_u'] = grid.interp(z,'x')
            ds['z_v'] = grid.interp(z,'y')
            ds['z_f'] = grid.interp(ds.z_u,'y')
            # set as coordinates in the dataset
            _coords = ['z','z_w','z_u','z_v','z_f']
            ds = ds.set_coords(_coords)

        # add vertical metrics for u, v, rho and psi points
        if 'z' in [v for v in ds.coords]:
            ds['dz'] = grid.diff(ds.z,'z')
            ds['dz_w'] = grid.diff(ds.z_w,'z')
            ds['dz_u'] = grid.diff(ds.z_u,'z')
            ds['dz_v'] = grid.diff(ds.z_v,'z')
            ds['dz_f'] = grid.diff(ds.z_f,'z')
            
        # add coords and metrics for xgcm for the vertical direction
        if 'z' in ds:
#             coords.update({'z': {'center':'s', 'outer':'s_w'}})
            metrics.update({('z',): ['dz', 'dz_u', 'dz_v', 'dz_f', 'dz_w']}), # Z distances
        
        # generate xgcm grid
        grid = Grid(ds,
                    coords=coords,
                    periodic=False,
                    metrics=metrics,
                    boundary='extend')

        model.xgrid = grid
        model.ds = ds

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



def adjust_grid(model, ds):
    
    for k,v in model.rename_vars.items():
        if (k in ds and v not in ds) or \
            k in ds.dims.keys():
            ds = ds.rename({k: v})
    return ds
    

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
                       for d in ['z','lat','lon'] )
    for k,c in coords.items():
        if c is not None and v.coords[c].size==1: coords[k]= None
    return coords




def x2rho(ds,v, grid):
    """ Interpolate from any grid to rho grid
    """
    dims = get_spatial_dims(v)
    coords = get_spatial_coords(v)
    vout = v.copy()
    if coords['lon']: lonout = v[coords['lon']].copy()
    if coords['lat']: latout = v[coords['lat']].copy()
    if coords['z']: zout = v[coords['z']].copy()
    if dims['x'] == 'x_u':
        vout = grid.interp(vout, 'x')
        if coords['lon']: lonout = grid.interp(lonout, 'x')
        if coords['lat']: latout = grid.interp(latout, 'x')
        if coords['z']: zout = grid.interp(zout, 'x')
    if dims['y'] == 'y_v':
        vout = grid.interp(vout, 'y')
        if coords['lon']: lonout = grid.interp(lonout, 'y')
        if coords['lat']: latout = grid.interp(latout, 'y')
        if coords['z']: zout = grid.interp(zout, 'y')
    if dims['s'] == 's_w':
        vout = grid.interp(vout, 'z')
        if coords['z']: zout = grid.interp(zout, 'z')
    # assign coordinates
    if coords['lon']: vout = vout.assign_coords(coords={'lon':lonout})
    if coords['lat']: vout = vout.assign_coords(coords={'lat':latout})
    if coords['z']: vout = vout.assign_coords(coords={'z':zout})

    return vout

def x2u(ds,v, grid):
    """ Interpolate from any grid to u grid
    """
    dims = get_spatial_dims(v)
    coords = get_spatial_coords(v)
    vout = v.copy()
    if coords['lon']: lonout = v[coords['lon']].copy()
    if coords['lat']: latout = v[coords['lat']].copy()
    if coords['z']: zout = v[coords['z']].copy()
    if dims['x'] == 'x':
        vout = grid.interp(vout, 'x')
        if coords['lon']: lonout = grid.interp(lonout, 'x')
        if coords['lat']: latout = grid.interp(latout, 'x')
        if coords['z']: zout = grid.interp(zout, 'x')
    if dims['y'] == 'y_v':
        vout = grid.interp(vout, 'y')
        if coords['lon']: lonout = grid.interp(lonout, 'y')
        if coords['lat']: latout = grid.interp(latout, 'y')
        if coords['z']: zout = grid.interp(zout, 'y')
    if dims['s'] == 's_w':
        vout = grid.interp(vout, 'z')
        if coords['z']: zout = grid.interp(zout, 'z')
    # assign coordinates
    if coords['lon']: vout = vout.assign_coords(coords={'lon_u':lonout})
    if coords['lat']: vout = vout.assign_coords(coords={'lat_u':latout})
    if coords['z']:vout = vout.assign_coords(coords={'z_u':zout})
    return vout

def x2v(ds,v, grid):
    """ Interpolate from any grid to v grid
    """
    dims = get_spatial_dims(v)
    coords = get_spatial_coords(v)
    vout = v.copy()
    if coords['lon']: lonout = v[coords['lon']].copy()
    if coords['lat']: latout = v[coords['lat']].copy()
    if coords['z']: zout = v[coords['z']].copy()
    if dims['x'] == 'x_u':
        vout = grid.interp(vout, 'x')
        if coords['lon']: lonout = grid.interp(lonout, 'x')
        if coords['lat']: latout = grid.interp(latout, 'x')
        if coords['z']: zout = grid.interp(zout, 'x')
    if dims['y'] == 'y':
        vout = grid.interp(vout, 'y')
        if coords['lon']: lonout = grid.interp(lonout, 'y')
        if coords['lat']: latout = grid.interp(latout, 'y')
        if coords['z']: zout = grid.interp(zout, 'y')
    if dims['s'] == 's_w':
        vout = grid.interp(vout, 'z')
        if coords['z']: zout = grid.interp(zout, 'z')
    # assign coordinates
    if coords['lon']: vout = vout.assign_coords(coords={'lon_v':lonout})
    if coords['lat']: vout = vout.assign_coords(coords={'lat_v':latout})
    if coords['z']:vout = vout.assign_coords(coords={'z_v':zout})
    return vout

def x2w(ds,v, grid):
    """ Interpolate from any grid to w grid
    """
    dims = get_spatial_dims(v)
    coords = get_spatial_coords(v)
    vout = v.copy()
    if coords['lon']: lonout = v[coords['lon']].copy()
    if coords['lat']: latout = v[coords['lat']].copy()
    if coords['z']: zout = v[coords['z']].copy()
    if dims['x'] == 'x_u':
        vout = grid.interp(vout, 'x')
        if coords['lon']: lonout = grid.interp(lonout, 'x')
        if coords['lat']: latout = grid.interp(latout, 'x')
        if coords['z']: zout = grid.interp(zout, 'x')
    if dims['y'] == 'y_v':
        vout = grid.interp(vout, 'y')
        if coords['lon']: lonout = grid.interp(lonout, 'y')
        if coords['lat']: latout = grid.interp(latout, 'y')
        if coords['z']: zout = grid.interp(zout, 'y')
    if dims['s'] == 's':
        vout = grid.interp(vout, 'z')
        if coords['z']: zout = grid.interp(zout, 'z')
    # assign coordinates
    if coords['lon']: vout = vout.assign_coords(coords={'lon':lonout})
    if coords['lat']: vout = vout.assign_coords(coords={'lat':latout})
    if coords['z']:vout = vout.assign_coords(coords={'z_w':zout})
    return vout


def x2f(ds,v, grid):
    """ Interpolate from any grid to psi grid
    """
    dims = get_spatial_dims(v)
    coords = get_spatial_coords(v)
    vout = v.copy()
    if coords['lon']: lonout = v[coords['lon']].copy()
    if coords['lat']: latout = v[coords['lat']].copy()
    if coords['z']: zout = v[coords['z']].copy()
    if dims['x'] == 'x':
        vout = grid.interp(vout, 'x')
        if coords['lon']: lonout = grid.interp(lonout, 'x')
        if coords['lat']: latout = grid.interp(latout, 'x')
        if coords['z']: zout = grid.interp(zout, 'x')
    if dims['y'] == 'y':
        vout = grid.interp(vout, 'y')
        if coords['lon']: lonout = grid.interp(lonout, 'y')
        if coords['lat']: latout = grid.interp(latout, 'y')
        if coords['z']: zout = grid.interp(zout, 'y')
    if dims['s'] == 's_w':
        vout = grid.interp(vout, 'z')
        if coords['z']: zout = grid.interp(zout, 'z')
    # assign coordinates
    if coords['lon']: vout = vout.assign_coords(coords={'lon_f':lonout})
    if coords['lat']: vout = vout.assign_coords(coords={'lat_f':latout})
    if coords['z']:vout = vout.assign_coords(coords={'z_f':zout})
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
    elif target in ['psi', 'p', 'f']:
        return x2psi(ds,v, grid)




def get_z(model, ds=None, z_sfc=None, h=None, xgrid=None, vgrid='r',
          hgrid='r', vtransform=2):
    ''' Compute vertical coordinates
        Spatial dimensions are placed last, in the order: s_rho/s_w, y, x

        Parameters
        ----------
        ds: xarray dataset
        z_sfc: xarray.DataArray, optional
            Sea level data, default to 0 if not provided
            If you use slices, make sure singleton dimensions are kept, i.e do:
                z_sfc.isel(x=[i])
            and not :
                z_sfc.isel(x=i)
        h: xarray.DataArray, optional
            Water depth, searche depth in grid if not provided
        vgrid: str, optional
            Vertical grid, 'r'/'rho' or 'w'. Default is 'rho'
        hgrid: str, optional
            Any horizontal grid: 'r'/'rho', 'u', 'v'. Default is 'rho'
        vtransform: int, str, optional
            croco vertical transform employed in the simulation.
            1="old": z = z0 + (1+z0/_h) * _z_sfc  with  z0 = hc*sc + (_h-hc)*cs
            2="new": z = z0 * (_z_sfc + _h) + _z_sfc  with  z0 = (hc * sc + _h * cs) / (hc + _h)
    '''

    xgrid = model.xgrid if xgrid is None else xgrid
    ds = model.ds if ds is None else ds

    h = ds.h if h is None else h
    z_sfc = 0*ds.h if z_sfc is None else z_sfc

    # switch horizontal grid if needed
    if hgrid in ['u','v']:
        h = x2x(ds, h, xgrid, hgrid)
        z_sfc = x2x(ds, z_sfc, xgrid, hgrid)

    # align datasets (z_sfc may contain a slice along one dimension for example)
    h, z_sfc  = xr.align(h, z_sfc, join='inner')

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
        z = z0 + (1+z0/h) * z_sfc
    elif vtransform == 2:
        z0 = (hc * sc + h * cs) / (hc + h)
        z = z0 * (z_sfc + h) + z_sfc

    # reorder spatial dimensions and place them last
    sdims = list(get_spatial_dims(z).values())
    sdims = tuple(filter(None,sdims)) # delete None values
    reordered_dims = tuple(d for d in z.dims if d not in sdims) + sdims
    z = z.transpose(*reordered_dims, transpose_coords=True)

    return z.fillna(0.).rename('z_'+vgrid)



def rotuv(model, ds=None, xgrid=None, u=None, v=None, angle=None):
    '''
    Rotate winds or u,v to lat,lon coord -> result on rho grid by default
    '''

    import timeit
    
    xgrid = model.xgrid if xgrid is None else xgrid
    ds = model.ds if ds is None else ds
        
    u = ds.u if u is None else u
    hgrid = get_grid_point(u)
    if hgrid != 'r': u = x2rho(ds, u, xgrid)
    #u = ds_hor_chunk(u, wanted_chunk=100)
        
    v = ds.v if v is None else v
    hgrid = get_grid_point(v)
    if hgrid != 'r': v = x2rho(ds, v, xgrid)
    #v = ds_hor_chunk(v, wanted_chunk=100)
        
    angle = ds.angle if angle is None else angle
    hgrid = get_grid_point(angle)
    if hgrid != 'r': angle = x2rho(ds, angle, xgrid)
    
    cosang = np.cos(angle)
    sinang = np.sin(angle)

    # All the program statements
    urot = (u*cosang - v*sinang)
    
    #start = timeit.default_timer()
    vrot = (u*sinang + v*cosang)
    #stop = timeit.default_timer()
    #print("time vrot: "+str(stop - start))
    
    # assign coordinates to urot/vrot
    dims = get_spatial_dims(u)
    coords = get_spatial_coords(u)
    for k,c in coords.items(): 
        if c is not None: urot = urot.assign_coords(coords={c:u[c]})
    dims = get_spatial_dims(v)
    coords = get_spatial_coords(v)
    for k,c in coords.items(): 
        if c is not None: vrot = vrot.assign_coords(coords={c:v[c]})

    return [urot,vrot]

    
def get_grid_point(var):
    dims = var.dims
    if "x_u" in dims:
        if "y_rho" in dims:
            return 'u'
        else:
            return 'f'
    elif "y_v" in dims:
        return 'v'
    else:
        if 's_rho' in dims:
            return 'r'
        else:
            return 'w'
        
def slices(model, var, z, ds=None, xgrid=None, longitude=None, latitude=None, depth=None):
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
  
    xgrid = model.xgrid if xgrid is None else xgrid
    ds = model.ds if ds is None else ds
    
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
    if dims['s'] is not None and coords['z'] is None: 
        var = var.assign_coords(coords={'z':z})
        coords = get_spatial_coords(var)
    hgrid = get_grid_point(var)

    if longitude is not None:
        axe = 'x'
        coord_ref = coords['lon']
        coord_x = coords['lat']
        coord_y = coords['z']
        slices_values = longitude
    elif latitude is not None:
        axe = 'y'
        coord_ref = coords['lat']
        coord_x = coords['lon']
        coord_y = coords['z']
        slices_values = latitude
    else:
        axe = 'z'
        coord_ref = coords['z']
        coord_x = coords['lon']
        coord_y = coords['lat']
        slices_values = depth

    # Recursively loop over time if needed
    if len(var.squeeze().dims) == 4:
        vnew = [slices(ds, var.isel(t=t), z.isel(t=t),
                      longitude=longitude, latitude=latitude, depth=depth)
                      for t in range(len(var.t))]
        vnew = xr.concat(vnew, dim='t')
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

#     return vnew.squeeze().unify_chunks().fillna(0.)  #unify_chunks() 
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

def auto_chunk(ds, keep_complete_dim=None, wanted_chunk=150):
    """
    Rechunk Dataset or DataArray such as each partition size is about 100Mb
    Input:
        - ds : (Dataset or DataArray) object to rechunk
        - keep_complete_dim : (character) Horizontal axe to keep with no chunk ('x','y','s')
        - wanted_chunk : (integer) size of each partition in Mb
    Output:
        - object rechunked
    """

    #check input parameters
    if not isinstance(ds, (xr.Dataset,xr.DataArray)):
        print('argument must be a xarray.DataArray or xarray.Dataset')
        return
    if keep_complete_dim and keep_complete_dim != 'x' \
                         and keep_complete_dim != 'y' \
                         and keep_complete_dim != 's':
        print('keep_complete_dim must equal x or y or s')
        return

    # get horizontal dimensions names of the Dataset/DataArray
    dname = get_spatial_dims(ds)
    chunks_name = dname.copy()
    
    # get horizontal dimensions sizes of the Dataset/DataArray
    chunks_size={}
    for k,v in chunks_name.items():
        chunks_size[k] = ds.sizes[v]

    # always chunk in time
    if 't' in ds.dims: chunks_size['t'] = 1
        
    if keep_complete_dim:
        # remove keep_complete_dim from the dimensions of the Dataset/DatAarray
        del chunks_name[keep_complete_dim]
        
        # reduce chunks size  beginning by 's' then 'y' then 'x' if necessary
        for k in chunks_name.keys():
            for d in range(chunks_size[k],0,-1):
                chunk_size = (chunks_size['x']*chunks_size['y']*chunks_size['s']*4 / 1.e6)        
                if chunk_size > wanted_chunk:
                    chunks_size[k] = d
                else:
                    break
            if chunk_size > wanted_chunk:
                break
    else : 
        
        # reduce chunks size  beginning by 's' then 'y' then 'x' if necessary
        for k in chunks_name.keys():
            for d in range(chunks_size[k],0,-1):
                chunk_size = (chunks_size['x']*chunks_size['y']*chunks_size['s']*4 / 1.e6)        
                if chunk_size > wanted_chunk:
                    chunks_size[k] = d
                else:
                    break
            if chunk_size > wanted_chunk:
                break
            
    if isinstance(ds,xr.Dataset):
        # set chunk for all the dimensions of the dataset (ie : x and x_u)
        for c in list(itertools.product(dname.keys(), ds.dims.keys())):
            if c[1].startswith(c[0]):
                chunks_size[c[1]] = chunks_size[c[0]]
    else:
        # rename the dimension name by the right values (ie: x_u instead of x)
        for key in dname.keys():
            chunks_size[dname[key]] = chunks_size.pop(key)

        
    return ds.chunk(chunks_size)



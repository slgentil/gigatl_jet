from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from xgcm import Grid

def adjust_grid(ds):
        # relevant to regular/analytical grid for now
        #
        ds = ds.reset_coords([c for c in ds.coords if 'nav' in c])

        # rename redundant dimensions
        _dims = (d for d in ['x_v', 'y_u', 'x_w', 'y_w'] if d in ds.dims)
        for d in _dims:
            ds = ds.rename({d: d[0]+'_rho'})

        # change nav variables to coordinates
        _coords = [d for d in [d for d in ds.data_vars.keys()] if "nav_" in d]
        ds = ds.set_coords(_coords)

        # rename coordinates
        eta_suff={}
        for c in ds.coords:
            new_c = c.replace('nav_lat','eta').replace('nav_lon','xi')
            ds = ds.rename({c:new_c})
            # reset names and units
            ds[new_c] = (ds[new_c].assign_attrs(units='m',
                                               standard_name=new_c,
                                               long_name=new_c)
                        )
        return ds

def xgcm_grid(ds):
        # Create xgcm grid
        coords={'xi':{'center':'x_rho', 'inner':'x_u'}, 
                'eta':{'center':'y_rho', 'inner':'y_v'}, 
                's':{'center':'s_rho', 'outer':'s_w'}}
        ds.attrs['xgcm-Grid'] = Grid(ds, coords=coords)
        
        return ds
    
# Ajout des coordonnées au DataArray
def add_coords(ds, var, coords):
    for co in coords:
        var.coords[co] = ds.coords[co]



# passage du DataArray du point rho au point u sur la grille C
def rho2u(v, ds):
    """
    interpolate horizontally variable from rho to u point
    """
    grid = ds.attrs['xgcm-Grid']
    var = grid.interp(v,'xi')
    add_coords(ds, var, ['xi_u','eta_u'])
    var.attrs = v.attrs
    return var.rename(v.name)

# passage du DataArray du point u au point rho sur la grille C
def u2rho(v, ds):
    """
    interpolate horizontally variable from u to rho point
    """
    grid = ds.attrs['xgcm-Grid']
    var = grid.interp(v,'xi')
    add_coords(ds, var, ['xi_rho','eta_rho'])
    var.attrs = v.attrs
    return var.rename(v.name)

# passage du DataArray du point v au point rho sur la grille C
def v2rho(v, ds):
    """
    interpolate horizontally variable from rho to v point
    """
    grid = ds.attrs['xgcm-Grid']
    var = grid.interp(v,'eta')
    add_coords(ds, var, ['xi_rho','eta_rho'])
    var.attrs = v.attrs
    return var.rename(v.name)

# passage du DataArray du point rho au point v sur la grille C
def rho2v(v, ds):
    """
    interpolate horizontally variable from rho to v point
    """
    grid = ds.attrs['xgcm-Grid']
    var = grid.interp(v,'eta')
    add_coords(ds, var, ['xi_v','eta_v'])
    var.attrs = v.attrs
    return var.rename(v.name)

# passage du DataArray du niveau w au niveau rho sur la verticale
def w2rho(v,ds):
    """
    interpolate vertically variable from w to rho point
    """
    grid = ds.attrs['xgcm-Grid']
    var = grid.interp(v,'s')
    add_coords(ds, var, ['xi_rho','eta_rho'])
    var.attrs = v.attrs
    return var.rename(v.name)

def get_z(run, zeta=None, h=None, vgrid='r', hgrid='r', vtrans=None):
    ''' compute vertical coordinates
        zeta should have the size of the final output
        vertical coordinate is first in output
    '''

    ds = run
    N = run.dims['s_rho']
    hc = run.hc

    _h = ds.h if h is None else h
    _zeta = 0*ds.h if zeta is None else zeta

    # swith horizontal grid if needed (should we use grid.interp instead?)
    if hgrid in ['u','v']:
        funtr = eval("rho2"+hgrid)
        _zeta = funtr(_zeta, ds)
        _h = funtr(_h, ds)

    # determine what kind of vertical corrdinate we are using (NEW_S_COORD)
    if vtrans is None:
        vtrans = ds.Vtransform.values
    else:
        if isinstance(vtrans, str):
            if vtrans.lower()=="old":
                vtrans = 1
            elif vtrans.lower()=="new":
                vtrans = 2
            else:
                raise ValueError("unable to understand what is vtransform")

    # sc = sc_r ou sc_w suivant le paramètre vgrid
    sc=ds['sc_'+vgrid]
    cs=ds['Cs_'+vgrid]

    if vtrans == 2:
        z0 = (hc * sc + cs * _h) / (hc + _h)
        z = _zeta + (_zeta + _h) * z0
    else:
        z0 = hc*sc + (_h-hc)*cs
        z = z0 + _zeta*(1+z0/_h)

    z = z.squeeze()
    xdim = "x_"+hgrid.replace('r','rho').replace('v','rho')
    ydim = "y_"+hgrid.replace('r','rho').replace('u','rho')
    zdim = "s_"+vgrid.replace('r','rho')

    if 'time_counter' in z.dims :
        z = z.transpose(*('time_counter',zdim,ydim,xdim))
    else:
        if z.dims[0] != zdim:
            z = z.transpose(*(zdim,)+_zeta.dims)
    return z.rename('z_'+vgrid)

def rotuv(ds, hgrid='r'):
    '''
    Rotate winds or u,v to lat,lon coord -> result on rho grid by default
    '''

    import timeit

    startime = timeit.default_timer()
    angle = ds.angle.persist()
    u=u2rho(ds.u,ds).persist()
    v=v2rho(ds.v,ds).persist()
    print("elaps is :", timeit.default_timer() - startime)

    startim = timeit.default_timer()
    cosang = np.cos(angle).persist()
    sinang = np.sin(angle).persist()
    urot = u*cosang - v*sinang
    vrot = u*sinang + v*cosang
    print("elaps is :", timeit.default_timer() - startime)

    if 'hgrid' == 'u':
        urot = rho2u(urot,ds)
        vrot = rho2u(vrot,ds)
    elif 'hgrid' == 'v':
        urot = rho2v(urot,ds)
        vrot = rho2v(vrot,ds)

    return [urot,vrot]

def find_nearest_above(my_array, target, axis=0):
    diff = target - my_array
    diff = diff.where(diff>0,np.inf)
    return xr.DataArray(diff.argmin(axis=axis))

def slice(ds, var, z, longitude=None, latitude=None, depth=None):
        """
        #
        #
        # This function interpolate a 3D variable on a slice at a constant depth or
        # constant longitude or constant latitude
        #
        # On Input:
        #
        #    ds      dataset to find the grid
        #    var     (dataArray) Variable to process (3D matrix).
        #    z       (dataArray) Depths at the same point than var (3D matrix).
        #    longitude   (scalar) longitude of the slice (scalar meters, negative).
        #    latitude    (scalar) latitude of the slice (scalar meters, negative).
        #    depth       (scalar) depth of the slice (scalar meters, negative).
        #
        # On Output:
        #
        #    vnew    (dataArray) Horizontal slice
        #
        #
        """

        if z.shape != var.shape:
            print('slice: var and z shapes are different')
            return

        [N, M, L] = var.shape

        # Find dimensions of the variable
        xdim = [s for s in var.dims if "x_" in s][0]
        ydim = [s for s in var.dims if "y_" in s][0]
        zdim = [s for s in var.dims if "s_" in s][0]

        # Find horizontal coordinates of the variable
        x = [var.coords[s] for s in var.coords if s in ["xi_rho","xi_u","xi_v"]][0]
        y = [var.coords[s] for s in var.coords if s in ["eta_rho","eta_u","eta_v"]][0]
        s = [var.coords[s] for s in var.coords if "s_" in s][0]

        # Adapt the mask on the C-grid
        mask = ds.mask_rho
        if 'u' in xdim : mask = rho2u(mask,ds)
        if 'v' in ydim : mask = rho2v(mask,ds)


        # Find the indices of the grid points just below the longitude/latitude/depth
        if longitude is not None:
            indices = find_nearest_above(x, longitude, axis=1)
        elif latitude is not None:
            indices = find_nearest_above(y, latitude, axis=0)
        elif depth is not None:
            indices = find_nearest_above(z, depth, axis=0)
            #indices = indices.where(indices<=N-2,N-2)
        else:
            "Longitude or latitude or depth must be defined"
            return None

        # Initializes the 2 slices around the longitude/latitude/depth
        if longitude is not None:
            #var.chunk({xdim : 1, ydim : M, zdim : N})
            Mr = np.arange(M)
            x1 = x[Mr,indices]
            x2 = x[Mr,indices+1]
            y1 = y[Mr,indices]
            y2 = y[Mr,indices+1]
            z1 = z[:,Mr,indices]
            z2 = z[:,Mr,indices+1]
            v1 = var[:,Mr,indices]
            v2 = var[:,Mr,indices+1]
        elif latitude is not None:
            #var.chunk({xdim: L, ydim : 1, zdim: N})
            Lr = np.arange(L)
            x1 = x[indices,Lr]
            x2 = x[indices+1,Lr]
            y1 = y[indices,Lr]
            y2 = y[indices+1,Lr]
            z1 = z[:,indices,Lr]
            z2 = z[:,indices+1,Lr]
            v1 = var[:,indices,Lr]
            v2 = var[:,indices+1,Lr]
        elif depth is not None:
            z1 = z[indices]
            z2 = z[indices+1]
            v1 = var[indices]
            v2 = var[indices+1]

        # Do the linear interpolation
        if longitude is not None:
            xdiff = x1 - x2
            ynew =  (((y1 - y2) * longitude + y2 * x1 - y1 * x2) / xdiff)
            znew =  (((z1 - z2) * longitude + z2 * x1 - z1 * x2) / xdiff)
            vnew =  (((v1 - v2) * longitude + v2 * x1 - v1 * x2) / xdiff)
        elif latitude is not None:
            ydiff = y1 - y2
            xnew =  (((x1 - x2) * latitude + x2 * y1 - x1 * y2) / ydiff)
            znew =  (((z1 - z2) * latitude + z2 * y1 - z1 * y2) / ydiff)
            vnew =  (((v1 - v2) * latitude + v2 * y1 - v1 * y2) / ydiff)
        elif depth is not None:
            #zmask = np.where(z1>depth,np.nan,1)
            zmask = z1 * 0. + 1
            zmask = zmask.where(z1<depth,np.nan)
            vnew =  mask * zmask * (((v1 - v2) * depth + v2 * z1 - v1 * z2) / (z1 - z2))

        # Add the coordinates to dataArray
        if longitude is not None:
            ynew = ynew.expand_dims({s.name: N})
            vnew = vnew.assign_coords(coords={"z":znew})
            vnew = vnew.assign_coords(coords={y.name:ynew})
            #vnew.chunk({ydim: M, zdim: N})

        elif latitude is not None:
            xnew = xnew.expand_dims({s.name: N})
            vnew = vnew.assign_coords(coords={"z":znew})
            vnew = vnew.assign_coords(coords={x.name:xnew})
            #vnew.chunk({xdim: L, zdim: N})

        elif depth is not None:
            vnew = vnew.assign_coords(coords={y.name:y})
            vnew = vnew.assign_coords(coords={x.name:x})
            #vnew.chunk({xdim: L, ydim: M})


        return vnew



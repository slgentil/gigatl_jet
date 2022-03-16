#!/usr/bin/python
# -*- coding:Utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fftfreq, fft, fft2, fftshift
from scipy.signal import tukey, detrend

from scipy.interpolate import interp2d

import xarray as xr

# dispersion relations dimensionalisée
B=2.3e-11
modes2 = xr.open_dataset('/home/durand/Documents/GIGATL/relationDispersion/lambda_G3.nc')

def compute_2d_spectrum(F,nx,ny,dx,dy):
    """
    Compute 2d spectrum by using the 
    2d fast fourier transform
    """
    ### detrend
    Fdtr = detrend(detrend(F,axis=1),axis=0)

    ## Window Tapering
    cff_tukey = 0.5
    wdw_y = tukey(ny, cff_tukey)
    wdw_x = tukey(nx, cff_tukey)
    wdw = np.outer(wdw_x,wdw_y)
    Ftpr = Fdtr * wdw[:,:]

    ### spectre 2d
    Fsp = np.zeros(Ftpr.shape)
    ld = 1./dy * 1./dx * (wdw**2).sum()

    pu = fftshift(fft2(Ftpr))
    Fsp = (np.conjugate(pu)*pu).real/ ld

    ### Wavelength and period
    kx = fftshift(fftfreq(nx,dx))
    ky = fftshift(fftfreq(ny,dy))

    return kx, ky, Fsp


def compute_constant(n):
    
    """
    Compute the constant of separation c and
    the dimension O and K of omega and k such 
    that omega = f(k) -> omega*/O = f(k*/K)
    """
    modes2 = xr.open_dataset('/home/durand/Documents/GIGATL/relationDispersion/lambda_G3.nc')
    c = 1./(modes2.lamba.values[n])
#     print('c = ',c)
    
    L = np.sqrt(c/B)
    T = 1./np.sqrt(c*B)

    O=1./(2*np.pi*T)
    K=1./(2*np.pi*L)

    return c,O,K



def rossby_relation(k,m,n,**kwargs):

    """
    approximated relation dispersion for Rossby waves
    """

    options={
        'dim':False,
        }

    options.update(kwargs)

    dim=options['dim']

    if dim==False:
        omega = -1.*(k/(k**2+2*m+1))
    if dim==True:
        c,O,K = compute_constant(n)
        omega = -1.*O*((K**-1)*k/((K**-2)*(k**2)+2*m+1))

    return omega


def gravity_relation(k,m,n,**kwargs):

    """
    Approximated relation dispersion for Inerta-Gravity waves
    """
    
    options={
        'dim':False,
        }

    options.update(kwargs)

    dim=options['dim']

    if dim==False:
        omega=np.sqrt(k**2+2*m+1)
    if dim==True:
        c,O,K = compute_constant(n)
        omega= O*np.sqrt((K**-2)*(k**2)+2*m+1)

    return omega


def yanai_relation(k,n,**kwargs):

    """
    Relation dispersion for Yanai waves
    """
    
    options={
        'dim':False,
        }

    options.update(kwargs)

    dim=options['dim']


    if dim==False:
        omega=(1/2.)*(k+np.sqrt(k**2+4))
    if dim==True:
        c,O,K = compute_constant(n)
        omega= O*(1./2.)*((K**(-1))*k+np.sqrt((K**(-2))*(k**2)+4))

    return omega


def kelvin_relation(k,n,**kwargs):

    """
    Relation dispersion for Kelvin waves
    """
    
    options={
        'dim':False,
        }

    options.update(kwargs)

    dim=options['dim']

   
    if dim==False:
        omega=k
    if dim==True:
        c,O,K = compute_constant(n)
        omega= O*(K**-1)*k

    return omega

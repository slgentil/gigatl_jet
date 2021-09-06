# Audrey D.
# May 2020

import numpy as np

import scipy.linalg as la
import scipy.interpolate as itp


def cheb(N, inter=None):
    
    """
    Chebyshev polynomial
    """

    N -= 1

    if N==0:
        return None
    
    xx = np.cos(np.pi*np.arange(N+1)/N)
    cc = np.r_[2, np.ones(N-1), 2]*(-1)**np.arange(N+1)
    X = np.tile(xx[:,None], (1, N+1))
    dX = X - X.T
    D = (cc[:,None]/cc[None,:])/(dX + np.diag(np.ones(N+1)))
    D = D - np.diag(D.sum(axis=1))

    if not inter is None:
        L = inter[1] - inter[0]
        D = -D*2/L
        xx = (xx[::-1] + 1) * L/2. + inter[0]
    
    return D, xx



def SL_chebsolve(alsq, zw, Nmod="auto", Nz="auto", grav=0, sm=0, ksplin=3, zbot=None):
    
    """ 
    Solve Sturm-Liouville problem with ev k: w'' + k*alsq*w = 0, w(-H)=w(0)=0
    between zbot (default: zw[0]) and zw[-1] = 0
    wmod and umod (=wmod') are normalized by max value, with u positive at surface
    if grav != 0: free-surface boundary condition. 
    :return: tuple with (wmod, umod), eigenvalue sqrt(k) and z-cheb
    Based on Noe code, May 2020
    """
    if Nz=="auto":
        Nz = int(len(zw)*3/2.)
    if Nmod == "auto":
        Nmod = int(Nz/2)
    if zbot is None: 
        zbot = zw[0]
    
    # Chebyshev Polynomial Interpolation
    Dz, zz = cheb(Nz, [zbot, zw[-1]])
    alsq = itp.UnivariateSpline(zw, alsq, k=ksplin, s=sm)(zz)
    
    # Construc Operator
    LL = np.r_[ np.c_[ np.diag(np.ones(Nz)), -Dz ] \
                    , np.c_[ -Dz, np.zeros((Nz,Nz)) ] ]
    AA = np.diag(np.r_[np.zeros(Nz), alsq])
            
    # Boundary Conditions
    LL[Nz,:] = 0. # bottom
    LL[-1,:] = 0. # top
    if grav > 0:
        LL[-1,-1] = 0.
        AA[-1,-1] = grav
        LL[-1,Nz-1] = 1.
    
    # Diagonalize Operator
    lam, vect = la.eig(LL, AA)
    
    # Filter eigenvalues
    inds, = np.where( (np.isfinite(lam)) & (abs(lam.real)<1e3) & (abs(lam.imag)<1e-6) & (lam.real>0) )
    lam, vect = lam[inds], vect[:,inds]
    
    # Sort eigenvalues
    inds = lam.real.argsort()[:Nmod]
    
    # Normalize the eigenvectors
    vect = vect[:,inds]/abs(vect[:,inds]).max(axis=0)[None,:]
    lam = lam[inds]

    ww = vect[Nz:,:]
    uu = vect[:Nz,:]
    ww *= np.sign(uu[-1:,:])
    uu *= np.sign(uu[-1:,:])

    return (ww, uu), np.sqrt(lam), zz



def norm_mode(mode,z):
    
    """
    Normalize the basis of the eigenmode given 
    the scalar product <f|g>= \int_-H^0 fxg dz
    \int_-H^0 phi(z)^2 dz = H (for all mode phi)
    :return: normalized eigenmode vector
    Audrey Oct. 2019
    """

    scl = 0
    
    for k in range(z.size):
        if k==0 :
            scl += mode[k]**2 *(z[k+1]-z[k])
        else :
            scl += mode[k]**2 *(z[k]-z[k-1])

    scl = np.sqrt(np.abs(scl/(z[0]-z[-1])))

    return mode/scl









import numpy as np
import pylab as plt
import matplotlib
from matplotlib import cm
cm_my=cm.viridis
from matplotlib import rc
import astropy
from astropy.io import fits
import sys
import math
import scipy
from scipy import ndimage as nd
from scipy import stats as st
import healpy as hp

###################
def map_iqu2teb(map_in, nside):
    '''
        From I, Q, U maps to I, E, B maps
        '''
    npix = hp.nside2npix(nside)
    lmax = 3*nside - 1
    alms = hp.map2alm(map_in,lmax = lmax,pol=True)
    maps_teb = hp.alm2map(alms,nside,pol=False)

    return maps_teb

###################
def map_teb2iqu(map_in, nside):
    '''
        From I, E, B maps to I, Q, U maps
        '''
    npix = hp.nside2npix(nside)
    lmax = 3*nside - 1
    alms_T = hp.map2alm(map_in[0,:],lmax = lmax,pol=False)
    alms_E = hp.map2alm(map_in[1,:],lmax = lmax,pol=False)
    alms_B = hp.map2alm(map_in[2,:],lmax = lmax,pol=False)
    alms_v = np.zeros([3,np.shape(alms_T)[0]],dtype=complex)
    alms_v[0,:] = alms_T; alms_v[1,:] = alms_E; alms_v[2,:] = alms_B
    maps_tqu = hp.alm2map(alms_v,nside,pol=True)
    
    return maps_tqu

###################
def iqu_degrade(mapiqu, NSIDE_in, NSIDE_out):
    '''
        Degrading I, Q, and U maps from NSIDE_in to NSIDE_out
        '''
    maps_teb = map_iqu2teb(mapiqu, NSIDE_in)
    mape = hp.ud_grade(maps_teb[1,:],nside_out=NSIDE_out,order_in='RING')
    mapb = hp.ud_grade(maps_teb[2,:],nside_out=NSIDE_out,order_in='RING')
    mapt = hp.ud_grade(maps_teb[0,:],nside_out=NSIDE_out,order_in='RING')
    cbin = np.zeros([3,hp.nside2npix(NSIDE_out)])
    cbin[0,:] = mapt; cbin[1,:] = mape; cbin[2,:] = mapb
    mapiqu = map_teb2iqu(cbin, NSIDE_out)

    return mapiqu

###################
def iqu_coord_rot(data_cube, str_coord):
    '''
        Rotating Stokes parameters from two coordinate systems e.g., 'GC' to change from Galactic to Celestial.
        '''
    r = hp.Rotator(coord=str_coord)
    rot_cube = hp.Rotator.rotate_map_alms(r,data_cube)
    
    return rot_cube

###################
def plot_B_streamlines(mi, mq, mu, imax, imin):
    '''
        Plotting B-field lines on top of Stokes I map
        '''
    import pylab as plt
    import os
    
    if np.shape(np.shape(mi))[0] == 1: return print('********** NO PLOT: Maps must be 2D arrays ************'),os.system('afplay /System/Library/Sounds/Sosumi.aiff')
    
    nx = np.shape(mi)[0]
    ny = np.shape(mi)[1]
    x = np.linspace(0, nx-1, nx)
    y = np.linspace(0, ny-1, ny)
    xp = x
    yp = y
    y2, x2 = np.meshgrid(xp, yp, indexing='ij')
    psi=-0.5*np.arctan2(mu,mq)
    x3=np.cos(psi)
    y3=np.sin(psi)
    plt.figure(10)
    plt.contourf(mi,levels=np.arange( imin, imax, (imax-imin)/500.), extent=[0, nx, 0, ny])
    plt.colorbar()
    plt.streamplot(x2, y2, x3, y3, density=[3, 3], color='w',arrowstyle='-',linewidth=0.5)
    plt.show()
    return











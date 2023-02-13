
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

#########################

def random_map_hpx(nside, alphaM, resol):
    '''
    Creates a random map with a given power-law power spectrum of index alphaM
    calling sequence:
    map_random = random_map_hpx(nside, alphaM, resol)
    '''
    import numpy as np
    import pylab as plt
    import matplotlib
    from matplotlib import cm
    cm_my=cm.viridis
    from matplotlib import rc
    import astropy
    #import idlwrap as idl
    from astropy.io import fits
    import sys
    import math
    import scipy
    from scipy import ndimage as nd
    from scipy import stats as st
    from scipy import constants as ko
    from scipy import signal
    from numpy.random import MT19937
    from numpy.random import RandomState, SeedSequence
    import healpy as hp
    
    el=np.linspace(0,nside*3,nside*3+1)
    cl=el**(alphaM)
    cl[0:1]=0.
    rs = RandomState(MT19937(SeedSequence(123456789)))
    mapran = hp.synfast(cl,nside=nside,lmax=2*nside+1,fwhm=resol/60*np.pi/180)
    
    return mapran


#######################
def powerplus_ran(mapin, nsidein, nsideout, resol_o, lminfit, lmaxfit, lcut, plot=False):
 
    '''
    Adding small-scale structure to a map (map_in) using Gausssian random fields with power-law power spectra as the input map
    calling sequence
    
    m_smaller_scales = powerplus_ran( mapin, nsidein, nsideout, resol_o, lminfit, lmaxfit, lcut, plot=False/True)
    '''

    lmax0 = 2*nsidein-1
    cl0=hp.sphtfunc.anafast(mapin,lmax=lmax0)
    el = np.arange(np.shape(cl0)[0])
    sel = np.where((el >= lminfit) & (el <= lmaxfit))
    res = np.polyfit(np.log10(el[sel]),np.log10(cl0[sel]),1)

    lmax1 = 2*nsideout-1
    mrmd = hp.ud_grade(mapin,nside_out=nside1,order_in='RING')
    cl1=hp.sphtfunc.anafast(mrmd,lmax=lmax1)
    el1 = np.arange(np.shape(cl1)[0])
    
    mrm_sc = random_map_hpx(nsideout,res[0],resol_o)
    cls1=hp.sphtfunc.anafast(mrm_sc,lmax=lmax1)
    ind = lcut
    mtest = mrm_sc*np.sqrt(cl0[ind])/np.sqrt(cls1[ind])
    clsf=hp.sphtfunc.anafast(mtest,lmax=lmax1)

    clin = clsf.copy()
    clin[np.where(el1 <= ind)] = 0
    almn = hp.synalm(clin, lmax = lmax1)
    mapn = hp.alm2map(almn,nsideout, lmax = lmax1, pol = False)

    alms = hp.map2alm(mrmd,lmax = lmax1,pol=False)
    fl = cl1.copy()*0.
    fl[np.where(el1 <= ind)]=1
    alms1 = hp.almxfl(alms,fl)
    mrmc = hp.alm2map(alms1,nsideout, lmax = lmax1, pol = False)
    mapfin = mrmc + mapn
    clfin = hp.anafast(mapfin,lmax= lmax1)
    
    if plot == True :
        plt.loglog(el[1:],cl0[1:],label='Input map spectrum')
        plt.loglog(el1[1:],clfin[1:],alpha=0.5,label='Small-scale structure added')
        plt.axvline(x = ind,color='black',linestyle='dashed',alpha=0.5)
        plt.ylabel(r'$C_{\mathcal{l}}$')
        plt.xlabel(r'$\mathcal{l}$')
        plt.legend()

    return mapfin
    








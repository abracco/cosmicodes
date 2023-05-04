
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
    mrmd = hp.ud_grade(mapin,nside_out=nsideout,order_in='RING')
    cl1=hp.sphtfunc.anafast(mrmd,lmax=lmax1)
    el1 = np.arange(np.shape(cl1)[0])
    
    mrm_sc = random_map_hpx(nsideout,res[0],resol_o)
    cls1=hp.sphtfunc.anafast(mrm_sc,lmax=lmax1)
    ind = lcut
    mtest = mrm_sc*np.sqrt(np.median(cl0[ind:ind+5]))/np.sqrt(np.median(cls1[ind:ind+5]))
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
    
##########################
def dirade_hpx(nside, resol, spec, N=5):
    '''
    #####
    DIrectional RAndom cascADE in Healpix. See Robitaille et al. 2020 and Matthew Price's page http://astro-informatics.github.io/s2fft/index.html#
    
    Inputs: NSIDE, angular resolution in arcmin, spectral index of the Gaussian random field, Number (N) of wavelet directions (2*N - 1);
    Outputs: filametary and Gaussian random maps
    
    Needed packages and libraries:
    - Planck_routines.py : https://github.com/abracco/cosmicodes/tree/master/4GMIMS
    - healpy : pip install healpy
    - JAX wavelet transform code : pip install git+https://github.com/astro-informatics/s2wav.git
    #####
        '''
    from jax.config import config
    config.update("jax_enable_x64", True)
    import healpy as hp
    import s2wav, s2fft
   
    L = 2*nside
    m = random_map_hpx(nside,spec,resol)
    flm_healpix = hp.sphtfunc.map2alm(m, lmax=L-1, iter=10)
    flm = s2fft.sampling.s2_samples.flm_hp_to_2d(flm_healpix, L)
    f = s2fft.inverse_jax(flm, L, reality=True)
    J_min = 0
    wavelet_filters = s2wav.filter_factory.filters.filters_directional_vectorised(L, N, J_min)
    wavelet_maps, scaling_maps = s2wav.analysis(f, L, N, J_min, reality=True, filters=wavelet_filters)
    prod = 1
    for j in range(np.shape(wavelet_maps)[0]): prod = prod*np.exp(wavelet_maps[j]/np.std(wavelet_maps[j]))
    sum0 = np.sum(prod,axis=0)
    mfil = np.log10(np.abs(sum0)/np.shape(wavelet_maps)[1])
    flm_back = s2fft.forward_jax((mfil),L,reality=True)
    flm_b_hpx = s2fft.sampling.s2_samples.flm_2d_to_hp(flm_back, L)
    mapf = hp.sphtfunc.alm2map(flm_b_hpx, nside, lmax = L-1)
 
    return mapf, m

###############################

def stereo_proj_hpx(nside, nx, ny, radius, m_in, l0, b0, plot=False, step = 10):
    '''
    #####
    This routine produces a stereographic projection (m_out) of the HEALPix map given as input (m_in).
    The code is adapted from http://mathworld.wolfram.co;m StereographicProjection.html
    INPUTS
    nside - is the HEALPix parameter of m_in
    nx, ny, - are the x,y dimensions of the output map
    radius - is the radius in degrees of the selected area that is projected
    m_in - input map
    l0 - the central Galactic longitude in degrees
    b0 - the central Galactic latitude in degrees
    ncont - number of coordinates contours in the final image
    OUTPUTS
    m_out - output map nx times ny
    l1 and b1 - output maps of projected Galactic coordinates in radians
    #####
        '''
    npix = hp.nside2npix(nside)
    pix = np.arange(hp.nside2npix(nside))
    theta, phi = hp.pix2ang(nside,pix)
    glon = phi
    glat = np.pi/2. - theta
    
    l0=np.deg2rad(l0)
    b0=np.deg2rad(b0)
    vec_tmp=np.array([np.cos(l0)*np.cos(b0),np.sin(l0)*np.cos(b0),np.sin(b0)])
    sel=np.where(glon != 0)
    
    if radius !=  0 : sel = hp.query_disc(nside,vec_tmp,np.deg2rad(radius))
    mask=m_in.copy()*0
    mask[sel]=1
    morto = hp.orthview(m_in*mask,rot=[np.rad2deg(l0),np.rad2deg(b0)],half_sky=True,norm='hist',return_projected_map=True).data;plt.title='Orthographic test projection'
    hp.graticule()
    #print('1')
    lr=glon[sel]
    br=glat[sel]
    
    R=1
    A=2.*R/(1+np.sin(b0)*np.sin(br)+np.cos(b0)*np.cos(br)*np.cos(lr-l0))
    sel2=np.where(A < 1e4)
    x1= A*np.cos(br)*np.sin(lr-l0)
    y1= A*(np.cos(b0)*np.sin(br)-np.sin(b0)*np.cos(br)*np.cos(lr-l0))
    x1=x1[sel2]
    y1=y1[sel2]
    
    xn = np.linspace(0,nx-1,nx)
    yn = np.linspace(0,ny-1,ny)
    ynn, xnn = np.meshgrid(xn,yn,indexing='ij')
    
    xnn=-(xnn/(nx-1.)*(np.max(x1)-np.min(x1))+np.min(x1))
    ynn=ynn/(ny-1.)*(np.max(y1)-np.min(y1))+np.min(y1)
    #print('2')
    rho=np.sqrt(xnn**2+ynn**2)
    c=2*np.arctan(rho/(2.*R))
    
    b1=np.arcsin(np.cos(c)*np.sin(b0)+(ynn*np.sin(c)*np.cos(b0))/(rho))
    l1=l0+np.arctan2((xnn*np.sin(c)),(rho*np.cos(b0)*np.cos(c)-ynn*np.sin(b0)*np.sin(c)))
    
    sel1=np.isfinite(l1)
    
    m_out=l1.copy()*0.
    #print('3')
    for j in range(np.shape(sel1)[0]):
        indp = hp.ang2pix(nside,np.pi/2. - b1[j],l1[j])
        m_out[j]=m_in[indp]
    
    if plot == True :
        plt.figure()
        ax = plt.gca()
        plt.imshow(np.log10(m_out),cmap='Spectral_r',origin='lower')
        plt.contour((np.rad2deg(l1)),levels=np.arange(np.min(np.rad2deg(l1)),np.max(np.rad2deg(l1)),step),colors='black',linewidths=0.5,origin='lower',linestyles='dashed')
        plt.contour((np.rad2deg(b1)),levels=np.arange(np.min(np.rad2deg(b1)),np.max(np.rad2deg(b1)),step),colors='black',linewidths=0.5,origin='lower',linestyles='dashed')
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])

    return m_out, np.rad2deg(l1), np.rad2deg(b1), rho





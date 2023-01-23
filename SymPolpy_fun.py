def SymPolpy_fun(nside,resol,l0,b0,fM,Bstr,nfreq,f0,fF):
    '''
    Sky synchrotron simulation of Stokes I, Q and U WITHOUT Faraday rotation between frequencies f0 and fF with nfreq steps. Units of outputs are arbitrary. Requirements: healpy, idlwrap, Planck_routines.py
    INPUTS : nside - pixel resolution of Healpix maps
             resol - FWHM of output maps in arcmin
             l0,b0 - Galactic coordinates of mean magnetic field in degrees
             fM - relative ratio between turbulent and mean magnetic field
             Bstr - magnetic-field rms value
             nfreq - number of desired frequencies
             f0, fF - first and last frequencies
    OUTPUTS: blosm - mean magnetic field along the LOS
             Isy0, Qsy0, Usy0 - Stokes parameters
             nuv - frequency array
    Calling sequence:
             blosm, Isy0, Qsy0, Usy0, nuv = SymPolpy_fun(nside,resol,l0,b0,fM,Bstr,nfreq,f0,fF)
    '''
            
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
    import idlwrap as idl
    from scipy import ndimage as nd
    from scipy import stats as st
    from scipy import constants as ko
    from scipy import signal
    import healpy as hp
    from astropy.utils.data import get_pkg_data_filename

    N = 1
    alphaM = -2.6

    nuv = np.linspace(f0,fF,nfreq)
    pf = 0.7

    el=idl.findgen(nside*3+1)
    cl=el**(alphaM)
    cl[0:1]=0.

    # LOS vectors
    npix=hp.nside2npix(nside)
    lpix=(idl.findgen(npix))
    lpix = lpix.astype(int)
    los_vec = hp.pix2vec(nside, lpix)

    # North and East vectors
    theta,phi = hp.pix2ang(nside,lpix)
    glat=(np.pi/2-theta)
    north_vec=idl.dblarr(npix,3)
    east_vec=idl.dblarr(npix,3)
    north_vec = np.array([[-np.cos(phi)*np.sin(np.pi/2.-theta)],[-np.sin(phi)*np.sin(np.pi/2.-theta)],[np.cos(np.pi/2.-theta)]])[:,0,:]
    east_vec = -np.array([los_vec[1]*north_vec[2]-los_vec[2]*north_vec[1],-los_vec[0]*north_vec[2]+los_vec[2]*north_vec[0], los_vec[0]*north_vec[1]-los_vec[1]*north_vec[0]])

    # large scale field magnetic field
    lon_b0=l0*np.pi/180. ; lat_b0=b0*np.pi/180.
    B0vec=[np.cos(lon_b0)*np.cos(lat_b0),np.sin(lon_b0)*np.cos(lat_b0),np.sin(lat_b0)]

    # Sky map of B field
    Bvec=idl.fltarr(npix,3)
    Q_cube=idl.dblarr(npix,N)
    U_cube=idl.dblarr(npix,N)
    blos_cube=idl.dblarr(npix,N)


    np.random.seed(); mapx = hp.synfast(cl,nside=nside,lmax=2*nside+1,fwhm=resol/60*np.pi/180)
    np.random.seed(); mapy = hp.synfast(cl,nside=nside,lmax=2*nside+1,fwhm=resol/60*np.pi/180)
    np.random.seed(); mapz = hp.synfast(cl,nside=nside,lmax=2*nside+1,fwhm=resol/60*np.pi/180)
    sig_map=np.mean([np.std(mapx),np.std(mapy),np.std(mapz)])
    fac=fM/(sig_map*np.sqrt(3.)) # stdev(Bturb)/B0
        
    for i in range(np.shape(lpix)[0]):
            Bvec[0,i]=B0vec[0]+mapx[i]*fac
            Bvec[1,i]=B0vec[1]+mapy[i]*fac
            Bvec[2,i]=B0vec[2]+mapz[i]*fac
        
    # normalization to unit length
    Bvec=Bvec/np.sqrt(Bvec[0,:]**2.+Bvec[1,:]**2.+Bvec[2,:]**2.) * Bstr

    # compute maps of Stokes parameters

    Bvec_perp=idl.fltarr(npix,3)
    B_times_los=idl.fltarr(npix) # scalar producet B.los

    B_times_los=np.sum(Bvec*los_vec,axis=0)

    Bvec_perp[0]=Bvec[0]-los_vec[0]*B_times_los
    Bvec_perp[1]=Bvec[1]-los_vec[1]*B_times_los
    Bvec_perp[2]=Bvec[2]-los_vec[2]*B_times_los
    
    P_I0=(Bvec_perp[0]**2.+Bvec_perp[1]**2.+Bvec_perp[2]**2.)
        
    Isy0 = np.zeros([npix,nfreq])
    for k in range(nfreq): Isy0[:,k] = np.sqrt(P_I0)**(1.5)*(nuv[k]/f0)**(-0.5)

    #compute Q anbd U maps

    B_angle=idl.fltarr(npix)
    Qmap=idl.fltarr(npix) ; Umap=idl.fltarr(npix)
    g=np.where(P_I0 > 0.)[0]
    buf = (Bvec_perp[0,g]*north_vec[0,g]+Bvec_perp[1,g]*north_vec[1,g]+Bvec_perp[2,g]*north_vec[2,g])/np.sqrt(P_I0[g])
    b = np.where(buf > 1.)
    if np.shape(b)[1] > 0 : buf[b] = 1.
    b = np.where(buf < -1.)
    if np.shape(b)[1] > 0 : buf[b] = -1.
    B_angle[g] = np.arccos(buf)        #; from 0 to !pi
    buf = (Bvec_perp[0,:]*east_vec[0,:]+Bvec_perp[1,:]*east_vec[1,:]+Bvec_perp[2,:]*east_vec[2,:])
    neg_angle = np.where(buf < 0.)[0]
    B_angle[neg_angle] = -B_angle[neg_angle]
    pol_angle = B_angle + np.pi/2.
    b=np.where(pol_angle > np.pi)[0]
    pol_angle[b]=  pol_angle[b]-2.*np.pi
    Qmap = P_I0*np.cos(2.*pol_angle)
    Umap = -P_I0*np.sin(2.*pol_angle)   # minus sign Planck vs IAU convention
                            
    psi_pol=pol_angle # polarisation angle from -pi/2 to !pi/2
    bn=np.where(psi_pol < -np.pi/2.)[0]
    psi_pol[bn]=psi_pol[bn]+np.pi
    bp=np.where(psi_pol > np.pi/2.)[0]
    psi_pol[bp]=psi_pol[bp]-np.pi

    Q_cube=Qmap
    U_cube=Umap
    blos_cube = B_times_los

    # Preparing Outputs
    Qm = Q_cube
    Um = U_cube
    blosm = blos_cube

    Qsy0 = np.zeros([npix,nfreq])
    Usy0 = np.zeros([npix,nfreq])
    for k in range(nfreq):
        Qsy0[:,k] = np.sqrt(P_I0)**(1.5)*(nuv[k]/f0)**(-0.5)*pf*Qm
        Usy0[:,k] = np.sqrt(P_I0)**(1.5)*(nuv[k]/f0)**(-0.5)*pf*Um

    return -blosm, Isy0, Qsy0, Usy0, nuv


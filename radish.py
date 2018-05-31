# Reprojecting 3D fields in Dust polarization Stokes parameters with HEALPix (RaDiSH.py)
'''
    Contains the program to reproject on a HEALPix sphere the Stokes parameters for linear polarization of interstellar dust EMISSION.
    Equations from "Planck Collaboration int. results. XX. 2016"
    Calling sequence:
    StokesI, StokesQ, StokesU = radish(rho,bfield0,p0,nx,ny,nz,nside)
'''

def radish(rho,bfield0,p0,nx,ny,nz,nside):

    import numpy as np
    import pylab as plt
    from pylab import cm
    import matplotlib
    import astropy
    import sys
    import math
    import scipy
    from scipy import ndimage as nd
    from scipy import stats as st
    #IMPORTANT you need to install healpy package!
    import healpy as hp

    # Define the box size.
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    dx=(x[1]-x[0])
    dy=(y[1]-y[0])
    dz=(z[1]-z[0])

    rr=np.sqrt(xx**2+yy**2+zz**2)
    thc=np.arccos(zz/(rr+1e-6))
    phic=np.arctan2(yy,xx)
    phic[phic < 0]=phic[phic < 0]+2*math.pi

    # create reference frame
    zn=np.array([0,0,1])
    rn=np.array([np.cos(phic)*np.sin(thc),np.sin(phic)*np.sin(thc),np.cos(thc)])

    # north and east directions
    # the part in the following is used to create grids of norths and easts for each line of sight.

    nn = rn.copy()*0.
    ee = rn.copy()*0.

    for i in range(0,nx):
        print(i)
        for j in range(0,ny):
            for k in range(0,nz):
                vectmp = np.cross(np.cross(rn[:,i,j,k],zn),rn[:,i,j,k])
                nn[:,i,j,k]= vectmp/np.linalg.norm(vectmp)
                vectmp1 = -np.cross(rn[:,i,j,k],nn[:,i,j,k])
                ee[:,i,j,k]= vectmp1/np.linalg.norm(vectmp1)

    b_n= rho.copy()*0.
    for i in range(0,nx):
        for j in range(0,ny):
            for k in range(0,nz):
                b_n[i,j,k] = np.linalg.norm(bfield0[:,i,j,k])

    bfield=bfield0/b_n
    bperp = bfield-dot(bfield,rn)*rn
    cos2g = 1-(dot(bfield,rn))**2.
    bperp_n= rho.copy()*0.
    for i in range(0,nx):
        for j in range(0,ny):
            for k in range(0,nz):
                bperp_n[i,j,k] = np.linalg.norm(bperp[:,i,j,k])

    rat=dot(bperp,nn)/(bperp_n)
    rat[(rat>1)]=1.
    rat[(rat<-1)]=-1.

    chi = np.arccos(rat)
    parm = dot(bperp,ee)

    sel_n = np.where(parm < 0)
    chi[sel_n]= -chi[sel_n]

    psi=chi+math.pi/2.
    sel_p = np.where(psi > math.pi)
    psi[sel_p]=psi[sel_p]-2.*math.pi

    #HEALPix projection
    pxr=hp.nside2resol(nside,arcmin=True)/60.*math.pi/180.
    npix=hp.nside2npix(nside)
    lpix=np.arange(0,npix,1)
    thlist, phlist = hp.pixelfunc.pix2ang(nside, lpix, nest=False, lonlat=False)

    rhoc = thlist.copy()*0.
    r0=np.sqrt(x**2+y**2+z**2)
    dr=np.abs(r0[1]-r0[0])
    hits=rhoc.copy()*0.
    rho1=rho.copy()*0.+1
    qmap=rhoc.copy()*0.
    umap=rhoc.copy()*0.


    for j in range(0,npix) :
        sel= np.where((np.abs(phlist[j]-phic) <= pxr) & (np.abs(thlist[j]-thc) <= pxr))
        rhoc[j]=np.sum(rho[sel]*(1-p0*(cos2g[sel]-2./3.)))*dr
        hits[j]=np.sum(rho1[sel])*dr
        c2gm[j]=np.sum(cos2g[sel])*dr
        qmap[j]=np.sum(p0*cos2g[sel]*rho[sel]*np.cos(2*psi[sel]))*dr
        umap[j]=np.sum(-p0*cos2g[sel]*rho[sel]*np.sin(2*psi[sel]))*dr
        mask[sel]=1

    rhof=rhoc/hits
    qf=qmap/hits
    uf=umap/hits

    return rhof, qf, uf



import numpy as np
import pylab as plt
import matplotlib
from matplotlib import cm
cm_my=cm.viridis
import astropy
from astropy.io import fits
import sys
import math
import scipy
from scipy import constants as ko

exec(open('Planck_routines.py').read())
exec(open('SymPolpy_fun.py').read())
plt.ion()

### create RM map
def random_map_hpx(nside, alphaM, resol):
    
    '''
        Routine to create healpix Gaussian random maps, with given nside, from power-law power spectra with slope alphaM. The FWHM of the maps must be given in input in arcmin (resol). The routine returns one random map with values between +1 and -1.
        '''
    
    from numpy.random import MT19937
    from numpy.random import RandomState, SeedSequence
    import healpy as hp
    import idlwrap as idl
    
    plt.ion()
    
    el=idl.findgen(nside*3+1)
    cl=el**(alphaM)
    cl[0:1]=0.
    
    rs = RandomState(MT19937(SeedSequence(123456789)))
    mapran = hp.synfast(cl,nside=nside,lmax=2*nside+1,fwhm=resol/60*np.pi/180)
    
    return mapran/np.max(np.abs(mapran))
    
def pow2_hp(map,nside):
    cl0=hp.sphtfunc.anafast(map,lmax=2*nside+1)
    el0=np.arange(np.shape(cl0)[0])
    plt.figure();
    plt.loglog(el0,cl0,'.-',label='angular power spectrum')
    plt.xlabel('multipole')
    plt.ylabel('power [a.u.]')
    return

def rm_rot_hp(I,Q,U,RM,nu_v,beam):
    '''
        Rotating Stokes Q and U given rotation measure (RM)
        Inputs: Stokes I, Q, U
        RM map
        frequency range in Hz
        beam: if smoothing is not required put 0 otherwise in arcmin
        '''
    from scipy import constants as ko
    N_ch = np.shape(nu_v)[0]
    nx = np.shape(Q)[0]
    Qnu = np.zeros([nx,N_ch])
    Unu = np.zeros([nx,N_ch])
    
    for j in range(0,N_ch):
        P = np.sqrt(Q[:,j]**2 + U[:,j]**2)
        phi = 0.5*np.arctan2(U[:,j],Q[:,j])
        phi[np.isnan(phi)]=0
        Qnu[:,j] = P*np.cos(2*(RM*(ko.c/nu_v[j])**2 + phi))
        Unu[:,j] = P*np.sin(2*(RM*(ko.c/nu_v[j])**2 + phi))
        if (beam != 0) :
            cube = np.zeros([3,nx])
            cube[0,:] = I[:,j]
            cube[1,:] = Qnu[:,j]
            cube[2,:] = Unu[:,j]
            cbo = hp.smoothing(cube,fwhm=beam/60.*np.pi/180.,pol=True)
            I[:,j] = cbo[0,:]; Qnu[:,j]= cbo[1,:]; Unu[:,j]=cbo[2,:]
    
    return I, Qnu, Unu

#### starting code
nside=128
blos, Is, Qs, Us, nuv = SymPolpy_fun(nside,1,70,24,1,5,20,400e6,800e6) # synchrotron emission only
rm_map = random_map_hpx(nside, -3, 1)*10 # RM map
Irm, Qrm, Urm = rm_rot_hp(Is,Qs,Us,rm_map,nuv,60) # Rotating Stokes parameters by RM and smoothing

# showing plots
pow2_hp(Is[:,0],nside)
mi0 = hp.mollview(Is[:,0],return_projected_map=True)
mq0 = hp.mollview(Qs[:,0],return_projected_map=True)
mu0 = hp.mollview(Us[:,0],return_projected_map=True)
mi1 = hp.mollview(Irm[:,0],return_projected_map=True)
mq1 = hp.mollview(Qrm[:,0],return_projected_map=True)
mu1 = hp.mollview(Urm[:,0],return_projected_map=True)

plt.figure();plt.subplot(231);plt.imshow(mi0);plt.title('Stokes I');plt.subplot(232);plt.imshow(mq0);plt.title('Stokes Q');plt.subplot(233);plt.imshow(mu0);plt.title('Stokes U')
plt.subplot(234);plt.imshow(mi1);plt.title('Stokes I');plt.subplot(235);plt.imshow(mq1);plt.title('Stokes Q RM');plt.subplot(236);plt.imshow(mu1);plt.title('Stokes U RM')
plt.tight_layout()

plt.figure();plt.plot(nuv/1e6,Qs[10,:],label='NO RM');plt.plot(nuv/1e6,Qrm[10,:],label='RM rotated');plt.xlabel('frequency [MHz]');plt.ylabel('Stokes Q [a.u.]');plt.legend()

hp.mollview(rm_map,cmap='seismic');plt.title('RM')

# Welcome to the MISalignment Operator (MISO), a simple code to cook your preferred misalignment configuration between polarization angles. Enjoy your soup!

def ang_diff(Q1,U1,Q2,U2):
    '''
        computing polarization angle difference in radians from Stokes parameters (Q1,U1) and (Q2,U2)
        '''
    alpha = 0.5*np.arctan2(U1,Q1) #Planck
    beta = 0.5*np.arctan2(U2,Q2) #Hi
    diff = 0.5*np.arctan2(np.sin(2*alpha)*np.cos(2*beta)-np.cos(2*alpha)*np.sin(2*beta),np.cos(2*beta)*np.cos(2*alpha)+np.sin(2*beta)*np.sin(2*alpha))
    return diff

def SOSDPolpy_fun(nside = 128, resol = 60, l0 =70, b0=20, N = 1, fM = 0, Bstr = 1, alphaM = -2.6):
    '''
        computing dust Stokes parameters from the uniform+turbulent magnetic-field field model as in Planck XLIV 2016. Output: -Blos, P, Q, U
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
    from scipy import ndimage as nd
    from scipy import stats as st
    from scipy import constants as ko
    from scipy import signal
    import healpy as hp
    from astropy.utils.data import get_pkg_data_filename
    el=np.linspace(0,nside*3,nside*3+1)
    cl=el**(alphaM)
    cl[0:1]=0.
    # LOS vectors
    npix=hp.nside2npix(nside)
    lpix=np.linspace(0,npix-1,npix)
    lpix = lpix.astype(int)
    los_vec = hp.pix2vec(nside, lpix)
    # North and East vectors
    theta,phi = hp.pix2ang(nside,lpix)
    glat=(np.pi/2-theta)
    north_vec=np.zeros([npix,3])
    east_vec=np.zeros([npix,3])
    north_vec = np.array([[-np.cos(phi)*np.sin(np.pi/2.-theta)],[-np.sin(phi)*np.sin(np.pi/2.-theta)],[np.cos(np.pi/2.-theta)]])[:,0,:]
    east_vec = -np.array([los_vec[1]*north_vec[2]-los_vec[2]*north_vec[1],-los_vec[0]*north_vec[2]+los_vec[2]*north_vec[0], los_vec[0]*north_vec[1]-los_vec[1]*north_vec[0]])
    # large scale field magnetic field
    lon_b0=l0*np.pi/180. ; lat_b0=b0*np.pi/180.
    B0vec=[np.cos(lon_b0)*np.cos(lat_b0),np.sin(lon_b0)*np.cos(lat_b0),np.sin(lat_b0)]
    # Sky map of B field
    Bvec=np.zeros([3,npix])
    Q_cube=np.zeros([N,npix])
    U_cube=np.zeros([N,npix])
    blos_cube=np.zeros([N,npix])
    for j in range(N):
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
        Bvec_perp=np.zeros([3,npix])
        B_times_los=np.zeros([npix]) # scalar producet B.los
        B_times_los=np.sum(Bvec*los_vec,axis=0)
        Bvec_perp[0]=Bvec[0]-los_vec[0]*B_times_los
        Bvec_perp[1]=Bvec[1]-los_vec[1]*B_times_los
        Bvec_perp[2]=Bvec[2]-los_vec[2]*B_times_los
        P_I0=(Bvec_perp[0]**2.+Bvec_perp[1]**2.+Bvec_perp[2]**2.)
        #compute Q anbd U maps
        B_angle=np.zeros([npix])
        Qmap=np.zeros([npix]) ; Umap=np.zeros([npix])
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
        if N == 1 :
            Q_cube=Qmap
            U_cube=Umap
            blos_cube = B_times_los
        if N > 1 :
            Q_cube[j,:]=Qmap
            U_cube[j,:]=Umap
            blos_cube[j,:] = B_times_los
        Bvec[:,:]=0.
    # Preparing Outputs
    if N == 1:
        Qm = Q_cube
        Um = U_cube
        blosm = blos_cube
    if N > 1:
        Qm=Q_cube[0,:].copy()*0.
        Um=Qm.copy()*0.
        blosm =Qm.copy()*0.
        for j in range(N):
            Qm[:]= Qm[:] + Q_cube[j,:]
            Um[:]= Um[:] + U_cube[j,:]
            blosm[:]= blosm[:] + blos_cube[j,:]
        Qm = Qm/(1.*N)
        Um = Um/(1.*N)
        blosm = blosm/(1.*N)
    return -blosm, np.sqrt(Um**2+Qm**2), Qm, Um

### calculating misalignment with MISO

import healpy as hp
from healpy.newvisufunc import projview, newprojplot
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
import cmasher as cmr
from scipy import constants as ko
from scipy.ndimage import label
from astropy.visualization import make_lupton_rgb
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.optimize import curve_fit
import h5py as h5

plt.ion()
plt.show()
font = {'size' : 13}
plt.rc('font', **font)

nside= 64 #smaller is faster
npix = hp.nside2npix(nside)
pix = np.arange(hp.nside2npix(nside))
theta, phi = hp.pix2ang(nside,pix)
glon = phi
glat = np.pi/2. - theta

mis0s = 12 #intrinsic misalignment in the southern Galactic hemisphere
mis0n = 12 #intrinsic misalignment in the northern Galactic hemisphere

l = 73; b = 0  #uniform direction of the mean field
blos, pI, qm, um = SOSDPolpy_fun(l0 = l,b0 = b, nside=nside) #mean field model

l = 73+mis0s; b = 0 #uniform direction of the southern distortion
blos, pI, qm1, um1 = SOSDPolpy_fun(l0 = l,b0 = b, nside=nside) #south field model

l = 73-mis0n; b = 0 #uniform direction of the northern distortion
blos, pI, qm2, um2 = SOSDPolpy_fun(l0 = l,b0 = b, nside=nside) #north field model

# stokes parameters of the local bubble (local ISM)
qlb = qm2*(glat > 0) + qm1*(glat < 0)
ulb = um2*(glat > 0) + um1*(glat < 0)

# fraction of polarization coming from the local field with respect to the total one
fr_m = 0.0

# total stokes parameters
Qtot = qlb*fr_m + qm*(1-fr_m)
Utot = ulb*fr_m + um*(1-fr_m)
diff_fil = ang_diff(Qtot,Utot,qlb,ulb)
seln = np.where((glat > np.deg2rad(60)) ==1)
sels = np.where((glat < np.deg2rad(-60)) ==1)

# plotting misalignment distribution
plt.figure();plt.hist(np.rad2deg(diff_fil[seln]),bins=150,alpha=0.5,range=[-90,90],label='North');plt.xlabel('misalignment angle [deg]');plt.ylabel('counts')
plt.hist(np.rad2deg(diff_fil[sels]),bins=150,alpha=0.1,hatch='//',range=[-90,90],label='South');plt.xlabel('misalignment angle [deg]');plt.ylabel('counts')
plt.axvline(x=0,color='k',ls='dashed')
plt.xlim(-20,20)
plt.legend()
plt.title('intrinsic misalignments = North: {0:0.1f} South: {1:0.1f}  -- Local-to-mean fraction = {2:0.1f}'.format(mis0n,mis0s,fr_m),fontsize=9)
plt.tight_layout()

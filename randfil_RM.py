def powspec2D(im,im1,reso):
    '''
    Computes 2D angular power spectra between the input maps im and im1. nx and nz are the dimensions of the maps.
        '''
    import numpy as np
    import astropy
    import pylab as plt
    from astropy.io import fits
    import matplotlib
    from matplotlib import cm
    import sys
    
    nx = np.shape(im)[0]
    nz = np.shape(im)[1]
    
    x = np.linspace(-1, 1, nx)
    z = np.linspace(-1, 1, nz)
    xx, zz = np.meshgrid(x, z, indexing='ij')
    dk_x = 1./(x[-1]-x[0])
    dk_z = 1./(z[-1]-z[0])
    k_x = np.linspace(-nx/2, nx/2, nx)
    k_z = np.linspace(-nz/2, nz/2, nz)
    kk_x, kk_z = np.meshgrid(k_x, k_z, indexing='ij')

    kk = np.sqrt(kk_x**2 + kk_z**2)
    dk = np.min([dk_x, dk_z])
    k_shells = np.arange(0, np.sqrt(np.max(kk_x**2 + kk_z**2))+dk, dk)
    power = np.zeros(len(k_shells))
    
    im_k = np.fft.fftshift(np.fft.fft2(im))
    im1_k = np.fft.fftshift(np.fft.fft2(im1))
    
    for j, k in enumerate(k_shells):
        mask0 = (abs(kk - k - dk) <= dk)
        power[j] = np.mean((im_k[mask0])*np.conj(im1_k[mask0]))/(nx*nz)
    
    power[np.isnan(power)]=0
    
    return k_shells, power


def compism(spec,nx,ny):
    '''
        Generate a Gaussian random field with power-law power spectrum k^(-spec).
        Packages needed: standard astropy packages.
        '''

    import numpy as np
    import pylab as plt
    from numpy.random import RandomState, SeedSequence
    from numpy.random import MT19937

    # Define the box size.
    x = np.linspace(0, nx-1, nx)
    y = np.linspace(0, ny-1, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    
    # Generate a random afield (white noise).
    phase = np.random.random([nx, ny])*2*np.pi
    
    dk = 1./(x[-1] - x[0])
    k = np.linspace(-1, 1, len(x))*dk*len(x)/2.
    kk_x, kk_y = np.meshgrid(k, k, indexing='ij')
    kk = np.sqrt(kk_x**2 + kk_y**2)
    
    # Apply a fourier filter.
    a_k = kk**(-spec/2.)

    # Apply teh inverse transform.
    m_k = a_k*np.cos(phase) + 1j*a_k*np.sin(phase)
    mout = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(m_k))))
    
    return mout

def ranfil_ab(nax = 256, spec = 3, ndiri = 13, plot=False):
    '''
        Generate a filamentary random field from input Gaussian field with power-law power spectrum k^(-spec).
        Packages needed: standard astropy packages + pywavan (http://github.com/jfrob27/pywavan)
        '''
    mran = compism(spec,nax,nax)
    wt, s11a, wavk, s1a, q = pywavan.fan_trans(mran,reso=1,q=0,qdyn=False,Ndir=ndiri,angular=True)
    prod = 1
    beta = (-spec+1.6)/2.
    for i in range(np.shape(wt)[0]): prod = prod*np.exp(wt[i,:,:,:]/np.std(wt[i,:,:,:])*wavk[i]**(beta))
    sum0  = np.sum(prod,axis=0)
    mfil =np.log10(np.abs(sum0)/np.shape(wt)[1])
    k1,p1 = powspec2D((mfil),(mfil),1)
    k0,p0 = powspec2D((mran),(mran),1)
    pmod= (k0+1e-5)**(-spec)
    if plot == True:
        plt.figure(1)
        plt.loglog(k1,p1/np.percentile(p1,99),label='filaments')
        plt.loglog(k0,p0/np.percentile(p0,99),label='random')
        plt.loglog(k0,pmod/np.percentile(pmod,99),label='input')
        plt.ylim(0,1e4)
        plt.legend()
        plt.figure(2)
        plt.subplot(121);plt.imshow(mran,cmap='coolwarm')
        plt.subplot(122);plt.imshow(mfil,cmap='coolwarm')
        plt.tight_layout()
    res0 = np.polyfit(np.log10(k0[1:len(k0)-1]),np.log10(p0[1:len(k0)-1]),1)
    res1 = np.polyfit(np.log10(k1[1:len(k1)-1]),np.log10(p1[1:len(k1)-1]),1)
    return mran, mfil, k0[1:len(k0)-1],p0[1:len(k0)-1],k1[1:len(k1)-1],p1[1:len(k1)-1], res0[0], res1[0]


def randfil_RM(nax = 256, spec = 1.6, ndiri = 13, sigmaRM = 10, plot = False):
    '''
    Generate random filamentary and Gaussian rotation measure maps with standard deviations defined by sigmaRM and power spectrum defined by the spectral index spec.
    Packages needed: standard astropy packages + pywavan (http://github.com/jfrob27/pywavan)
        '''
    import numpy as np
    import pylab as plt
    import matplotlib
    import astropy
    import scipy
    from scipy import ndimage as nd
    from scipy import stats as st
    from scipy import constants as ko
    from scipy import signal
    import pywavan
    
    mRM_tmp = (ranfil_ab(spec= spec, nax=nax)[1] - ranfil_ab(spec= spec,nax=nax)[1])
    mRM_f = mRM_tmp/np.std(mRM_tmp)*sigmaRM #filamentary RM
    mRM_tmp_r = (ranfil_ab(spec= spec, nax=nax)[0] - ranfil_ab(spec= spec,nax=nax)[0])
    mRM_r = mRM_tmp_r/np.std(mRM_tmp_r)*sigmaRM #Gaussian RM
    
    if plot == True:
        plt.figure(figsize=[8,8])
        plt.subplot(221);plt.imshow(mRM_r,cmap='seismic',origin='lower');plt.colorbar();plt.title(r'Gaussian RM [rad m$^{-2}$]')
        plt.subplot(223);plt.hist(mRM_r.flatten(),bins=100,alpha=0.5,label='Gaussian RM')
        plt.hist(mRM_f.flatten(),bins=100,alpha=0.5,label='Filamentary RM');plt.xlabel(r'RM [rad m$^{-2}$]')
        plt.ylabel('histogram');plt.legend(fontsize=7)
        plt.subplot(222);plt.imshow(mRM_f,cmap='seismic',origin='lower');plt.colorbar();plt.title(r'Filamentary RM [rad m$^{-2}$]')
        k3,p3 = powspec2D((mRM_f),(mRM_f),1);k2,p2 = powspec2D((mRM_r),(mRM_r),1)
        pmod= 100*(k3[1:])**(-spec)
        plt.subplot(224)
        plt.loglog(2*np.pi*k3/(nax),p3/np.percentile(p3,99),label='Filamentary RM')
        plt.loglog(2*np.pi*k2/(nax),p2/np.percentile(p2,99),label='Gaussian RM')
        plt.loglog(2*np.pi*k3[1:]/(nax),pmod/np.percentile(pmod,99),label='input slope')
        plt.ylabel('Normalized power');plt.xlabel(r'$k$ [px$^{-1}$]')
        plt.legend()
        plt.tight_layout()

    return mRM_r,mRM_f

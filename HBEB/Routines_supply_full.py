# (Routines_supply_full.py)
'''
    Contains all the routines for computing observables for given magnetic fields
    in combining E, B with the magnetic helicity of the field. See Bracco et al. 2018
    andrea.bracco@su.se
'''

def dot(aa, bb):
    '''
        Compute the dot product of two 3d vector fields.
        
        call signature:
        
        dot(aa, bb)
        
        Keyword arguments:
        
        *aa, bb*:
        3d fields with 3 components of shape [3, nx, ny, nz]
        '''
    
    import numpy as np
    
    return np.sum(aa*bb, axis=0)


def find_qu(bb, rho, p0, dy, north=[1, 0]):
    '''
        Compute u with line of sight being z and a given projected north.
        
        call signature:
        
        find_qu(bb, rho=1, dz=1, north=[1, 0])
        
        Keyword arguments:
        
        *bb*:
        The magnetic field of shape [3, nx, ny, nz]
        
        *rho*:
        The dust density.
        
        *dy:
        Grid spacing in the direction of integration.
        
        *north*:
        North vector.
        '''
    
    import numpy as np
    
    # Extract the in plane magnetic field component.
    bb_plane = (bb.copy()/np.sqrt(dot(bb, bb)))[::2]
    # Perform the integration.
    qq = np.sum(p0*rho[:,:,:]*(bb_plane[0,:,:,:]**2 - bb_plane[1,:,:,:]**2), axis=1)*dy
    uu = np.sum(p0*rho[:,:,:]*2*bb_plane[0,:,:,:]*bb_plane[1,:,:,:], axis=1)*dy
    ii = np.sum(rho[:,:,:]*(1 - p0*((bb_plane[0,:,:,:]**2 + bb_plane[1,:,:,:]**2) - 2./3.)), axis=1)*dy
    
    return ii, qq, uu


def abc(xx, yy, zz, A, B, C, lam):
    '''
        Create the ABC flow field with maximal helicity.
        
        call signature:
        
        abc(xx, yy, zz, A=1, B=1, C=1, lam=1)
        
        Keyword arguments:
        
        *xx, yy, zz*:
        The full 3d grid positions.
        
        *A, B, C*:
        The ABC-flow parameters.
        
        *lam*:
        The relative helicity.
        '''
    
    import numpy as np
    import math
    
    xx=xx*math.pi
    yy=yy*math.pi
    zz=zz*math.pi
    
    afield = np.zeros([3, xx.shape[0], xx.shape[1], xx.shape[2]])
    afield[0, ...] = A*np.sin((lam)*zz) + C*np.cos((lam)*yy)
    afield[1, ...] = B*np.sin((lam)*xx) + A*np.cos((lam)*zz)
    afield[2, ...] = C*np.sin((lam)*yy) + B*np.cos((lam)*xx)
    

    bfield = lam*afield.copy()
    
    return afield, bfield


def twist(xx, yy, zz, kappa=1, beta=1, bkg=1, theta=0, phi=0, shift=[0,0,0]):
    '''
        Create a twisted magnetic field similar to E3, but only one rotation region.
        
        call signature:
        
        twist(xx, yy, zz, kappa=1, beta=1, bkg=1, theta=0, phi=0)
        
        Keyword arguments:
        
        *xx, yy, zz*:
        The full 3d grid positions.
        
        *kappa*:
        Twist parameter.
        
        *beta*:
        Inverse stretch parameter in z for the twisting region.
        
        *bkg*:
        Strength of the background magnetic field.
        
        *theta*:
        Rotation angle around the y-axis.
        
        *phi*:
        Rotation angle around the z-axis.
        '''
    
    import numpy as np
    
    # Define the rotation matrices.
    Ry = np.matrix([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    Rz = np.matrix([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    MM = Ry*Rz
    
    # Perform the coordinate transformation.
    xx=xx-shift[0]
    yy=yy-shift[1]
    zz=zz-shift[2]
    
    uu = xx*MM.item((0, 0)) + yy*MM.item((0, 1)) + zz*MM.item((0, 2))
    vv = xx*MM.item((1, 0)) + yy*MM.item((1, 1)) + zz*MM.item((1, 2))
    ww = xx*MM.item((2, 0)) + yy*MM.item((2, 1)) + zz*MM.item((2, 2))
    
    # Define the field in the uvw coordinate system.
    factor = kappa*np.exp(-(uu**2 + vv**2 + ww**2/2*beta))
    
    afield_uvw = np.zeros([3, uu.shape[0], vv.shape[1], ww.shape[2]])
    afield_uvw[2, ...] = factor
    
    bfield_uvw = np.zeros([3, uu.shape[0], vv.shape[1], ww.shape[2]])
    bfield_uvw[0, ...] = -2*factor*vv
    bfield_uvw[1, ...] = 2*factor*uu
    
    # Add a homogeneous field.
    afield_uvw[0, ...] += -uu/2*bkg
    afield_uvw[1, ...] += vv/2*bkg
    bfield_uvw[2, ...] += 1*bkg
    
    # Transform the field into the xyz coordinate system
    NN = MM.I
    
    afield = np.zeros([3, xx.shape[0], yy.shape[1], zz.shape[2]])
    bfield = np.zeros([3, xx.shape[0], yy.shape[1], zz.shape[2]])
    
    afield[0] = afield_uvw[0]*NN.item((0, 0)) + afield_uvw[1]*NN.item((0, 1)) + afield_uvw[2]*NN.item((0, 2))
    afield[1] = afield_uvw[0]*NN.item((1, 0)) + afield_uvw[1]*NN.item((1, 1)) + afield_uvw[2]*NN.item((1, 2))
    afield[2] = afield_uvw[0]*NN.item((2, 0)) + afield_uvw[1]*NN.item((2, 1)) + afield_uvw[2]*NN.item((2, 2))
    
    bfield[0] = bfield_uvw[0]*NN.item((0, 0)) + bfield_uvw[1]*NN.item((0, 1)) + bfield_uvw[2]*NN.item((0, 2))
    bfield[1] = bfield_uvw[0]*NN.item((1, 0)) + bfield_uvw[1]*NN.item((1, 1)) + bfield_uvw[2]*NN.item((1, 2))
    bfield[2] = bfield_uvw[0]*NN.item((2, 0)) + bfield_uvw[1]*NN.item((2, 1)) + bfield_uvw[2]*NN.item((2, 2))
    
    return afield, bfield


def column(xx, yy, zz, rho_i=2, rho_e=1, r=1, shift=[0,0,0], theta=0, phi=0):
    '''
        Create a density profile of the shape of a column.
        
        call signature:
        
        column(xx, yy, zz, rho_i=2, rho_e=1, r=1, theta=0, phi=0)
        
        Keyword arguments:
        
        *xx, yy, zz*:
        The full 3d grid positions.
        
        *rho_i*:
        Internal density.
        
        *rho_e*:
        External density.
        
        *r*:
        Column radius.
        
        *theta*:
        Rotation angle around the y-axis.
        
        *phi*:
        Rotation angle around the z-axis.
        '''
    
    import numpy as np
    
    # Define the rotation matrices.
    Ry = np.matrix([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    Rz = np.matrix([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    MM = Ry*Rz
    
    # Perform the coordinate transformation.
    uu = (xx-shift[0])*MM.item((0, 0)) + (yy-shift[1])*MM.item((0, 1)) + (zz-shift[2])*MM.item((0, 2))
    vv = (xx-shift[0])*MM.item((1, 0)) + (yy-shift[1])*MM.item((1, 1)) + (zz-shift[2])*MM.item((1, 2))
    
    rho = (rho_i-rho_e)*np.exp(-(uu**2 + vv**2)/r**2) + rho_e
    
    return rho


def homogeneous(xx, yy, zz, n=[1, 0]):
    '''
    Create a homogeneous magnetic field.

    call signature:

    abc(xx, yy, zz, n=[1, 0])

    Keyword arguments:

    *xx, yy, zz*:
      The full 3d grid positions.

    *n*:
      The directon of the homogeneous magnetic field in the xz-plane.
    '''

    import numpy as np

    # Normalize the directional vector n.
    n = np.array(n)
    n = n/np.linalg.norm(n)

    afield = np.zeros([3, xx.shape[0], xx.shape[1], xx.shape[2]])
    bfield = np.zeros([3, xx.shape[0], xx.shape[1], xx.shape[2]])
    bfield[0, ...] = n[0]
    bfield[2, ...] = n[1]

    return afield, bfield


def polpy(qq,uu,nx,nz):
    '''
    Computes E and B modes maps in 2D given input Stokes Q and U parameter maps.
        '''

    import numpy as np
    import astropy
    import pylab as plt
    from astropy.io import fits
    import matplotlib
    from matplotlib import cm
    import sys
    
    qq_k = np.fft.fftshift(np.fft.fft2(qq))
    uu_k = np.fft.fftshift(np.fft.fft2(uu))
    
    x = np.linspace(-1, 1, nx)
    z = np.linspace(-1, 1, nz)
    dk_x = 1./(x[-1]-x[0])
    dk_z = 1./(z[-1]-z[0])
    k_x = np.linspace(-1, 1, nx)*dk_x*nx/2.
    k_z = np.linspace(-1, 1, nz)*dk_z*nz/2.
    kk_x, kk_z = np.meshgrid(k_x, k_z, indexing='ij')
    kv=np.sqrt(kk_x**2.+kk_z**2.)
    nnz = (kv == 0)
    kv[nnz] = 1e-4
    
    sn=(2.*kk_x*kk_z)/(kv*kv)
    cs=(kk_x**2.-kk_z**2.)/(kv*kv)
    
    sf=sn+1j*0.
    cf=cs+1j*0.
    
    qf=qq_k
    uf=uu_k
    
    ef=(-qf*cf-uf*(sf))
    bf=(+qf*(sf)-uf*cf)
    
    ee = np.real(np.fft.ifft2(np.fft.ifftshift(ef)))
    bb = np.real(np.fft.ifft2(np.fft.ifftshift(bf)))
    
    return ee, bb


def powspec2D(im,im1,nx,nz):
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
    
    x = np.linspace(-1, 1, nx)
    z = np.linspace(-1, 1, nz)
    xx, zz = np.meshgrid(x, z, indexing='ij')
    dk_x = 1./(x[-1]-x[0])
    dk_z = 1./(z[-1]-z[0])
    k_x = np.linspace(-nx/2., nx/2.-1, nx)
    k_z = np.linspace(-nz/2., nz/2.-1, nz)
    kk_x, kk_z = np.meshgrid(k_x, k_z, indexing='ij')

    kk = np.sqrt(kk_x**2 + kk_z**2)
    dk = np.min([dk_x, dk_z])
    k_shells = np.arange(0, np.sqrt(np.max(kk_x**2 + kk_z**2))+dk, dk)
    power = np.zeros(len(k_shells))
    
    im_k = np.fft.fftshift(np.fft.fft2(im))
    im1_k = np.fft.fftshift(np.fft.fft2(im1))
    
    for j, k in enumerate(k_shells):
        mask0 = (abs(kk - k - dk) <= dk)
        power[j] = np.mean((im_k[mask0])*np.conj(im1_k[mask0]))
    
    return k_shells, power


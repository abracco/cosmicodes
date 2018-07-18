# Hessian.py

'''
    Computing the curvature and orientation of the structure of a map based on the eigenvalues of its hessian matrix. See Planck Collaboration int. results. XXXII. 2016
    andrea.bracco@su.se
    '''

def hessian(m_in):
    import numpy as np
    import astropy
    import pylab as plt
    from astropy.io import fits
    import matplotlib
    from matplotlib import cm
    import sys
    import scipy
    from scipy import ndimage as nd

    dx = np.gradient(m_in,axis=0)
    dy = np.gradient(m_in,axis=1)

    dxx=np.gradient(dx,axis=0)
    dyy=np.gradient(dy,axis=1)
    dxy=np.gradient(dx,axis=1)

    dxxf=dxx
    dyyf=dyy
    dxyf=dxy

    c1f=0.5*(dxxf+dyyf-np.sqrt((dxxf-dyyf)**2+4*dxyf**2))
    c2f=0.5*(dxxf+dyyf+np.sqrt((dxxf-dyyf)**2+4*dxyf**2))
    thetaf=0.5*np.arctan2(2*dxyf,dxxf-dyyf)

    return dx, dy, c1f, c2f, thetaf


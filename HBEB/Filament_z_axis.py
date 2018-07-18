# (Filament_z_axis.py)
'''
    Computing E and B modes for density filament with a magnetic field wrapped around it. See Sect. 3.2 in Bracco et al. 2018.
    andrea.bracco@su.se
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

font = {'size'   : 18}

# Import our module with all the routine.
exec(open('Routines_supply_full.py').read())

# Define some constants.
nx = 101
ny = 101
nz = 101 

# Define the box size.
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
z = np.linspace(-1, 1, nz)
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
dx=(x[1]-x[0])
dy=(y[1]-y[0])
dz=(z[1]-z[0])

theta0=0*math.pi/180. # around the y-axis
phi0=0*math.pi/180.  # around the z-axis

# Create the magnetic field with the vector potential. kappa [-inf,inf] beta [0,inf]
afield, bfield = twist(xx, yy, zz, kappa=0., beta=0, bkg=1, theta = theta0, phi = phi0, shift=[0,0,0])

# Create the density distribution.
rho = column(xx, yy, zz, rho_i=2, rho_e=1, r=0.2, shift=[0,0,0], theta = theta0, phi = phi0)

# Compute the column density.
rho_column = np.sum(rho, axis=1)*dy

# Compute Q and U.
p0=0.26
ii, qq, uu = find_qu(bfield, rho=rho, p0=p0, dy=y[1]-y[0], north=[1, 0])
#ii=rho_column

# Compute E and B modes
ee1, bb1 = polpy(qq,uu,nx,nz)

# Compute the line-of-sight integrated magnetic helicity.
hh1 = np.sum(afield*bfield, axis=(0, 2))*(y[1]-y[0])
h_total1 = np.sum(afield*bfield)*(x[1]-x[0])*(y[1]-y[0])*(z[1]-z[0])
# Compute the power spectra.
k_shells, ee_power1 = powspec2D(ee1,ee1,nx,nz)
k_shells, bb_power1 = powspec2D(bb1,bb1,nx,nz)
k_shells, eb_power1 = powspec2D(ee1,bb1,nx,nz)
k_shells, te_power1 = powspec2D(ii,ee1,nx,nz)
k_shells, tb_power1 = powspec2D(ii,bb1,nx,nz)
k_shells, tt_power1 = powspec2D(ii,ii,nx,nz)

el0=np.isfinite(ee_power1)
kg=k_shells[el0]

r_tb1 = tb_power1[el0]/np.sqrt(bb_power1[el0]*tt_power1[el0])
r_eb1 = eb_power1[el0]/np.sqrt(bb_power1[el0]*ee_power1[el0])
r_te1 = te_power1[el0]/np.sqrt(tt_power1[el0]*ee_power1[el0])


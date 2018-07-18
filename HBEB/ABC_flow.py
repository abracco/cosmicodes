# ABC_flow model Bracco et al. 2018 (ABC_flow.py)
'''
    This routine computes the ABC flow model presented in Bracco et al. 2018
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

# Import our module with all the routines.
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


# Create the magnetic field with the vector potential.
lamb=+2
afield0, bfield0 = abc(xx, yy, zz, A=1, B=1, C=1, lam=-lamb)
afield1, bfield1 = abc(xx, yy, zz, A=1, B=1, C=1, lam=+lamb)


# Create the density distribution.
rho = column(xx, yy, zz, rho_i=2, rho_e=1, r=0.2, theta = 0, phi = 0)
rho = rho*0. + 1. # uniform density
rho_column = np.sum(rho, axis=1)

p0=1.
ii0, qq0, uu0 = find_qu(bfield0, rho=rho, p0=p0, dy=y[1]-y[0], north=[1, 0])
ii1, qq1, uu1 = find_qu(bfield1, rho=rho, p0=p0, dy=y[1]-y[0], north=[1, 0])

# Compute E and B modes
ee0, bb0 = polpy(qq0,uu0,nx,nz)
ee1, bb1 = polpy(qq1,uu1,nx,nz)

# Compute the line-of-sight integrated magnetic helicity.
hh0 = np.sum(afield0*bfield0, axis=(0, 2))*(y[1]-y[0])
h_total0 = np.sum(afield0*bfield0)*(x[1]-x[0])*(y[1]-y[0])*(z[1]-z[0])
hh1 = np.sum(afield1*bfield1, axis=(0, 2))*(y[1]-y[0])
h_total1 = np.sum(afield1*bfield1)*(x[1]-x[0])*(y[1]-y[0])*(z[1]-z[0])

# Compute the power spectra.
k_shells, ee_power0 = powspec2D(ee0,ee0,nx,nz)
k_shells, bb_power0 = powspec2D(bb0,bb0,nx,nz)
k_shells, ee_power1 = powspec2D(ee1,ee1,nx,nz)
k_shells, bb_power1 = powspec2D(bb1,bb1,nx,nz)
k_shells, eb_power0 = powspec2D(ee0,bb0,nx,nz)
k_shells, eb_power1 = powspec2D(ee1,bb1,nx,nz)
k_shells, te_power0 = powspec2D(ii0,ee0,nx,nz)
k_shells, te_power1 = powspec2D(ii1,ee1,nx,nz)
k_shells, tb_power0 = powspec2D(ii0,bb0,nx,nz)
k_shells, tb_power1 = powspec2D(ii1,bb1,nx,nz)
k_shells, tt_power0 = powspec2D(ii0,ii0,nx,nz)
k_shells, tt_power1 = powspec2D(ii1,ii1,nx,nz)

el0=np.isfinite(eb_power0)
kg=k_shells[el0]

r_tb0 = tb_power0[el0]/np.sqrt(bb_power0[el0]*tt_power0[el0])
r_tb1 = tb_power1[el0]/np.sqrt(bb_power1[el0]*tt_power1[el0])
r_eb0 = eb_power0[el0]/np.sqrt(bb_power0[el0]*ee_power0[el0])
r_eb1 = eb_power1[el0]/np.sqrt(bb_power1[el0]*ee_power1[el0])
r_te0 = te_power1[el0]/np.sqrt(tt_power1[el0]*ee_power1[el0])

plt.ion()
plt.figure()
plt.rc('font', **font)
plt.plot(kg,r_tb0,'y--', label = r'$r_k^{{TB}}$, $H = \pm${0:.0f}'.format(np.abs(h_total0)))
plt.plot(kg,r_te0,'g--', alpha=0.4, label = r'$r_k^{{TE}}$, $H = \pm${0:.0f}'.format(np.abs(h_total0)))
plt.plot(kg,ee_power0[el0]/np.max(ee_power0[el0]),'b-', label = r'$C^{{EE}}_{{k}}/$max$(C^{{EE}}_{{k}})$, $H = \pm${0:.0f}'.format(np.abs(h_total0)))
plt.plot(kg,bb_power0[el0]/np.max(bb_power0[el0]),'r-', label = r'$C^{{BB}}_{{k}}/$max$(C^{{BB}}_{{k}})$, $H = \pm${0:.0f}'.format(np.abs(h_total0)))
plt.ylim(-1,1)
plt.xlim(-1,50)
plt.xlabel(r'$k$')
plt.legend()
plt.grid(linestyle='dotted')




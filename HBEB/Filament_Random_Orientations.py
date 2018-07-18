# (Filament_Random_Orientations.py)
'''
    Computing E and B modes for density filament with a magnetic field wrapped around it for 100 random orientations with respect to the line of sight. See Sect. 3.2 in Bracco et al. 2018.
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
exec(open('Hessian.py').read())

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

plt.ion()

nr=100
kap_grid = np.arange(-10,11,1)
p0=0.26
pie=np.zeros([len(kap_grid),nr])
pib=np.zeros([len(kap_grid),nr])

thv=np.zeros([nr])
phv=np.zeros([nr])

for k in range(0,nr):
    theta0=np.random.random([1])*(90)*math.pi/180. # around the y-axis
    phi0=np.random.random([1])*(180)*math.pi/180.  # around the z-axis

# Create the magnetic field with the vector potential. kappa [-inf,inf] beta [0,inf]
# Create the density distribution.
    thv[k]=theta0*180/math.pi
    phv[k]=phi0*180/math.pi
    
    rho = column(xx, yy, zz, rho_i=2, rho_e=1, r=0.2, shift=[0,0,0], theta = theta0, phi = phi0)

# Compute the column density.
    rho_column = np.sum(rho, axis=1)*dy

    print(theta0*180/math.pi, phi0*180/math.pi, k)
    for j in range(0,len(kap_grid)):
        kappa=kap_grid[j]
        afield, bfield = twist(xx, yy, zz, kappa=kappa, beta=0, bkg=1., theta = theta0[0], phi = phi0[0], shift=[0,0,0])
        ii, qq, uu = find_qu(bfield, rho=rho, p0=p0, dy=y[1]-y[0], north=[1, 0])
        devx, devy, curv1, curv2, theta = hessian(ii)
        ss1=(curv1 <= curv1.min()*0.95)
        beta = math.pi/2. + np.mean(theta[ss1])
        qq1 = qq*np.cos(2.*beta) + uu*np.sin(2.*beta)
        uu1 = -qq*np.sin(2.*beta) + uu*np.cos(2.*beta)
        qq1 = qq*np.cos(2.*beta) + uu*np.sin(2.*beta)
        uu1 = -qq*np.sin(2.*beta) + uu*np.cos(2.*beta)
        
        ii=rho_column.copy()
        ss=np.where(ii >= 0.95*ii.max())
        pie[j,k] = st.pearsonr(ii[ss],-qq1[ss])[0]
        pib[j,k] = st.pearsonr(ii[ss],-uu1[ss])[0]
        if  np.max(np.abs(uu1)) < 1e-2:
            pib[j,k]=0
        if  np.max(np.abs(qq1)) < 1e-2:
            pie[j,k]=0

pem = np.zeros([len(kap_grid)])
spem=pem.copy()*0.
pbm = np.zeros([len(kap_grid)])
spbm=pem.copy()*0.

for i in range(0,len(kap_grid)):
    pem[i]=np.mean(pie[i,:])
    spem[i]=np.std(pie[i,:])
    pbm[i]=np.mean(pib[i,:])
    spbm[i]=np.std(pib[i,:])

plt.figure()
plt.rc('font', **font)
plt.plot(kap_grid,pem,'bo',label = 'Stokes $I$, $E$ modes')
plt.plot(kap_grid,pem,'b-')
plt.fill_between(kap_grid,pem-spem,pem+spem,facecolor='blue',alpha=0.3)
plt.plot(kap_grid,pbm,'r-')
plt.plot(kap_grid,pbm,'ro',label = 'Stokes $I$, $B$ modes')
plt.fill_between(kap_grid,pbm-spbm,pbm+spbm,facecolor='red',alpha=0.3)
plt.grid(linestyle='dotted')
plt.xlabel(r'$\alpha_b$')
plt.ylabel('Pearson coefficient')
plt.legend()
plt.ylim(-1.1,1.1)
plt.xlim(-11,11)





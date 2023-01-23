
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
from astropy.utils.data import get_pkg_data_filename

exec(open('Planck_routines.py').read())
exec(open('SymPolpy_fun.py').read())
plt.ion()

nside=256
blos, Is, Qs, Us, nuv = SymPolpy_fun(nside,30,70,24,1,5,2,400e6,800e6)
cl0=hp.sphtfunc.anafast(Is[:,0],lmax=2*nside+1)
el0=np.arange(np.shape(cl0)[0])
#plt.figure();
plt.loglog(el0,cl0,'.-',label='output spectrum')
plt.xlabel('multipole')
plt.ylabel('power [a.u.]')


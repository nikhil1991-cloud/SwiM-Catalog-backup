from matplotlib import pyplot as plt
from astropy.io import fits
import scipy.stats as st
import math
from scipy import signal
from astropy import wcs
import sys
from scipy import interpolate
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
import scipy.stats as st
from scipy import signal
from astropy import wcs
import sys
from scipy import interpolate
from numpy import inf
from astropy.io.fits import getheader
from astropy.utils.data import get_pkg_data_filename
from reproject import reproject_interp
from reproject import reproject_exact
import time
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import scipy.interpolate as spi
import pandas as pd

corr_fac = (1.000)
mf = pd.read_csv('/Users/nikhil/code/Newtext/Matchtxt/spec.txt', comment='#', header=None, delim_whitespace=True)
Indices = np.array(mf[2])
H_Blue = np.array(mf[9])
L_Blue = np.array(mf[8])
H_Red = np.array(mf[13])
L_Red = np.array(mf[12])
H_Index = np.array(mf[5])
L_Index = np.array(mf[4])
Unit_indx = np.array(mf[16])
KJ = np.where(Unit_indx == 'mag')
KL = Unit_indx*0
KL[KJ] = 1
Rmid = (H_Red + L_Red)/2
Bmid = (H_Blue + L_Blue)/2
Imid = (H_Index + L_Index)/2
Delt_In = H_Index - L_Index
Delt_R = H_Red - L_Red
Delt_B = H_Blue - L_Blue

start_time = time.time()
with open('/Users/Nikhil/code/Newtext/Matchtxt/swim_ba_agn_cut.txt') as f:
   Line = [line.rstrip('\n') for line in open('/Users/Nikhil/code/Newtext/Matchtxt/swim_ba_agn_cut.txt')]


q=0
KT = 1
for q in range (0,np.shape(Line)[0]):
    one = 1
#path to drpall
    drpall = fits.open('/volumes/Nikhil/MPL-7_Files/MaNGAPipe3D/Newmanga/drpall-v2_3_1.fits')
    tbdata = drpall[1].data
    ind = np.where(tbdata['mangaid'] == Line[q])
    objectra = tbdata['objra'][ind][0]
    objectdec = tbdata['objdec'][ind][0]
    redshift = tbdata['nsa_z'][ind][0]
    plate = tbdata['plate'][ind][0]
    ifu = tbdata['ifudsgn'][ind][0]
    sloan = tbdata['nsa_iauname'][ind][0]
    axs = tbdata['nsa_elpetro_ba'][ind][0]
    pa = tbdata['nsa_elpetro_phi'][ind][0]
    Ref = tbdata['NSA_ELPETRO_TH50_R'][ind][0]
    #plt.scatter(axs,q,s=2,c='r')
    #if axs>0.6:
    #   print(Line[q])
    

    hdu = fits.open("/volumes/Nikhil/MPL-7_Files/SwiM_binned/SwiM_"+str(Line[q])+".fits")
    #hdu = fits.open("/Users/nikhil/Data/SWIFT/MPL-7_W/SwiM_"+str(Line[q])+".fits")
    D4000 = hdu[18].data[43]
    HD = hdu[18].data[21]
    FE = hdu[18].data[13]
    FE1 = hdu[18].data[4]
    FE2 = hdu[18].data[14]
    GMI = (22.5 - 2.5*np.log10(hdu[22].data[4])) - (22.5 - 2.5*np.log10(hdu[22].data[6]))
    UMI = (22.5 - 2.5*np.log10(hdu[22].data[3]))-(22.5 - 2.5*np.log10(hdu[22].data[6]))
    W2MU = (22.5 - 2.5*np.log10(hdu[22].data[0])) - (22.5 - 2.5*np.log10(hdu[22].data[3]))
    W1MU = (22.5 - 2.5*np.log10(hdu[22].data[1])) - (22.5 - 2.5*np.log10(hdu[22].data[3]))
    GMR = (22.5 - 2.5*np.log10(hdu[22].data[4]))-(22.5 - 2.5*np.log10(hdu[22].data[5]))
    UMZ = (22.5 - 2.5*np.log10(hdu[22].data[3])) - (22.5 - 2.5*np.log10(hdu[22].data[7]))
    #errors
    ED4 = hdu[19].data[43]
    EHD = hdu[19].data[21]*0.33
    EFE = hdu[19].data[13]*0.33
    EFE1 = hdu[19].data[4]*0.33
    EFE2 = hdu[19].data[14]*0.33
    EW2 = hdu[23].data[0]
    EW1 = hdu[23].data[1]
    EU = hdu[23].data[3]
    EG = hdu[23].data[4]
    ER = hdu[23].data[5]
    EI = hdu[23].data[6]
    EZ = hdu[23].data[7]



    plt.subplot(2,2,1)
    plt.scatter(D4000,HD,s=10,c='r')
    plt.xlabel('Dn4000')
    plt.ylabel('HD')
    plt.xlim([1.15,2.25])
    plt.ylim([-4,8])
    
    plt.subplot(2,2,2)
    plt.scatter(W2MU,HD,s=10,c='r')
    plt.xlabel('W2-u')
    plt.ylabel('HD')
    plt.xlim([-0.7,5.2])
    plt.ylim([-4,8])
    
    plt.subplot(2,2,3)
    plt.scatter(D4000,FE,s=10,c='r')
    plt.xlabel('Dn4000')
    plt.ylabel('Fe4383')
    plt.ylim([1.3,5])
    plt.xlim([1.15,2.25])
    
    plt.subplot(2,2,4)
    plt.scatter(W2MU,GMI,s=10,c='r')
    plt.xlabel('W2-u')
    plt.ylabel('g-i')
    plt.xlim([-0.7,5.2])
    plt.ylim([0.4,1.7])

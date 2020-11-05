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
corr_factor = 1.000
start_time = time.time()
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
with open('/Users/Nikhil/code/Newtext/Matchtxt/W2W1M2TOT.txt') as f:
   Line = [line.rstrip('\n') for line in open('/Users/Nikhil/code/Newtext/Matchtxt/W2W1M2TOT.txt')]


q=0
for q in range (0,np.shape(Line)[0]):
    #path to drpall, read plate,ifu,mangaid,z, ra, dec for the galaxy
    drpall = fits.open('/Users/Nikhil/Data/MaNGAPipe3D/Newmanga/drpall-v2_3_1.fits')
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

    #Define psfs of Swift
    FWHMar = np.array([2.92/2.355,2.45/2.355,2.37/2.355])
    SwftSigmaw2 = FWHMar[0]
    SwftSigmam2 = FWHMar[1]
    SwftSigmaw1 = FWHMar[2]
    
    #Read spectra from DRP Logcube
    hdu = fits.open("/Volumes/Nikhil/Data/LOGCUBE/manga"+"-" + str(plate) + "-" + str(ifu)+"-"+"LOGCUBE.fits")
    ifura = hdu[0].header['IFURA']
    ifudec = hdu[0].header['IFUDEC']
    Good_bit = 2**10
    Flux_spectra = hdu['FLUX'].data #Read spectra
    Mask_spectra = hdu['MASK'].data #Read Mask
    Index_S = np.where(Mask_spectra>=0)
    Index_S_N = np.where(Mask_spectra<=0)
    Mask_spectra[Index_S] = 1
    Mask_spectra[Index_S_N] = 0
    Variance_spectra = 1/hdu['IVAR'].data #Read variance
    Variance_spectra[np.isnan(Variance_spectra)]=0
    Variance_spectra[Variance_spectra == -inf] = 0
    Variance_spectra[Variance_spectra == inf] = 0
    RSeeing = hdu[0].header['RFWHM'] #Read seeing FWHM
    SigmaM = RSeeing/2.355 #Convert to sigma
    Predisp = hdu['PREDISP'].data #Read instrumental resolution of MaNGA
    Waveh = hdu['WAVE'].data
    #Read velocities, dispersions, spec index masks and emission line EWs and fluxes and its vars and masks
    hdu = fits.open("/Volumes/Nikhil/HYB10-GAU-MILESHC/"+str(plate)+"/"+str(ifu)+"/"+"manga"+"-"+ str(plate) + "-" + str(ifu)+"-MAPS-HYB10-GAU-MILESHC.fits.gz")
    SP_MASK = hdu['SPECINDEX_MASK'].data
    SP_D = hdu['SPECINDEX'].data
    SP_S = np.sqrt(1/hdu['SPECINDEX_IVAR'].data)
    I_SP = np.where(SP_MASK>Good_bit)
    I_SP_N = np.where(SP_MASK<=Good_bit)
    SP_MASK[I_SP]=1
    SP_MASK[I_SP_N]=0
    H_manga = hdu['STELLAR_VEL'].header
    Vel = hdu['STELLAR_VEL'].data #Stellar velocity
    Shift = 1 + redshift + Vel/299792 #Define shift
    Vel_sigma_corr = hdu['STELLAR_SIGMACORR'].data
    Vel_sigma = hdu['STELLAR_SIGMA'].data
    Vel_sigma_err = 1/hdu['STELLAR_SIGMA_IVAR'].data
    Vel_sigma_mask = hdu['STELLAR_SIGMA_MASK'].data
    Vel_index = np.where(Vel_sigma_mask>Good_bit)
    Vel_index_N = np.where(Vel_sigma_mask<=Good_bit)
    Vel_sigma_mask[Vel_index] = 1
    Vel_sigma_mask[Vel_index_N] = 0
    Astro_sigma_mnga = np.ma.array((Vel_sigma**2 - Vel_sigma_corr**2),mask=Vel_sigma_mask)
    #Read Emission Line fluxes
    GFLUX = hdu['EMLINE_GFLUX'].data
    GEW = hdu['EMLINE_GEW'].data
    GFLUX_VAR = (1/hdu['EMLINE_GFLUX_IVAR'].data)
    GFLUX_VAR[np.isnan(GFLUX_VAR)] = 0
    GFLUX_VAR[GFLUX_VAR == inf] = 0
    GFLUX_VAR[GFLUX_VAR == -inf] = 0
    GEW_VAR = 1/hdu['EMLINE_GEW_IVAR'].data
    GEW_VAR[np.isnan(GEW_VAR)] = 0
    GEW_VAR[GEW_VAR == inf] = 0
    GEW_VAR[GEW_VAR == -inf] = 0
    GFMASK = hdu['EMLINE_GFLUX_MASK'].data
    GEWMASK = hdu['EMLINE_GEW_MASK'].data
    Index_F = np.where(GFMASK>Good_bit)
    Index_E = np.where(GFMASK>Good_bit)
    Index_F_N = np.where(GFMASK<=Good_bit)
    Index_E_N = np.where(GFMASK<=Good_bit)
    GFMASK[Index_F] = 1
    GEWMASK[Index_E] = 1
    GFMASK[Index_F_N] = 0
    GEWMASK[Index_E_N] = 0
    GFLUX_Clip = np.zeros(hdu['EMLINE_GFLUX'].data.shape)
    GEW_Clip = np.zeros(hdu['EMLINE_GEW'].data.shape)
    p=0
    for p in range (0,np.shape(GFLUX)[0]):
           GFLUX_Clip[p,:,:] = np.clip(GFLUX[p,:,:],0,np.max(GFLUX[p,:,:]))
           GEW_Clip[p,:,:] = np.clip(GEW[p,:,:],0,np.max(GEW[p,:,:]))

    #Derive continuum
    GCONT_Clip = GFLUX_Clip/GEW_Clip
    GCONT_Clip[np.isnan(GCONT_Clip)] = 0
    GCONT_Clip[GCONT_Clip == inf] = 0
    GCONT_Clip[GCONT_Clip == -inf] = 0
    
    #Read best fit emission spectra from DAP Logcube
    hdu = fits.open("/Volumes/Nikhil/Data/LOGCUBE/manga"+"-" + str(plate) + "-" + str(ifu)+"-LOGCUBE-HYB10-GAU-MILESHC.fits")
    Emission = hdu['EMLINE'].data
    Emission_base = hdu['EMLINE_BASE'].data
    Continuum_Flux = Flux_spectra - Emission - Emission_base
    Continuum_Flux_Rest = np.zeros(Continuum_Flux.shape)
    Predisp_Rest = np.zeros(Continuum_Flux.shape)

    i=0
    for i in range (0,np.shape(Continuum_Flux)[1]):
           j = 0
           for j in range (0,np.shape(Continuum_Flux)[1]):
              Waven = Waveh/Shift[i,j]
              Y_naught = Continuum_Flux[:,i,j]*Shift[i,j]
              Y_PD = (Predisp[:,i,j]/Waveh)*(299792)
              X_naught = Waven
              Z = spi.interp1d(X_naught,Y_naught,fill_value="extrapolate")
              K_predisp = spi.interp1d(X_naught,Y_PD,fill_value="extrapolate")
              Continuum_Flux_Rest[:,i,j] = Z(Waveh)
              Predisp_Rest[:,i,j] = K_predisp(Waveh)
    
    Rest_Flux = np.ma.array(Continuum_Flux_Rest,mask=Mask_spectra)
    Rest_Predisp = np.ma.array(Predisp_Rest,mask=Mask_spectra)
    Spec_variance = np.ma.array(Variance_spectra,mask=Mask_spectra)
    #Calculate Dn4000
    cvel = 2.997*math.pow(10,10)
    #Red window
    H1_R = np.where(Waveh < 4000)
    L1_R = np.where(Waveh > 4100)
    H10_R = np.max(H1_R)
    L10_R = np.min(L1_R)
    ar1_R = Waveh[H10_R:L10_R]
    Numerator = Rest_Flux[H10_R:L10_R,:,:]
    NUM = Numerator*(0)
    i=0
    for i in range (0,len(Numerator)):
        NUM[i,:,:] = ((ar1_R[i])**2)*Numerator[i,:,:]/cvel
    #Blue window
    H2_B = np.where(Waveh < 3850)
    L2_B = np.where(Waveh > 3950)
    H20_B = np.max(H2_B)
    L20_B = np.min(L2_B)
    ar2_B = Waveh[H20_B:L20_B]
    Denominator = Rest_Flux[H20_B:L20_B,:,:]
    DENO = Denominator*(0)
    i=0
    for i in range (0,len(Denominator)):
        DENO[i,:,:] = ((ar2_B[i])**2)*Denominator[i,:,:]/cvel
    NNUM= np.trapz(NUM,x=ar1_R,axis=0)
    NDENO = np.trapz(DENO,x=ar2_B,axis=0)
    
    N_VAR = Spec_variance[H10_R:L10_R,:,:]
    D_VAR = Spec_variance[H20_B:L20_B,:,:]
    N_ER = np.zeros(np.shape(N_VAR))
    D_ER = np.zeros(np.shape(D_VAR))
    i=0
    for i in range (0,len(N_VAR)):
        N_ER[i,:,:] = (ar1_R[i]**2)*np.sqrt(N_VAR[i,:,:])/cvel
    i=0
    for i in range (0,len(D_VAR)):
        D_ER[i,:,:] = (ar2_B[i]**2)*np.sqrt(D_VAR[i,:,:])/cvel
    N_ERS = np.trapz(N_ER**2,x=ar1_R,axis=0)
    D_ERS = np.trapz(D_ER**2,x=ar2_B,axis=0)
    
    #Spectral Indices calculation
    SPEC_Index_Mask = np.zeros((np.shape(mf)[0] ,np.shape(Rest_Flux)[1],np.shape(Rest_Flux)[1]))
    SPEC_Index_Flux = np.zeros((np.shape(mf)[0] ,np.shape(Rest_Flux)[1],np.shape(Rest_Flux)[1]))
    SPEC_Index_Cont = np.zeros((np.shape(mf)[0] ,np.shape(Rest_Flux)[1],np.shape(Rest_Flux)[1]))
    SPEC_Index_Flux_sigma = np.zeros((np.shape(mf)[0] ,np.shape(Rest_Flux)[1],np.shape(Rest_Flux)[1]))
    SPEC_Index_Cont_sigma = np.zeros((np.shape(mf)[0] ,np.shape(Rest_Flux)[1],np.shape(Rest_Flux)[1]))
    COMB_AVG = np.zeros((np.shape(mf)[0] ,np.shape(Rest_Flux)[1],np.shape(Rest_Flux)[1]))
    COMB_AVG_MASK = np.zeros((np.shape(mf)[0] ,np.shape(Rest_Flux)[1],np.shape(Rest_Flux)[1]))
    COMB_AVG_SIGMA = np.zeros((np.shape(mf)[0] ,np.shape(Rest_Flux)[1],np.shape(Rest_Flux)[1]))
    l =0
    for l in range (0,np.shape(mf)[0]):
       # # Blue band selection
       High_Blue = np.where(Waveh < L_Blue[l])
       Low_Blue = np.where(Waveh > H_Blue[l])
       HB0 = np.max(High_Blue)
       LB0 = np.min(Low_Blue)
       Wave_Blue = Waveh[HB0:LB0]
       Dlambda_Blue = Waveh[LB0] - Waveh[HB0]
       mid_Blue = np.int(np.shape(Wave_Blue)[0]/2)
       MB = np.where(Waveh == Wave_Blue[mid_Blue] )
       Flux_Blue = Rest_Flux[HB0:LB0]
       Cntm_Blue = np.trapz(Flux_Blue,x=Wave_Blue,axis=0)/Delt_B[l]
       Var_Blue = np.trapz(Spec_variance[HB0:LB0,:,:],x=Wave_Blue,axis=0)/(Delt_B[l]**2)
       # # Red band selection
       High_Red = np.where(Waveh < L_Red[l])
       Low_Red = np.where(Waveh > H_Red[l])
       HR0 = np.max(High_Red)
       LR0 = np.min(Low_Red)
       Wave_Red = Waveh[HR0:LR0]
       Dlambda_Red = (Waveh[LR0] - Waveh[HR0])
       mid_Red = np.int(np.shape(Wave_Red)[0]/2)
       MR = np.where(Waveh == Wave_Red[mid_Red] )
       Flux_Red = Rest_Flux[HR0:LR0]
       Cntm_Red = np.trapz(Flux_Red,x=Wave_Red,axis=0)/Delt_R[l]
       Slope = (Cntm_Red - Cntm_Blue)/(Rmid[l]-Bmid[l])
       Var_Red = np.trapz(Spec_variance[HR0:LR0,:,:],x=Wave_Red,axis=0)/(Delt_R[l]**2)
       # # Index band selection
       High_Index = np.where(Waveh < L_Index[l])
       Low_Index = np.where(Waveh > H_Index[l])
       HI0 = np.max(High_Index)
       LI0 = np.min(Low_Index)
       Wave_Index = Waveh[HI0:LI0]
       Dlambda_Index = Waveh[LI0] - Waveh[HI0]
       mid_Index = np.int(np.shape(Wave_Index)[0]/2)
       IndexFlux = Rest_Flux[HI0:LI0,:,:]
       Dlambda = Wave_Index.max() - Wave_Index.min()
       Var_Index = np.trapz(Spec_variance[HI0:LI0,:,:],x=Wave_Index,axis=0)
       Continua = Slope*(Imid[l]-Bmid[l]) + Cntm_Blue
       K_lambda = (Imid[l] - Bmid[l])/(Rmid[l]-Bmid[l])
       Var_Conti = (Var_Red)*(K_lambda**2) + (Var_Blue)*((K_lambda-1)**2)
       LGCB_EW = Delt_In[l] - np.trapz(IndexFlux/Continua,x=Wave_Index,axis=0)
       SPEC_Index_Flux[l,:,:] = np.trapz(IndexFlux,x=Wave_Index,axis=0)
       SPEC_Index_Cont[l,:,:] = Continua
       SPEC_Index_Flux_sigma[l,:,:] = Var_Index
       SPEC_Index_Cont_sigma[l,:,:] = Var_Conti
       SPEC_Index_Mask[l,:,:] = SP_MASK[l]
       
       IPK = np.min(np.where(Waveh>Imid[l]))
       C_DISP = Rest_Predisp[IPK,:,:]**2 + Astro_sigma_mnga
       COMB_AVG[l] = C_DISP*SPEC_Index_Flux[l]
       COMB_AVG_SIGMA[l] = (SPEC_Index_Flux[l]*Vel_sigma)**2 * Vel_sigma_err
       COMB_AVG_MASK[l] = np.logical_or(SP_MASK[l],Vel_sigma_mask)

    #Define convolution kernels
    sm = np.sqrt((SwftSigmaw2-0.0419)**2 - SigmaM**2)
    dmnga = 0.5
    lmnga = math.ceil(3*sm/dmnga)
    lw1 = math.ceil(SwftSigmaw1)
    dw1=1
    lm2 = math.ceil(SwftSigmam2)
    dm2=1
    
    #Read W2
    hdu_w2 = fits.open(get_pkg_data_filename("/Volumes/Nikhil/Data/SWIFT/Flux/"+str(Line[q])+"_UVW2_flx.fits"))[0]
    wcs_w2 = WCS(hdu_w2.header)
    pcw2_1,pcw2_2 = wcs_w2.wcs_world2pix(objectra,objectdec,1)
    #Read M2
    hdu_m2 = fits.open(get_pkg_data_filename("/Volumes/Nikhil/Data/SWIFT/Flux/"+str(Line[q])+"_UVM2_flx.fits"))[0]
    wcs_m2 = WCS(hdu_m2.header)
    pcm2_1,pcm2_2 = wcs_m2.wcs_world2pix(objectra,objectdec,1)#Calculate frac pixel diff between W2 and M2
    dxm2,dym2 = pcw2_1 - pcm2_1,pcw2_2 - pcm2_2
    frac_dxm2,frac_dym2 = dxm2 - np.floor(dxm2),dym2 - np.floor(dym2)
    dist_m2 = np.sqrt((frac_dxm2 - 0.5)**2 + (frac_dym2 - 0.5)**2)
    Dsigma_m2 = -0.03*(dist_m2**4) + 0.081*(dist_m2**3) - 0.25*(dist_m2**2) - 0.006*(dist_m2) + 0.096
    if Dsigma_m2<0:
       Dsigma_m2=0
    #Read W1
    hdu_w1 = fits.open(get_pkg_data_filename("/Volumes/Nikhil/Data/SWIFT/Flux/"+str(Line[q])+"_UVW1_flx.fits"))[0]
    wcs_w1 = WCS(hdu_w1.header)
    pcw1_1,pcw1_2 = wcs_w1.wcs_world2pix(objectra,objectdec,1)#Calculate frac pixel diff between W2 and W1
    dxw1,dyw1 = pcw2_1 - pcw1_1,pcw2_2 - pcw1_2
    frac_dxw1,frac_dyw1 = dxw1 - np.floor(dxw1),dyw1 - np.floor(dyw1)
    dist_w1 = np.sqrt((frac_dxw1 - 0.5)**2 + (frac_dyw1 - 0.5)**2)
    Dsigma_w1 = -0.03*(dist_w1**4) + 0.081*(dist_w1**3) - 0.25*(dist_w1**2) - 0.006*(dist_w1) + 0.096
    if Dsigma_w1<0:
       Dsigma_w1=0
    #Manga - uvw2 kernel
    x_mnga = np.arange((-lmnga)*dmnga,(lmnga*dmnga)+dmnga,step=dmnga)
    XM,YM = np.meshgrid(x_mnga,x_mnga)
    KM = np.exp(-(XM ** 2 + YM ** 2) / (2 * sm ** 2))
    Ga = KM/np.sum(KM)
    #uvw1 - uvw2 kernel
    sw1 = np.sqrt((SwftSigmaw2-Dsigma_w1)**2 - SwftSigmaw1**2)
    x_swft_W1 = np.arange(-lw1,lw1+dw1,dw1)
    XW1,YW1 = np.meshgrid(x_swft_W1,x_swft_W1)
    KW1 = np.exp(-(XW1 ** 2 + YW1 ** 2) / (2 * sw1 ** 2))
    Ga1 = KW1/np.sum(KW1)
    #uvm2 - uvw2 kernel
    sm2 = np.sqrt((SwftSigmaw2-Dsigma_m2)**2 - SwftSigmam2**2)
    x_swft_M2 = np.arange(-lm2,lm2+dm2,dm2)
    XM2,YM2 = np.meshgrid(x_swft_M2,x_swft_M2)
    KM2 = np.exp(-(XM2 ** 2 + YM2 ** 2) / (2 * sm2 ** 2))
    Ga2 = KM2/np.sum(KM2)
    
    #Calculating correlation and covariance matrix for the manga maps
    Nifu = np.shape(GFLUX)[1]
    Npix = Nifu**2
    hdu = fits.open("/Volumes/Nikhil/Data/SWIFT/MPL-7_Dist/Dist_"+str(Nifu)+".fits")
    RHO = hdu[0].data
    
    #Put in for loop for elines fluxes/ews and spec_indices
    COV_EF = np.zeros((np.shape(GFLUX)[0],Npix,Npix))
    COV_EL = np.zeros((np.shape(GFLUX)[0],Npix,Npix))
    COV_IF = np.zeros((np.shape(SPEC_Index_Flux_sigma)[0],Npix,Npix))
    COV_IC = np.zeros((np.shape(SPEC_Index_Flux_sigma)[0],Npix,Npix))
    COV_D4 = np.zeros((2,Npix,Npix))
    
    #generate cov_ij for derived quantities
    em=0
    for em in range (0,np.shape(COV_EF)[0]):
        CEFVAR_J = np.repeat(GFLUX_VAR[em].reshape(Npix)[:,np.newaxis],Npix,axis=1)
        CELVAR_J = np.repeat(GEW_VAR[em].reshape(Npix)[:,np.newaxis],Npix,axis=1)
        COV_EF[em] = RHO*np.sqrt(CEFVAR_J*np.transpose(CEFVAR_J))
        COV_EL[em] = RHO*np.sqrt(CELVAR_J*np.transpose(CELVAR_J))
    
    sp=0
    for sp in range (0,np.shape(COV_IF)[0]):
        CIFVAR_J = np.repeat(SPEC_Index_Flux_sigma[sp].reshape(Npix)[:,np.newaxis],Npix,axis=1)
        CICVAR_J = np.repeat(SPEC_Index_Cont_sigma[sp].reshape(Npix)[:,np.newaxis],Npix,axis=1)
        COV_IF[sp] = RHO*np.sqrt(CIFVAR_J*np.transpose(CIFVAR_J))
        COV_IC[sp] = RHO*np.sqrt(CICVAR_J*np.transpose(CICVAR_J))
    
    D4RVAR = N_ERS.reshape(Npix)
    D4BVAR = D_ERS.reshape(Npix)
    CD4RVAR_J = np.repeat(D4RVAR[:,np.newaxis],Npix,axis=1)
    CD4BVAR_J = np.repeat(D4BVAR[:,np.newaxis],Npix,axis=1)
    COV_D4[0] = RHO*np.sqrt(CD4RVAR_J*np.transpose(CD4RVAR_J))
    COV_D4[1] = RHO*np.sqrt(CD4BVAR_J*np.transpose(CD4BVAR_J))
    
    #Calculating W (the derivative matric for psf convolution)
    m=Nifu
    P_conv = np.shape(Ga)[0]
    W_CONV = np.zeros((Npix,Npix))
    q0=0
    for q0 in range (0,Npix):
        z0=0
        for z0 in range (0,P_conv):
            i0 = int((P_conv-1)/2)
            start = q0-(i0-z0)*m-i0
            finish = q0-(i0-z0)*m+i0+1
            start2 = np.min([np.max([start,0]),Npix])
            finish2 = np.min([np.max([finish,0]),Npix])
            W_CONV[q0,start2:finish2] = Ga[z0,start2-start:P_conv+finish2-finish]

    
    
    #Read wcs and data for UVW2
    radec = np.array([objectra,objectdec])
    S_cutout = (int(np.shape(GFLUX)[1]/2)+1,int(np.shape(GFLUX)[1]/2)+1)
    hdu = fits.open("/Volumes/Nikhil/Data/SWIFT/Flux/"+str(Line[q])+"_UVW2_flx.fits")
    hdu_Flux = fits.open(get_pkg_data_filename("/Volumes/Nikhil/Data/SWIFT/Flux/"+str(Line[q])+"_UVW2_flx.fits"))[0]
    convfact_W2 = hdu[0].header['FLMBDA']
    abz_W2 = hdu[0].header['ABMAGZP']
    skycnts_W2 = hdu[0].header['SKYC']
    eskycnts_W2 = hdu[0].header['ESKYC']
    wcs = WCS(hdu_Flux.header)
    P1,P2 = wcs.wcs_world2pix((ifura),(ifudec),0)
    cutout = Cutout2D(hdu_Flux.data, position=(P1,P2), size=S_cutout, wcs=wcs)
    Flux_W2 = Cutout2D(hdu[0].data, position=(P1,P2), size=S_cutout, wcs=wcs)
    Flux_W2_sigma = Cutout2D(hdu[1].data, position=(P1,P2), size=S_cutout, wcs=wcs)
    Cnts_W2 = Cutout2D(hdu[2].data, position=(P1,P2), size=S_cutout, wcs=wcs)
    Ecnts_W2 = Cutout2D(hdu[3].data, position=(P1,P2), size=S_cutout, wcs=wcs)
    Exp_W2 = Cutout2D(hdu[4].data, position=(P1,P2), size=S_cutout, wcs=wcs)
    Mask_W2 = Cutout2D(hdu[5].data, position=(P1,P2), size=S_cutout, wcs=wcs)
    hdu_Flux.data = cutout.data
    hdu_Flux.header.update(cutout.wcs.to_header())
    wcs_N = WCS(hdu_Flux.header)
    wcs_mnga = WCS(H_manga)
    
    #CONVOLUTION START for Dn4000
    CONV_D4R = signal.convolve2d(NNUM*4,Ga,boundary='symm',mode='same')
    CONV_D4B = signal.convolve2d(NDENO*4, Ga,boundary='symm',mode='same')
    CONV_D4_MASK = signal.convolve2d(SP_MASK[44],Ga,boundary='symm',mode='same')
    ND4_R = reproject_exact((CONV_D4R,H_manga), hdu_Flux.header)[0]
    ND4_B = reproject_exact((CONV_D4B,H_manga), hdu_Flux.header)[0]
    ND4_MASK = reproject_exact((CONV_D4_MASK,H_manga), hdu_Flux.header)[0]

    #Generating derivative matrix Z for reproject
    New_ifu = np.shape(ND4_R)[0]
    New_pix = New_ifu**2
    COV_RP = np.ones((3,3))
    m=New_ifu
    P1_conv = np.shape(COV_RP)[0]
    Z_CONV = np.zeros((New_pix,Npix))
    # Build 2D coordinates for the all pixels in the Swift array
    x1 = np.arange(New_ifu)
    y1 = np.arange(New_ifu)
    xs,ys = np.meshgrid(x1,y1)
    # Find their corresponding pixel centers in the MaNGA array
    ra,dec = wcs_N.wcs_pix2world(xs,ys,0)
    xm,ym = wcs_mnga.wcs_world2pix(ra,dec,0) # Here the mode has to be zero because we are going to use xm and ym in python arrays which are always 0-indexed.
    xmi = ((xm+0.5).astype('int')).flatten()
    xmp = xm.flatten() - (xmi-0.5)
    ymi = ((ym+0.5).astype('int')).flatten()
    ymp = ym.flatten() - (ymi-0.5)
    mcenters = ymi*Nifu+xmi
    dy_swift = wcs_N.wcs_pix2world(0,1,0)[1]-wcs_N.wcs_pix2world(0,0,0)[1]
    dy_mnga = wcs_mnga.wcs_pix2world(0,1,0)[1]-wcs_mnga.wcs_pix2world(0,0,0)[1]
    pixratio = dy_swift/dy_mnga
    q_1=0
    for q_1 in range (0,New_pix):
        if (xmi[q_1]>=0 and xmi[q_1]<Nifu and ymi[q_1]>=0 and ymi[q_1]<Nifu):
            z_1=0
            for z_1 in range (0,P1_conv):
                xfrac = np.array([pixratio/2-xmp[q_1],1.0,xmp[q_1]+pixratio/2-1]).reshape(1,3)
                yfrac = np.array([pixratio/2-ymp[q_1],1.0,ymp[q_1]+pixratio/2-1]).reshape(3,1)
                COV_RP = np.matmul(yfrac,xfrac)
                COV_RP = COV_RP/np.sum(COV_RP)
                i0 = int((P1_conv-1)/2)
                start = mcenters[q_1]-(i0-z_1)*Nifu-i0
                finish = mcenters[q_1]-(i0-z_1)*Nifu+i0+1
                start2 = np.clip(start,0,Npix-1)
                finish2 = np.clip(finish,0,Npix-1)
                Z_CONV[q_1,start2:finish2] = COV_RP[z_1,start2-start:P1_conv+finish2-finish]

    
    Ref_pix = wcs_N.wcs_pix2world(0,0,0)
    
    CONV_EFLUX = np.zeros(np.shape(GFLUX_Clip))
    CONV_ECONT = np.zeros(np.shape(GFLUX_Clip))
    CONV_MASK_EF = np.zeros(np.shape(GFLUX_Clip))
    CONV_MASK_EE = np.zeros(np.shape(GFLUX_Clip))
    NEW_EFLUX = np.zeros((np.shape(GFLUX_Clip)[0],np.shape(cutout)[0],np.shape(cutout)[0]))
    NEW_ECONT = np.zeros((np.shape(GFLUX_Clip)[0],np.shape(cutout)[0],np.shape(cutout)[0]))
    NEW_MASK_EF = np.zeros((np.shape(GFLUX_Clip)[0],np.shape(cutout)[0],np.shape(cutout)[0]))
    NEW_MASK_EE = np.zeros((np.shape(GFLUX_Clip)[0],np.shape(cutout)[0],np.shape(cutout)[0]))
    p=0
    for p in range (0,np.shape(GFLUX_Clip)[0]):
        CONV_EFLUX[p,:,:] = signal.convolve2d(GFLUX_Clip[p,:,:]*4,Ga,boundary='symm',mode='same')
        CONV_ECONT[p,:,:] = signal.convolve2d(GCONT_Clip[p,:,:]*4,Ga,boundary='symm',mode='same')
        CONV_MASK_EF[p,:,:] = signal.convolve2d(GFMASK[p,:,:],Ga,boundary='symm',mode='same')
        CONV_MASK_EE[p,:,:] = signal.convolve2d(GEWMASK[p,:,:],Ga,boundary='symm',mode='same')
        NEW_EFLUX[p,:,:] = reproject_exact((CONV_EFLUX[p,:,:],H_manga), hdu_Flux.header)[0]
        NEW_ECONT[p,:,:] = reproject_exact((CONV_ECONT[p,:,:],H_manga), hdu_Flux.header)[0]
        NEW_MASK_EF[p,:,:] = reproject_exact((CONV_MASK_EF[p,:,:],H_manga), hdu_Flux.header)[0]
        NEW_MASK_EE[p,:,:] = reproject_exact((CONV_MASK_EE[p,:,:],H_manga), hdu_Flux.header)[0]
        
    CONV_IFLUX = np.zeros(np.shape(SPEC_Index_Flux))
    CONV_ICONT = np.zeros(np.shape(SPEC_Index_Flux))
    CONV_MASK_SI = np.zeros(np.shape(SPEC_Index_Flux))
    CONV_COMB_AVG = np.zeros(np.shape(SPEC_Index_Flux))
    CONV_COMB_AVG_SIGMA = np.zeros(np.shape(SPEC_Index_Flux))
    CONV_COMB_AVG_MASK = np.zeros(np.shape(SPEC_Index_Flux))
    
    
    
    NEW_IFLUX = np.zeros((np.shape(SPEC_Index_Flux)[0],np.shape(cutout)[0],np.shape(cutout)[0]))
    NEW_ICONT = np.zeros((np.shape(SPEC_Index_Flux)[0],np.shape(cutout)[0],np.shape(cutout)[0]))
    NEW_MASK_SI = np.zeros((np.shape(SPEC_Index_Flux)[0],np.shape(cutout)[0],np.shape(cutout)[0]))
    NEW_COMB_AVG = np.zeros((np.shape(SPEC_Index_Flux)[0],np.shape(cutout)[0],np.shape(cutout)[0]))
    NEW_COMB_AVG_SIGMA = np.zeros((np.shape(SPEC_Index_Flux)[0],np.shape(cutout)[0],np.shape(cutout)[0]))
    NEW_COMB_AVG_MASK = np.zeros((np.shape(SPEC_Index_Flux)[0],np.shape(cutout)[0],np.shape(cutout)[0]))
    p=0
    for p in range (0,np.shape(SPEC_Index_Flux)[0]):
        CONV_IFLUX[p,:,:] = signal.convolve2d(SPEC_Index_Flux[p,:,:]*4,Ga,boundary='symm',mode='same')
        CONV_ICONT[p,:,:] = signal.convolve2d(SPEC_Index_Cont[p,:,:]*4,Ga,boundary='symm',mode='same')
        CONV_MASK_SI[p,:,:] = signal.convolve2d(SPEC_Index_Mask[p,:,:],Ga,boundary='symm',mode='same')
        CONV_COMB_AVG[p,:,:] = signal.convolve2d(COMB_AVG[p,:,:]*4,Ga,boundary='symm',mode='same')
        CONV_COMB_AVG_SIGMA[p,:,:] = signal.convolve2d(COMB_AVG_SIGMA[p,:,:],Ga,boundary='symm',mode='same')
        CONV_COMB_AVG_MASK[p,:,:] = signal.convolve2d(COMB_AVG_MASK[p,:,:],Ga,boundary='symm',mode='same')
        NEW_IFLUX[p,:,:] = reproject_exact((CONV_IFLUX[p,:,:],H_manga), hdu_Flux.header)[0]
        NEW_ICONT[p,:,:] = reproject_exact((CONV_ICONT[p,:,:],H_manga), hdu_Flux.header)[0]
        NEW_MASK_SI[p,:,:] = reproject_exact((CONV_MASK_SI[p,:,:],H_manga), hdu_Flux.header)[0]
        NEW_COMB_AVG[p,:,:] = reproject_exact((CONV_COMB_AVG[p,:,:],H_manga), hdu_Flux.header)[0]
        NEW_COMB_AVG_SIGMA[p,:,:] = reproject_exact((CONV_COMB_AVG_SIGMA[p,:,:],H_manga), hdu_Flux.header)[0]
        NEW_COMB_AVG_MASK[p,:,:] = reproject_exact((CONV_COMB_AVG_MASK[p,:,:],H_manga), hdu_Flux.header)[0]
        



    #CALCULATE NEW VARIANCES
    ZW_matrix = np.matmul(Z_CONV,W_CONV)
    NCOV_EF = np.zeros((np.shape(COV_EF)[0],New_pix,New_pix))
    NCOV_EL = np.zeros((np.shape(COV_EF)[0],New_pix,New_pix))
    NCOV_IF =  np.zeros((np.shape(COV_IF)[0],New_pix,New_pix))
    NCOV_IC =  np.zeros((np.shape(COV_IF)[0],New_pix,New_pix))
    NCOV_D4 = np.zeros((2,New_pix,New_pix))
    NVAR_EF = np.zeros((np.shape(COV_EF)[0],New_ifu,New_ifu))
    NVAR_IF = np.zeros((np.shape(COV_IF)[0],New_ifu,New_ifu))
    NVAR_IC = np.zeros((np.shape(COV_IF)[0],New_ifu,New_ifu))
    NVAR_D4 = np.zeros((2,New_ifu,New_ifu))

    em=0
    for em in range (0,np.shape(COV_EF)[0]):
        NCOV_EF[em] = np.matmul(np.matmul(ZW_matrix,COV_EF[em]),np.transpose(ZW_matrix))
        NVAR_EF[em] = np.reshape(np.diagonal(NCOV_EF[em]),(New_ifu,New_ifu))
        
    sp=0
    for sp in range (0,np.shape(COV_IF)[0]):
        NCOV_IF[sp] = np.matmul(np.matmul(ZW_matrix,COV_IF[sp]),np.transpose(ZW_matrix))
        NCOV_IC[sp] = np.matmul(np.matmul(ZW_matrix,COV_IC[sp]),np.transpose(ZW_matrix))
        NVAR_IF[sp] = np.reshape(np.diagonal(NCOV_IF[sp]),(New_ifu,New_ifu))
        NVAR_IC[sp] = np.reshape(np.diagonal(NCOV_IC[sp]),(New_ifu,New_ifu))
    d=0
    for d in range (0,np.shape(COV_D4)[0]):
        NCOV_D4[d] = np.matmul(np.matmul(ZW_matrix,COV_D4[d]),np.transpose(ZW_matrix))
        NVAR_D4[d] = np.reshape(np.diagonal(NCOV_D4[d]),(New_ifu,New_ifu))
    

    NEW_IFLUX_SIG = np.sqrt(NVAR_IF)
    NEW_ICONT_SIG = np.sqrt(NVAR_IC)
    NEW_EFLUX_SIG = np.sqrt(NVAR_EF)
    
    
    
    #CONVERT ALL FRACTIONAL MASK VALUES TO INTEGER
    NEW_MASK_EF[np.isnan(NEW_MASK_EF)] = 1
    NEW_MASK_EE[np.isnan(NEW_MASK_EE)] = 1
    NEW_MASK_SI[np.isnan(NEW_MASK_SI)] = 1
    ND4_MASK[np.isnan(ND4_MASK)] = 1
    NEW_COMB_AVG_MASK[np.isnan(NEW_COMB_AVG_MASK)] = 1
    
    EFLUX_I1 = np.where(NEW_MASK_EF>0.4)
    ELINE_I1 = np.where(NEW_MASK_EE>0.4)
    SI_I1 = np.where(NEW_MASK_SI>0.4)
    D4_I1 = np.where(ND4_MASK>0.4)
    DISP_I1 = np.where(NEW_COMB_AVG_MASK>0.4)
    
    NEW_MASK_EF[EFLUX_I1]=1
    NEW_MASK_EE[ELINE_I1]=1
    NEW_MASK_SI[SI_I1]=1
    ND4_MASK[D4_I1]=1
    NEW_COMB_AVG_MASK[DISP_I1]=1
    
    EFLUX_I2 = np.where(NEW_MASK_EF<0.4)
    ELINE_I2 = np.where(NEW_MASK_EE<0.4)
    SI_I2 = np.where(NEW_MASK_SI<0.4)
    D4_I2 = np.where(ND4_MASK<0.4)
    DISP_I2 = np.where(NEW_COMB_AVG_MASK<0.4)
    
    NEW_MASK_EF[EFLUX_I2]=0
    NEW_MASK_EE[ELINE_I2]=0
    NEW_MASK_SI[SI_I2]=0
    ND4_MASK[D4_I2]=0
    NEW_COMB_AVG_MASK[DISP_I2]=0

    
    hdu = fits.open("/Volumes/Nikhil/Data/SWIFT/Flux/"+str(Line[q])+"_UVM2_flx.fits")
    uvm2_h = hdu[0].header
    Flux_M2 = signal.convolve2d(hdu[0].data,Ga2,boundary='symm',mode='same')
    Flux_M2_sigma = signal.convolve2d(hdu[1].data**2,Ga2,boundary='symm',mode='same')
    Cnts_M2 = signal.convolve2d(hdu[2].data,Ga2,boundary='symm',mode='same')
    Ecnts_M2 = signal.convolve2d(hdu[3].data**2,Ga2,boundary='symm',mode='same')
    Exp_M2 = signal.convolve2d(hdu[4].data,Ga2,boundary='symm',mode='same')
    Mask_M2 = signal.convolve2d(hdu[5].data,Ga2,boundary='symm',mode='same')
    convfact_M2 = hdu[0].header['FLMBDA']
    abz_M2 = hdu[0].header['ABMAGZP']
    skycnts_M2 = hdu[0].header['SKYC']
    eskycnts_M2 = hdu[0].header['ESKYC']

    
    FM2 = reproject_exact((Flux_M2,uvm2_h), hdu_Flux.header)[0]
    FEM2 = np.sqrt(reproject_exact((Flux_M2_sigma,uvm2_h), hdu_Flux.header)[0])
    CM2 = reproject_exact((Cnts_M2,uvm2_h), hdu_Flux.header)[0]
    EM2 = np.sqrt(reproject_exact((Ecnts_M2,uvm2_h), hdu_Flux.header)[0])
    EXM2 = reproject_exact((Exp_M2,uvm2_h), hdu_Flux.header)[0]
    MKM2 = reproject_exact((Mask_M2,uvm2_h),hdu_Flux.header)[0]
    MKM2[np.isnan(MKM2)] = 1
    IKM2 = np.where(MKM2>0.4)
    MKM2[IKM2]=1
    #Generate W derivative matrix for uvm2
    hdu_flx_m2 = fits.open(get_pkg_data_filename("/Volumes/Nikhil/Data/SWIFT/Flux/"+str(Line[q])+"_UVM2_flx.fits"))[0]
    hdu_err_m2 = fits.open(get_pkg_data_filename("/Volumes/Nikhil/Data/SWIFT/Flux/"+str(Line[q])+"_UVM2_flx.fits"))[1]
    wcs_m2 = WCS(hdu_err_m2.header)
    P1_m2,P2_m2 = wcs_m2.wcs_world2pix((ifura),(ifudec),0)
    cut_m2 = Cutout2D(hdu_err_m2.data, position=(P1_m2,P2_m2), size=S_cutout, wcs=wcs_m2)
    hdu_err_m2.data = cut_m2.data
    hdu_err_m2.header.update(cut_m2.wcs.to_header())
    wcs_m2 = WCS(hdu_err_m2.header)
    VAR_m2 = np.reshape((hdu_err_m2.data)**2,(New_pix))
    COV_m2 = np.zeros((New_pix,New_pix))
    COV_m2[np.diag_indices(New_pix)] = VAR_m2
    m_m2=New_ifu
    P_conv_m2 = np.shape(Ga2)[0]
    W_CONV_m2 = np.zeros((New_pix,New_pix))
    q0m2=0
    for q0m2 in range (0,New_pix):
        z0m2=0
        for z0m2 in range (0,P_conv_m2):
            i0m2 = int((P_conv_m2-1)/2)
            start_m2 = q0m2-(i0m2-z0m2)*m_m2-i0m2
            finish_m2 = q0m2-(i0m2-z0m2)*m_m2+i0m2+1
            start2_m2 = np.min([np.max([start_m2,0]),New_pix])
            finish2_m2 = np.min([np.max([finish_m2,0]),New_pix])
            W_CONV_m2[q0m2,start2_m2:finish2_m2] = Ga2[z0m2,start2_m2-start_m2:P_conv_m2+finish2_m2-finish_m2]
    #Generate Z matrix for uvm2
    COV_RP_m2 = np.ones((2,2))
    P1_conv_m2 = np.shape(COV_RP_m2)[0]
    Z_CONV_m2 = np.zeros((New_pix,New_pix))
    # Find their corresponding pixel centers in the uvw2 array
    ra,dec = wcs_N.wcs_pix2world(xs,ys,0)
    xm_m2,ym_m2 = wcs_m2.wcs_world2pix(ra,dec,0) # Here the mode has to be zero because we are going to use xm and ym in python arrays which are always 0-indexed.
    xmi_m2 = (xm_m2.astype('int')).flatten()
    xmp_m2 = xm_m2.flatten() - xmi_m2
    ymi_m2 = (ym_m2.astype('int')).flatten()
    ymp_m2 = ym_m2.flatten() - ymi_m2
    mcenters_m2 = ymi_m2*New_ifu+xmi_m2
    q_1_m2=0
    for q_1_m2 in range (0,New_pix):
        if (xmi_m2[q_1_m2]>=0 and xmi_m2[q_1_m2]<New_ifu and ymi_m2[q_1_m2]>=0 and ymi_m2[q_1_m2]<New_ifu):
            z_1_m2=0
            for z_1_m2 in range (0,P1_conv_m2):
                xfrac_m2 = np.array([1-xmp_m2[q_1_m2],xmp_m2[q_1_m2]]).reshape(1,2)
                yfrac_m2 = np.array([1-ymp_m2[q_1_m2],ymp_m2[q_1_m2]]).reshape(2,1)
                COV_RP_m2 = np.matmul(yfrac_m2,xfrac_m2)
                COV_RP_m2 = COV_RP_m2/np.sum(COV_RP_m2)
                i0_m2 = int((P1_conv_m2-1)/2)
                start_m2 = mcenters_m2[q_1_m2]-(i0_m2-z_1_m2)*New_ifu-i0_m2
                finish_m2 = mcenters_m2[q_1_m2]-(i0_m2-z_1_m2)*New_ifu+i0_m2+1
                start2_m2 = np.clip(start_m2,0,New_pix-1)
                finish2_m2 = np.clip(finish_m2,0,New_pix-1)
                Z_CONV_m2[q_1_m2,start2_m2:finish2_m2] = COV_RP_m2[z_1_m2,start2_m2-start_m2:P1_conv_m2+finish2_m2-finish_m2]
    ZW_m2 = np.matmul(Z_CONV_m2,W_CONV_m2)
    New_var_m2 = np.reshape(np.diagonal(np.matmul(np.matmul(ZW_m2,COV_m2),np.transpose(ZW_m2))),(New_ifu,New_ifu))
  
    hdu = fits.open("/Volumes/Nikhil/Data/SWIFT/Flux/"+str(Line[q])+"_UVW1_flx.fits")
    uvw1_h = hdu[0].header
    Flux_W1 = signal.convolve2d(hdu[0].data,Ga1,boundary='symm',mode='same')
    Flux_W1_sigma = signal.convolve2d(hdu[1].data**2,Ga1,boundary='symm',mode='same')
    Cnts_W1 = signal.convolve2d(hdu[2].data,Ga1,boundary='symm',mode='same')
    Ecnts_W1 = signal.convolve2d(hdu[3].data**2,Ga1,boundary='symm',mode='same')
    Exp_W1 = signal.convolve2d(hdu[4].data,Ga1,boundary='symm',mode='same')
    Mask_W1 = signal.convolve2d(hdu[5].data,Ga1,boundary='symm',mode='same')
    convfact_W1 = hdu[0].header['FLMBDA']
    abz_W1 = hdu[0].header['ABMAGZP']
    skycnts_W1 = hdu[0].header['SKYC']
    eskycnts_W1 = hdu[0].header['ESKYC']
    FW1 = reproject_exact((Flux_W1,uvw1_h), hdu_Flux.header)[0]
    FEW1 = np.sqrt(reproject_exact((Flux_W1_sigma,uvw1_h), hdu_Flux.header)[0])
    CW1 = reproject_exact((Cnts_W1,uvw1_h), hdu_Flux.header)[0]
    EW1 = np.sqrt(reproject_exact((Ecnts_W1,uvw1_h), hdu_Flux.header)[0])
    EXW1 = reproject_exact((Exp_W1,uvw1_h), hdu_Flux.header)[0]
    MKW1 = reproject_exact((Mask_W1,uvw1_h),hdu_Flux.header)[0]
    MKW1[np.isnan(MKW1)] = 1
    IKW1 = np.where(MKW1>0.4)
    MKW1[IKW1]=1
    
    FW2 = Flux_W2.data
    FEW2 = Flux_W2_sigma.data
    CW2 = Cnts_W2.data
    EW2 = Ecnts_W2.data
    EXW2 = Exp_W2.data
    MKW2 = Mask_W2.data
    MKW2[np.isnan(MKW2)] = 1
    IKW2 = np.where(MKW2>0.4)
    MKW2[IKW2]=1
    
    
    #Generate W derivative matrix for uvw1
    hdu_flx_w1 = fits.open(get_pkg_data_filename("/Volumes/Nikhil/Data/SWIFT/Flux/"+str(Line[q])+"_UVW1_flx.fits"))[0]
    hdu_err_w1 = fits.open(get_pkg_data_filename("/Volumes/Nikhil/Data/SWIFT/Flux/"+str(Line[q])+"_UVW1_flx.fits"))[1]
    wcs_w1 = WCS(hdu_err_w1.header)
    P1_w1,P2_w1 = wcs_w1.wcs_world2pix((ifura),(ifudec),0)
    cut_w1 = Cutout2D(hdu_err_w1.data, position=(P1_w1,P2_w1), size=S_cutout, wcs=wcs_w1)
    hdu_err_w1.data = cut_w1.data
    hdu_err_w1.header.update(cut_w1.wcs.to_header())
    wcs_w1 = WCS(hdu_err_w1.header)
    VAR_w1 = np.reshape((hdu_err_w1.data)**2,(New_pix))
    COV_w1 = np.zeros((New_pix,New_pix))
    COV_w1[np.diag_indices(New_pix)] = VAR_w1
    m_w1=New_ifu
    P_conv_w1 = np.shape(Ga1)[0]
    W_CONV_w1 = np.zeros((New_pix,New_pix))
    q0w1=0
    for q0w1 in range (0,New_pix):
        z0w1=0
        for z0w1 in range (0,P_conv_w1):
            i0w1 = int((P_conv_w1-1)/2)
            start_w1 = q0w1-(i0w1-z0w1)*m_w1-i0w1
            finish_w1 = q0w1-(i0w1-z0w1)*m_w1+i0w1+1
            start2_w1 = np.min([np.max([start_w1,0]),New_pix])
            finish2_w1 = np.min([np.max([finish_w1,0]),New_pix])
            W_CONV_w1[q0w1,start2_w1:finish2_w1] = Ga1[z0w1,start2_w1-start_w1:P_conv_w1+finish2_w1-finish_w1]
    #Generate Z matrix for uvw1
    COV_RP_w1 = np.ones((2,2))
    P1_conv_w1 = np.shape(COV_RP_w1)[0]
    Z_CONV_w1 = np.zeros((New_pix,New_pix))
    # Find their corresponding pixel centers in the uvw2 array
    ra,dec = wcs_N.wcs_pix2world(xs,ys,0)
    xm_w1,ym_w1 = wcs_w1.wcs_world2pix(ra,dec,0) # Here the mode has to be zero because we are going to use xm and ym in python arrays which are always 0-indexed.
    xmi_w1 = (xm_w1.astype('int')).flatten()
    xmp_w1 = xm_w1.flatten() - xmi_w1
    ymi_w1 = (ym_w1.astype('int')).flatten()
    ymp_w1 = ym_w1.flatten() - ymi_w1
    mcenters_w1 = ymi_w1*New_ifu+xmi_w1
    q_1_w1=0
    for q_1_w1 in range (0,New_pix):
        if (xmi_w1[q_1_w1]>=0 and xmi_w1[q_1_w1]<New_ifu and ymi_w1[q_1_w1]>=0 and ymi_w1[q_1_w1]<New_ifu):
            z_1_w1=0
            for z_1_w1 in range (0,P1_conv_w1):
                xfrac_w1 = np.array([1-xmp_w1[q_1_w1],xmp_w1[q_1_w1]]).reshape(1,2)
                yfrac_w1 = np.array([1-ymp_w1[q_1_w1],ymp_w1[q_1_w1]]).reshape(2,1)
                COV_RP_w1 = np.matmul(yfrac_w1,xfrac_w1)
                COV_RP_w1 = COV_RP_w1/np.sum(COV_RP_w1)
                i0_w1 = int((P1_conv_w1-1)/2)
                start_w1 = mcenters_w1[q_1_w1]-(i0_w1-z_1_w1)*New_ifu-i0_w1
                finish_w1 = mcenters_w1[q_1_w1]-(i0_w1-z_1_w1)*New_ifu+i0_w1+1
                start2_w1 = np.clip(start_w1,0,New_pix-1)
                finish2_w1 = np.clip(finish_w1,0,New_pix-1)
                Z_CONV_w1[q_1_w1,start2_w1:finish2_w1] = COV_RP_w1[z_1_w1,start2_w1-start_w1:P1_conv_w1+finish2_w1-finish_w1]
    ZW_w1 = np.matmul(Z_CONV_w1,W_CONV_w1)
    New_var_w1 = np.reshape(np.diagonal(np.matmul(np.matmul(ZW_w1,COV_w1),np.transpose(ZW_w1))),(New_ifu,New_ifu))
    
    #Convolution kernel for sloan
    SigmaSloan = (1.4/2.355)
    lsln = math.ceil(4*SigmaSloan)
    dsln = 0.396
    sln = np.sqrt((SwftSigmaw2-0.0376)**2 - SigmaSloan**2)
    x_sln1 = np.arange(0,lsln+1,0.396)
    x_sln = np.zeros(np.shape(x_sln1)[0]+np.shape(x_sln1)[0]-1)
    x_sln[10:21] = x_sln1
    x_sln[0:10] = np.flip(x_sln1[1:21])*(-1)
    XS,YS = np.meshgrid(x_sln,x_sln)
    KS = np.exp(-(XS ** 2 + YS ** 2) / (2 * sln ** 2))
    G = KS/np.sum(KS)
    
    #COVARIANCE FOR SDSS
    hdu = fits.open("/Volumes/Nikhil/Data/SDSS/" + str(sloan) + "-u.fits")
    wcs_sloan = WCS(hdu[0].header)
    xps,yps = wcs_sloan.wcs_world2pix((ifura),(ifudec),0)
    sheader = getheader("/Volumes/Nikhil/Data/SDSS/" + str(sloan) + "-u.fits",0)
    cut_sdss = math.ceil((Nifu/2)/0.396+(4/0.396))
    fluxcutout = Cutout2D(hdu[0].data/(0.396**2), position=(xps,yps), size=(cut_sdss,cut_sdss), wcs=wcs_sloan)
    sheader.update(fluxcutout.wcs.to_header())
    #Calculating W (the derivative matric for psf convolution) for SDSS
    m_sdss=cut_sdss
    P_conv_sdss = np.shape(G)[0]
    W_CONV_sdss = np.zeros((cut_sdss**2,cut_sdss**2))
    q0_sdss=0
    for q0_sdss in range (0,cut_sdss**2):
         z0_sdss=0
         for z0_sdss in range (0,P_conv_sdss):
             i0_sdss = int((P_conv_sdss-1)/2)
             start_sdss = q0_sdss-(i0_sdss-z0_sdss)*m_sdss-i0_sdss
             finish_sdss = q0_sdss-(i0_sdss-z0_sdss)*m_sdss+i0_sdss+1
             start2_sdss = np.min([np.max([start_sdss,0]),cut_sdss**2])
             finish2_sdss = np.min([np.max([finish_sdss,0]),cut_sdss**2])
             W_CONV_sdss[q0_sdss,start2_sdss:finish2_sdss] = G[z0_sdss,start2_sdss-start_sdss:P_conv_sdss+finish2_sdss-finish_sdss]
   
    #Generating derivative matrix Z for reproject SDSS
   
    COV_RP_sdss = np.ones((5,5))
    P1_conv_sdss = np.shape(COV_RP_sdss)[0]
    Z_CONV_sdss = np.zeros((New_pix,cut_sdss**2))
    # Build 2D coordinates for the all pixels in the Swift array
    x1 = np.arange(New_ifu)
    y1 = np.arange(New_ifu)
    xs,ys = np.meshgrid(x1,y1)
    # Find their corresponding pixel centers in the MaNGA array
    ra,dec = wcs_N.wcs_pix2world(xs,ys,0)
    xm_sdss,ym_sdss = fluxcutout.wcs.wcs_world2pix(ra,dec,0) # Here the mode has to be zero because we are going to use xm and ym in python arrays which are always 0-indexed.
    xmi_sdss = ((xm_sdss+0.5).astype('int')).flatten()
    xmp_sdss = xm_sdss.flatten() - (xmi_sdss-0.5)
    ymi_sdss = ((ym_sdss+0.5).astype('int')).flatten()
    ymp_sdss = ym_sdss.flatten() - (ymi_sdss-0.5)
    mcenters_sdss = ymi_sdss*cut_sdss+xmi_sdss
    dy_swift = wcs_N.wcs_pix2world(0,1,0)[1]-wcs_N.wcs_pix2world(0,0,0)[1]
    dy_sdss = fluxcutout.wcs.wcs_pix2world(0,1,0)[1]-fluxcutout.wcs.wcs_pix2world(0,0,0)[1]
    pixratio_sdss = dy_swift/dy_sdss
    for q_1_sdss in range (0,New_pix):
      if (xmi_sdss[q_1_sdss]>=0 and xmi_sdss[q_1_sdss]<cut_sdss and ymi_sdss[q_1_sdss]>=0 and ymi_sdss[q_1_sdss]<cut_sdss):
        leftbound= -pixratio_sdss/2.+xmp_sdss[q_1_sdss]-0.5
        rightbound = pixratio_sdss/2.+xmp_sdss[q_1_sdss]-0.5
        xleft = np.arange(5)-2.5
        xright= np.arange(5)-1.5
        xleft[xleft<leftbound] = leftbound # selecting the greater one between leftbound and each element of xleft
        xright[xright>rightbound] = rightbound #selecting the smaller one between rightbound and each element of xright
        xfrac_sdss = np.clip(xright-xleft,0,1).reshape(1,5)
        lowerbound = -pixratio_sdss/2.+ymp_sdss[q_1_sdss]-0.5
        upperbound = pixratio_sdss/2.+ymp_sdss[q_1_sdss]-0.5
        ylow = np.arange(5)-2.5
        yhigh = np.arange(5)-1.5
        ylow[ylow<lowerbound] = lowerbound #selecting the greater one between lowerbound and each element of ylow
        yhigh[yhigh>upperbound] = upperbound #selecting the smaller one between upperbound and each element of yhigh
        yfrac_sdss = np.clip(yhigh-ylow,0,1).reshape(5,1)
        COV_RP_sdss = np.matmul(yfrac_sdss,xfrac_sdss)
        COV_RP_sdss = COV_RP_sdss/np.sum(COV_RP_sdss)
        i0_sdss = int((P1_conv_sdss-1)/2)
        for z_1_sdss in range (0,P1_conv_sdss):
            start_sdss = mcenters_sdss[q_1_sdss]-(i0_sdss-z_1_sdss)*cut_sdss-i0_sdss
            finish_sdss = mcenters_sdss[q_1_sdss]-(i0_sdss-z_1_sdss)*cut_sdss+i0_sdss+1
            start2_sdss = np.clip(start_sdss,0,(cut_sdss**2)-1)
            finish2_sdss = np.clip(finish_sdss,0,(cut_sdss**2)-1)
            Z_CONV_sdss[q_1_sdss,start2_sdss:finish2_sdss] = COV_RP_sdss[z_1_sdss,start2_sdss-start_sdss:P1_conv_sdss+finish2_sdss-finish_sdss]
    ZW_matrix_sdss = np.matmul(Z_CONV_sdss,W_CONV_sdss)

    sdsspixel = 0.396**2
    filters = np.array(['u','g','r','i','z'])
  
    newsdss = np.zeros((5,New_ifu,New_ifu),dtype='float')
    newsdss_sig = np.zeros((5,New_ifu,New_ifu),dtype='float')
    for sdss_j in range(5): 
        hdu = fits.open("/Volumes/Nikhil/Data/SDSS/" + str(sloan) + "-"+filters[sdss_j]+".fits")
        sheader = getheader("/Volumes/Nikhil/Data/SDSS/" + str(sloan) + "-"+filters[sdss_j]+".fits",0)
        wcs_sloan = WCS(hdu[0].header)
        xps,yps = wcs_sloan.wcs_world2pix((ifura),(ifudec),0)
        cutsize = math.ceil((Nifu/2)/0.396+(4/0.396))
        fluxcutout = Cutout2D(hdu[0].data, position=(xps,yps), size=(cutsize,cutsize), wcs=wcs_sloan)
        ivarcutout = Cutout2D(hdu[1].data,position=(xps,yps),size=(cutsize,cutsize),wcs=wcs_sloan)
        sheader.update(fluxcutout.wcs.to_header())
        fluxarr = fluxcutout.data
        ivararr = ivarcutout.data
        ivararr[np.isfinite(ivararr) == False] = 0
        p = np.where((fluxarr > np.percentile(fluxarr,99)) & (ivararr>0))
        coeff_rough=np.median(1/ivararr[p]/fluxarr[p])
        p = np.where((fluxarr > np.percentile(fluxarr,99)) & (ivararr>0) & (1/ivararr < fluxarr*coeff_rough*2))
        k = np.where((fluxarr<0) & (ivararr>0))
        tmp = np.median(fluxarr[k])
        k2 = np.where((fluxarr> tmp ) & (fluxarr < -tmp))
        coeff1 = np.median(1/ivararr[k2])
        coeff0 = np.median((1/ivararr[p]-coeff1)/fluxarr[p])
        coeff_set2=np.polyfit(fluxarr[p],1/ivararr[p],1)
        bounds = fluxarr*coeff0+coeff1
        bounds2 = fluxarr*coeff_set2[0]+coeff_set2[1]
        bad = np.where(((1/ivararr > bounds*1.5) & (1/ivararr> bounds2*1.5)) | (ivararr <=0))
        mask = np.zeros(np.shape(fluxarr),dtype='int')
        mask[bad] = 1
        fluxarr = fluxarr/sdsspixel
        vararr = 1/ivararr/sdsspixel**2
        vararr[np.isfinite(vararr)==False]=0
        conv_u = signal.convolve2d(fluxarr, G, boundary='symm', mode='same')
        conv_weight = signal.convolve2d((1-mask), G, boundary='symm',mode='same')
        RP_convw = reproject_exact((conv_weight,sheader), hdu_Flux.header)[0]
        RP_convw = np.reshape(RP_convw,(New_pix))
        RP_convw_J = np.repeat(RP_convw[:,np.newaxis],New_pix,axis=1)
        RP_convw_K = np.transpose(RP_convw_J)
        newsdss[sdss_j] = reproject_exact((conv_u,sheader), hdu_Flux.header)[0]
        covariance_sdss = np.zeros((cutsize**2,cutsize**2))
        covariance_sdss[np.diag_indices(cutsize**2)] = np.reshape(vararr*(1-mask),cutsize**2)
        new_covariance_sdss = (np.matmul(np.matmul(ZW_matrix_sdss,covariance_sdss),np.transpose(ZW_matrix_sdss)))/np.sqrt(RP_convw_J*RP_convw_K) #ZW(G)ZW^T
        newsdss_sig[sdss_j] = np.sqrt(np.reshape(np.diagonal(new_covariance_sdss),(New_ifu,New_ifu)))
    
    D4000_HDU = np.zeros((5,np.shape(ND4_R)[0],np.shape(ND4_R)[1]))
    D4000_HDU[0] = ND4_R*corr_factor
    D4000_HDU[1] = ND4_B*corr_factor
    D4000_HDU[2] = np.sqrt(NVAR_D4[0])*corr_factor
    D4000_HDU[3] = np.sqrt(NVAR_D4[1])*corr_factor
    D4000_HDU[4] = ND4_MASK
    D4000_HDU[np.isnan(D4000_HDU)] = 0
    D4000_HDU[D4000_HDU== inf] = 0
    D4000_HDU[D4000_HDU == -inf] = 0
    NEW_IFLUX[np.isnan(NEW_IFLUX)]=0
    NEW_IFLUX[NEW_IFLUX== inf] = 0
    NEW_IFLUX[NEW_IFLUX== -inf] = 0
    NEW_IFLUX_SIG[np.isnan(NEW_IFLUX_SIG)]=0
    NEW_IFLUX_SIG[NEW_IFLUX_SIG== inf] = 0
    NEW_IFLUX_SIG[NEW_IFLUX_SIG== -inf] = 0
    NEW_ICONT[np.isnan(NEW_ICONT)]=0
    NEW_ICONT[NEW_ICONT== inf] = 0
    NEW_ICONT[NEW_ICONT== -inf] = 0
    NEW_ICONT_SIG[np.isnan(NEW_ICONT_SIG)]=0
    NEW_ICONT_SIG[NEW_ICONT_SIG== inf] = 0
    NEW_ICONT_SIG[NEW_ICONT_SIG== -inf] = 0
    NEW_DISP = np.sqrt(NEW_COMB_AVG/(NEW_IFLUX))
    NEW_DISP[np.isnan(NEW_DISP)] = 0
    NEW_DISP[NEW_DISP== inf] = 0
    NEW_DISP[NEW_DISP == -inf] = 0
    NEW_DISP_SIGMA = np.sqrt(NEW_COMB_AVG_SIGMA)/NEW_IFLUX/NEW_DISP
    NEW_DISP_SIGMA[np.isnan(NEW_DISP_SIGMA)] = 0
    NEW_DISP_SIGMA[NEW_DISP_SIGMA== inf] = 0
    NEW_DISP_SIGMA[NEW_DISP_SIGMA == -inf] = 0
    NEW_EFLUX[np.isnan(NEW_EFLUX)]=0
    NEW_EFLUX[NEW_EFLUX== inf] = 0
    NEW_EFLUX[NEW_EFLUX== -inf] = 0
    NEW_EFLUX_SIG[np.isnan(NEW_EFLUX_SIG)]=0
    NEW_EFLUX_SIG[NEW_EFLUX_SIG== inf] = 0
    NEW_EFLUX_SIG[NEW_EFLUX_SIG== -inf] = 0
    NEW_EEW = NEW_EFLUX/NEW_ECONT
    NEW_EEW[np.isnan(NEW_EEW)]=0
    NEW_EEW[NEW_EEW== inf] = 0
    NEW_EEW[NEW_EEW == -inf] = 0
    NEW_EEW_SIG = NEW_EFLUX_SIG/NEW_ECONT
    NEW_EEW_SIG[np.isnan(NEW_EEW_SIG)]=0
    NEW_EEW_SIG[NEW_EEW_SIG == inf] = 0
    NEW_EEW_SIG[NEW_EEW_SIG == -inf] = 0
    PHOTO = np.zeros((8,np.shape(ND4_R)[0],np.shape(ND4_R)[1]))
    PHOTO_SIG = np.zeros((8,np.shape(ND4_R)[0],np.shape(ND4_R)[1]))
    PHOTO[0,:,:] = FW2
    PHOTO[1,:,:] = FW1
    PHOTO[2,:,:] = FM2
    PHOTO[3,:,:] = newsdss[0]
    PHOTO[4,:,:] = newsdss[1]
    PHOTO[5,:,:] = newsdss[2]
    PHOTO[6,:,:] = newsdss[3]
    PHOTO[7,:,:] = newsdss[4]
    PHOTO_SIG[0,:,:] = FEW2
    PHOTO_SIG[1,:,:] = np.sqrt(New_var_w1)
    PHOTO_SIG[2,:,:] = np.sqrt(New_var_m2)
    PHOTO_SIG[3,:,:] = newsdss_sig[0]
    PHOTO_SIG[4,:,:] = newsdss_sig[1]
    PHOTO_SIG[5,:,:] = newsdss_sig[2]
    PHOTO_SIG[6,:,:] = newsdss_sig[3]
    PHOTO_SIG[7,:,:] = newsdss_sig[4]
    PHOTO[np.isnan(PHOTO)]=0
    PHOTO[PHOTO== inf] = 0
    PHOTO[PHOTO == -inf] = 0
    PHOTO_SIG[np.isnan(PHOTO_SIG)]=0
    PHOTO_SIG[PHOTO_SIG== inf] = 0
    PHOTO_SIG[PHOTO_SIG == -inf] = 0
    UVOT = np.zeros((12,np.shape(ND4_R)[0],np.shape(ND4_R)[1]))
    UVOT[0,:,:] = CW2 + skycnts_W2
    UVOT[1,:,:] = CW1 + skycnts_W1
    UVOT[2,:,:] = CM2 + skycnts_M2
    UVOT[3,:,:] = EW2
    UVOT[4,:,:] = EW1
    UVOT[5,:,:] = EM2
    UVOT[6,:,:] = EXW2
    UVOT[7,:,:] = EXW1
    UVOT[8,:,:] = EXM2
    UVOT[9,:,:] = MKW2
    UVOT[10,:,:] = MKW1
    UVOT[11,:,:] = MKM2
    UVOT[np.isnan(UVOT)]=0
    UVOT[UVOT== inf] = 0
    UVOT[UVOT== -inf] = 0
    

    crval1 = np.asscalar(Ref_pix[0])
    crval2 = np.asscalar(Ref_pix[1])
    cpix1 = 1
    cpix2 = 1
    
    hdu0 = fits.PrimaryHDU(D4000_HDU)
    hdu1 = fits.ImageHDU(NEW_IFLUX*corr_factor)
    hdu2 = fits.ImageHDU(NEW_ICONT*corr_factor)
    hdu3 = fits.ImageHDU(NEW_IFLUX_SIG*corr_factor)
    hdu4 = fits.ImageHDU(NEW_ICONT_SIG*corr_factor)
    hdu5 = fits.ImageHDU(NEW_MASK_SI)
    hdu6 = fits.ImageHDU(NEW_DISP)
    hdu7 = fits.ImageHDU(NEW_DISP_SIGMA)
    hdu8 = fits.ImageHDU(NEW_COMB_AVG_MASK)
    hdu9 = fits.ImageHDU(NEW_EFLUX*corr_factor)
    hdu10 = fits.ImageHDU(NEW_EFLUX_SIG*corr_factor)
    hdu11 = fits.ImageHDU(NEW_MASK_EF)
    hdu12 = fits.ImageHDU(NEW_EEW)
    hdu13 = fits.ImageHDU(NEW_EEW_SIG)
    hdu14 = fits.ImageHDU(NEW_MASK_EE)
    hdu15 = fits.ImageHDU(PHOTO)
    hdu16 = fits.ImageHDU(PHOTO_SIG)
    hdu17 = fits.ImageHDU(UVOT)
    
    hdu5.scale('int32')
    hdu8.scale('int32')
    hdu11.scale('int32')
    hdu14.scale('int32')
    
    new_hdul = fits.HDUList([hdu0,hdu1, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7,hdu8,hdu9,hdu10,hdu11,hdu12,hdu13,hdu14,hdu15,hdu16,hdu17])
    new_hdul.writeto("/Volumes/Nikhil/Data/SWIFT/MPL-7_RP/SwiM_"+str(Line[q])+".fits")

    hdu = fits.open("/Volumes/Nikhil/Data/SWIFT/MPL-7_RP/SwiM_"+str(Line[q])+".fits")
    
    hdr0 = hdu[0].header
    hdr0['PLATE'] = plate
    hdr0['IFUDSGN'] = np.int(ifu)
    hdr0['OBJRA'] = objectra
    hdr0['OBJDEC'] = objectdec
    hdr0['IFURA'] = ifura
    hdr0['IFUDEC'] = ifudec
    hdr0['EXTNAME'] = 'D4000'
    hdr0['CTYPE1'] = 'RA---TAN'
    hdr0['CTYPE2'] = 'DEC--TAN'
    hdr0['CDELT1'] = wcs_N.wcs.cdelt[0]
    hdr0['CDELT2'] = wcs_N.wcs.cdelt[1]
    hdr0['CRPIX1'] = cpix1
    hdr0['CRPIX2'] = cpix2
    hdr0['CUNIT1'] = 'deg'
    hdr0['CUNIT2'] = 'deg'
    hdr0['RADESYS'] = 'FK5'
    hdr0['CRVAL1'] = crval1
    hdr0['CRVAL2'] = crval2
    hdr0['UNIT'] = 'erg/s/cm^2/Hz/arcsec^2'
    hdr0['C0'] = 'Fnu Red'
    hdr0['C1'] = 'Fnu Blue'
    hdr0['C2'] = 'Sigma Red'
    hdr0['C3'] = 'Sigma Blue'
    hdr0['C4'] = 'Mask'

    
    
    hdr1 = hdu[1].header
    hdr1['PLATE'] = plate
    hdr1['IFUDSGN'] = np.int(ifu)
    hdr1['EXTNAME'] = 'SPECINDX_FLUX'
    hdr1['CTYPE1'] = 'RA---TAN'
    hdr1['CTYPE2'] = 'DEC--TAN'
    hdr1['CDELT1'] = wcs_N.wcs.cdelt[0]
    hdr1['CDELT2'] = wcs_N.wcs.cdelt[1]
    hdr1['CRPIX1'] = cpix1
    hdr1['CRPIX2'] = cpix2
    hdr1['CUNIT1'] = 'deg'
    hdr1['CUNIT2'] = 'deg'
    hdr1['RADESYS'] = 'FK5'
    hdr1['CRVAL1'] = crval1
    hdr1['CRVAL2'] = crval2
    hdr1['UNIT'] = 'erg/s/cm^2/arcsec^2'
    hdr1['C0'] = Indices[0]
    hdr1['C1'] = Indices[1]
    hdr1['C2'] = Indices[2]
    hdr1['C3'] = Indices[3]
    hdr1['C4'] = Indices[4]
    hdr1['C5'] = Indices[5]
    hdr1['C6'] = Indices[6]
    hdr1['C7'] = Indices[7]
    hdr1['C8'] = Indices[8]
    hdr1['C9'] = Indices[9]
    hdr1['C10'] = Indices[10]
    hdr1['C11'] = Indices[11]
    hdr1['C12'] = Indices[12]
    hdr1['C13'] = Indices[13]
    hdr1['C14'] = Indices[14]
    hdr1['C15'] = Indices[15]
    hdr1['C16'] = Indices[16]
    hdr1['C17'] = Indices[17]
    hdr1['C18'] = Indices[18]
    hdr1['C19'] = Indices[19]
    hdr1['C20'] = Indices[20]
    hdr1['C21'] = Indices[21]
    hdr1['C22'] = Indices[22]
    hdr1['C23'] = Indices[23]
    hdr1['C24'] = Indices[24]
    hdr1['C25'] = Indices[25]
    hdr1['C26'] = Indices[26]
    hdr1['C27'] = Indices[27]
    hdr1['C28'] = Indices[28]
    hdr1['C29'] = Indices[29]
    hdr1['C30'] = Indices[30]
    hdr1['C31'] = Indices[31]
    hdr1['C32'] = Indices[32]
    hdr1['C33'] = Indices[33]
    hdr1['C34'] = Indices[34]
    hdr1['C35'] = Indices[35]
    hdr1['C36'] = Indices[36]
    hdr1['C37'] = Indices[37]
    hdr1['C38'] = Indices[38]
    hdr1['C39'] = Indices[39]
    hdr1['C40'] = Indices[40]
    hdr1['C41'] = Indices[41]
    hdr1['C42'] = Indices[42]

    
    hdr2 = hdu[2].header
    hdr2['PLATE'] = plate
    hdr2['IFUDSGN'] = np.int(ifu)
    hdr2['EXTNAME'] = 'SPECINDX_CONT'
    hdr2['CTYPE1'] = 'RA---TAN'
    hdr2['CTYPE2'] = 'DEC--TAN'
    hdr2['CDELT1'] = wcs_N.wcs.cdelt[0]
    hdr2['CDELT2'] = wcs_N.wcs.cdelt[1]
    hdr2['CRPIX1'] = cpix1
    hdr2['CRPIX2'] = cpix2
    hdr2['CUNIT1'] = 'deg'
    hdr2['CUNIT2'] = 'deg'
    hdr2['RADESYS'] = 'FK5'
    hdr2['CRVAL1'] = crval1
    hdr2['CRVAL2'] = crval2
    hdr2['UNIT'] = 'erg/s/cm^2/A/arcsec^2'
    hdr2['C0'] = Indices[0]
    hdr2['C1'] = Indices[1]
    hdr2['C2'] = Indices[2]
    hdr2['C3'] = Indices[3]
    hdr2['C4'] = Indices[4]
    hdr2['C5'] = Indices[5]
    hdr2['C6'] = Indices[6]
    hdr2['C7'] = Indices[7]
    hdr2['C8'] = Indices[8]
    hdr2['C9'] = Indices[9]
    hdr2['C10'] = Indices[10]
    hdr2['C11'] = Indices[11]
    hdr2['C12'] = Indices[12]
    hdr2['C13'] = Indices[13]
    hdr2['C14'] = Indices[14]
    hdr2['C15'] = Indices[15]
    hdr2['C16'] = Indices[16]
    hdr2['C17'] = Indices[17]
    hdr2['C18'] = Indices[18]
    hdr2['C19'] = Indices[19]
    hdr2['C20'] = Indices[20]
    hdr2['C21'] = Indices[21]
    hdr2['C22'] = Indices[22]
    hdr2['C23'] = Indices[23]
    hdr2['C24'] = Indices[24]
    hdr2['C25'] = Indices[25]
    hdr2['C26'] = Indices[26]
    hdr2['C27'] = Indices[27]
    hdr2['C28'] = Indices[28]
    hdr2['C29'] = Indices[29]
    hdr2['C30'] = Indices[30]
    hdr2['C31'] = Indices[31]
    hdr2['C32'] = Indices[32]
    hdr2['C33'] = Indices[33]
    hdr2['C34'] = Indices[34]
    hdr2['C35'] = Indices[35]
    hdr2['C36'] = Indices[36]
    hdr2['C37'] = Indices[37]
    hdr2['C38'] = Indices[38]
    hdr2['C39'] = Indices[39]
    hdr2['C40'] = Indices[40]
    hdr2['C41'] = Indices[41]
    hdr2['C42'] = Indices[42]

    
    hdr3 = hdu[3].header
    hdr3['PLATE'] = plate
    hdr3['IFUDSGN'] = np.int(ifu)
    hdr3['EXTNAME'] = 'SPECINDX_FLUX_SIGMA'
    hdr3['CTYPE1'] = 'RA---TAN'
    hdr3['CTYPE2'] = 'DEC--TAN'
    hdr3['CDELT1'] = wcs_N.wcs.cdelt[0]
    hdr3['CDELT2'] = wcs_N.wcs.cdelt[1]
    hdr3['CRPIX1'] = cpix1
    hdr3['CRPIX2'] = cpix2
    hdr3['CUNIT1'] = 'deg'
    hdr3['CUNIT2'] = 'deg'
    hdr3['RADESYS'] = 'FK5'
    hdr3['CRVAL1'] = crval1
    hdr3['CRVAL2'] = crval2
    hdr3['UNIT'] = 'erg/s/cm^2/arcsec^2'
    hdr3['C0'] = Indices[0]
    hdr3['C1'] = Indices[1]
    hdr3['C2'] = Indices[2]
    hdr3['C3'] = Indices[3]
    hdr3['C4'] = Indices[4]
    hdr3['C5'] = Indices[5]
    hdr3['C6'] = Indices[6]
    hdr3['C7'] = Indices[7]
    hdr3['C8'] = Indices[8]
    hdr3['C9'] = Indices[9]
    hdr3['C10'] = Indices[10]
    hdr3['C11'] = Indices[11]
    hdr3['C12'] = Indices[12]
    hdr3['C13'] = Indices[13]
    hdr3['C14'] = Indices[14]
    hdr3['C15'] = Indices[15]
    hdr3['C16'] = Indices[16]
    hdr3['C17'] = Indices[17]
    hdr3['C18'] = Indices[18]
    hdr3['C19'] = Indices[19]
    hdr3['C20'] = Indices[20]
    hdr3['C21'] = Indices[21]
    hdr3['C22'] = Indices[22]
    hdr3['C23'] = Indices[23]
    hdr3['C24'] = Indices[24]
    hdr3['C25'] = Indices[25]
    hdr3['C26'] = Indices[26]
    hdr3['C27'] = Indices[27]
    hdr3['C28'] = Indices[28]
    hdr3['C29'] = Indices[29]
    hdr3['C30'] = Indices[30]
    hdr3['C31'] = Indices[31]
    hdr3['C32'] = Indices[32]
    hdr3['C33'] = Indices[33]
    hdr3['C34'] = Indices[34]
    hdr3['C35'] = Indices[35]
    hdr3['C36'] = Indices[36]
    hdr3['C37'] = Indices[37]
    hdr3['C38'] = Indices[38]
    hdr3['C39'] = Indices[39]
    hdr3['C40'] = Indices[40]
    hdr3['C41'] = Indices[41]
    hdr3['C42'] = Indices[42]


    hdr4 = hdu[4].header
    hdr4['PLATE'] = plate
    hdr4['IFUDSGN'] = np.int(ifu)
    hdr4['EXTNAME'] = 'SPECINDX_CONT_SIGMA'
    hdr4['CTYPE1'] = 'RA---TAN'
    hdr4['CTYPE2'] = 'DEC--TAN'
    hdr4['CDELT1'] = wcs_N.wcs.cdelt[0]
    hdr4['CDELT2'] = wcs_N.wcs.cdelt[1]
    hdr4['CRPIX1'] = cpix1
    hdr4['CRPIX2'] = cpix2
    hdr4['CUNIT1'] = 'deg'
    hdr4['CUNIT2'] = 'deg'
    hdr4['RADESYS'] = 'FK5'
    hdr4['CRVAL1'] = crval1
    hdr4['CRVAL2'] = crval2
    hdr4['UNIT'] = 'erg/s/cm^2/A/arcsec^2'
    hdr4['C0'] = Indices[0]
    hdr4['C1'] = Indices[1]
    hdr4['C2'] = Indices[2]
    hdr4['C3'] = Indices[3]
    hdr4['C4'] = Indices[4]
    hdr4['C5'] = Indices[5]
    hdr4['C6'] = Indices[6]
    hdr4['C7'] = Indices[7]
    hdr4['C8'] = Indices[8]
    hdr4['C9'] = Indices[9]
    hdr4['C10'] = Indices[10]
    hdr4['C11'] = Indices[11]
    hdr4['C12'] = Indices[12]
    hdr4['C13'] = Indices[13]
    hdr4['C14'] = Indices[14]
    hdr4['C15'] = Indices[15]
    hdr4['C16'] = Indices[16]
    hdr4['C17'] = Indices[17]
    hdr4['C18'] = Indices[18]
    hdr4['C19'] = Indices[19]
    hdr4['C20'] = Indices[20]
    hdr4['C21'] = Indices[21]
    hdr4['C22'] = Indices[22]
    hdr4['C23'] = Indices[23]
    hdr4['C24'] = Indices[24]
    hdr4['C25'] = Indices[25]
    hdr4['C26'] = Indices[26]
    hdr4['C27'] = Indices[27]
    hdr4['C28'] = Indices[28]
    hdr4['C29'] = Indices[29]
    hdr4['C30'] = Indices[30]
    hdr4['C31'] = Indices[31]
    hdr4['C32'] = Indices[32]
    hdr4['C33'] = Indices[33]
    hdr4['C34'] = Indices[34]
    hdr4['C35'] = Indices[35]
    hdr4['C36'] = Indices[36]
    hdr4['C37'] = Indices[37]
    hdr4['C38'] = Indices[38]
    hdr4['C39'] = Indices[39]
    hdr4['C40'] = Indices[40]
    hdr4['C41'] = Indices[41]
    hdr4['C42'] = Indices[42]

    hdr5 = hdu[5].header
    hdr5['PLATE'] = plate
    hdr5['IFUDSGN'] = np.int(ifu)
    hdr5['EXTNAME'] = 'SPECINDX_MASK'
    hdr5['CTYPE1'] = 'RA---TAN'
    hdr5['CTYPE2'] = 'DEC--TAN'
    hdr5['CDELT1'] = wcs_N.wcs.cdelt[0]
    hdr5['CDELT2'] = wcs_N.wcs.cdelt[1]
    hdr5['CRPIX1'] = cpix1
    hdr5['CRPIX2'] = cpix2
    hdr5['CUNIT1'] = 'deg'
    hdr5['CUNIT2'] = 'deg'
    hdr5['RADESYS'] = 'FK5'
    hdr5['CRVAL1'] = crval1
    hdr5['CRVAL2'] = crval2
    hdr5['C0'] = Indices[0]
    hdr5['C1'] = Indices[1]
    hdr5['C2'] = Indices[2]
    hdr5['C3'] = Indices[3]
    hdr5['C4'] = Indices[4]
    hdr5['C5'] = Indices[5]
    hdr5['C6'] = Indices[6]
    hdr5['C7'] = Indices[7]
    hdr5['C8'] = Indices[8]
    hdr5['C9'] = Indices[9]
    hdr5['C10'] = Indices[10]
    hdr5['C11'] = Indices[11]
    hdr5['C12'] = Indices[12]
    hdr5['C13'] = Indices[13]
    hdr5['C14'] = Indices[14]
    hdr5['C15'] = Indices[15]
    hdr5['C16'] = Indices[16]
    hdr5['C17'] = Indices[17]
    hdr5['C18'] = Indices[18]
    hdr5['C19'] = Indices[19]
    hdr5['C20'] = Indices[20]
    hdr5['C21'] = Indices[21]
    hdr5['C22'] = Indices[22]
    hdr5['C23'] = Indices[23]
    hdr5['C24'] = Indices[24]
    hdr5['C25'] = Indices[25]
    hdr5['C26'] = Indices[26]
    hdr5['C27'] = Indices[27]
    hdr5['C28'] = Indices[28]
    hdr5['C29'] = Indices[29]
    hdr5['C30'] = Indices[30]
    hdr5['C31'] = Indices[31]
    hdr5['C32'] = Indices[32]
    hdr5['C33'] = Indices[33]
    hdr5['C34'] = Indices[34]
    hdr5['C35'] = Indices[35]
    hdr5['C36'] = Indices[36]
    hdr5['C37'] = Indices[37]
    hdr5['C38'] = Indices[38]
    hdr5['C39'] = Indices[39]
    hdr5['C40'] = Indices[40]
    hdr5['C41'] = Indices[41]
    hdr5['C42'] = Indices[42]
    
    hdrp = hdu[6].header
    hdrp['PLATE'] = plate
    hdrp['IFUDSGN'] = np.int(ifu)
    hdrp['EXTNAME'] = 'COMBINED_DISP'
    hdrp['CTYPE1'] = 'RA---TAN'
    hdrp['CTYPE2'] = 'DEC--TAN'
    hdrp['CDELT1'] = wcs_N.wcs.cdelt[0]
    hdrp['CDELT2'] = wcs_N.wcs.cdelt[1]
    hdrp['CRPIX1'] = cpix1
    hdrp['CRPIX2'] = cpix2
    hdrp['CUNIT1'] = 'deg'
    hdrp['CUNIT2'] = 'deg'
    hdrp['RADESYS'] = 'FK5'
    hdrp['UNIT'] = 'km/s'
    hdrp['CRVAL1'] = crval1
    hdrp['CRVAL2'] = crval2
    hdrp['C0'] = Indices[0]
    hdrp['C1'] = Indices[1]
    hdrp['C2'] = Indices[2]
    hdrp['C3'] = Indices[3]
    hdrp['C4'] = Indices[4]
    hdrp['C5'] = Indices[5]
    hdrp['C6'] = Indices[6]
    hdrp['C7'] = Indices[7]
    hdrp['C8'] = Indices[8]
    hdrp['C9'] = Indices[9]
    hdrp['C10'] = Indices[10]
    hdrp['C11'] = Indices[11]
    hdrp['C12'] = Indices[12]
    hdrp['C13'] = Indices[13]
    hdrp['C14'] = Indices[14]
    hdrp['C15'] = Indices[15]
    hdrp['C16'] = Indices[16]
    hdrp['C17'] = Indices[17]
    hdrp['C18'] = Indices[18]
    hdrp['C19'] = Indices[19]
    hdrp['C20'] = Indices[20]
    hdrp['C21'] = Indices[21]
    hdrp['C22'] = Indices[22]
    hdrp['C23'] = Indices[23]
    hdrp['C24'] = Indices[24]
    hdrp['C25'] = Indices[25]
    hdrp['C26'] = Indices[26]
    hdrp['C27'] = Indices[27]
    hdrp['C28'] = Indices[28]
    hdrp['C29'] = Indices[29]
    hdrp['C30'] = Indices[30]
    hdrp['C31'] = Indices[31]
    hdrp['C32'] = Indices[32]
    hdrp['C33'] = Indices[33]
    hdrp['C34'] = Indices[34]
    hdrp['C35'] = Indices[35]
    hdrp['C36'] = Indices[36]
    hdrp['C37'] = Indices[37]
    hdrp['C38'] = Indices[38]
    hdrp['C39'] = Indices[39]
    hdrp['C40'] = Indices[40]
    hdrp['C41'] = Indices[41]
    hdrp['C42'] = Indices[42]
    
    hdrp1 = hdu[7].header
    hdrp1['PLATE'] = plate
    hdrp1['IFUDSGN'] = np.int(ifu)
    hdrp1['EXTNAME'] = 'COMBINED_DISP_SIGMA'
    hdrp1['CTYPE1'] = 'RA---TAN'
    hdrp1['CTYPE2'] = 'DEC--TAN'
    hdrp1['CDELT1'] = wcs_N.wcs.cdelt[0]
    hdrp1['CDELT2'] = wcs_N.wcs.cdelt[1]
    hdrp1['CRPIX1'] = cpix1
    hdrp1['CRPIX2'] = cpix2
    hdrp1['CUNIT1'] = 'deg'
    hdrp1['CUNIT2'] = 'deg'
    hdrp1['RADESYS'] = 'FK5'
    hdrp1['UNIT'] = 'km/s'
    hdrp1['CRVAL1'] = crval1
    hdrp1['CRVAL2'] = crval2
    hdrp1['C0'] = Indices[0]
    hdrp1['C1'] = Indices[1]
    hdrp1['C2'] = Indices[2]
    hdrp1['C3'] = Indices[3]
    hdrp1['C4'] = Indices[4]
    hdrp1['C5'] = Indices[5]
    hdrp1['C6'] = Indices[6]
    hdrp1['C7'] = Indices[7]
    hdrp1['C8'] = Indices[8]
    hdrp1['C9'] = Indices[9]
    hdrp1['C10'] = Indices[10]
    hdrp1['C11'] = Indices[11]
    hdrp1['C12'] = Indices[12]
    hdrp1['C13'] = Indices[13]
    hdrp1['C14'] = Indices[14]
    hdrp1['C15'] = Indices[15]
    hdrp1['C16'] = Indices[16]
    hdrp1['C17'] = Indices[17]
    hdrp1['C18'] = Indices[18]
    hdrp1['C19'] = Indices[19]
    hdrp1['C20'] = Indices[20]
    hdrp1['C21'] = Indices[21]
    hdrp1['C22'] = Indices[22]
    hdrp1['C23'] = Indices[23]
    hdrp1['C24'] = Indices[24]
    hdrp1['C25'] = Indices[25]
    hdrp1['C26'] = Indices[26]
    hdrp1['C27'] = Indices[27]
    hdrp1['C28'] = Indices[28]
    hdrp1['C29'] = Indices[29]
    hdrp1['C30'] = Indices[30]
    hdrp1['C31'] = Indices[31]
    hdrp1['C32'] = Indices[32]
    hdrp1['C33'] = Indices[33]
    hdrp1['C34'] = Indices[34]
    hdrp1['C35'] = Indices[35]
    hdrp1['C36'] = Indices[36]
    hdrp1['C37'] = Indices[37]
    hdrp1['C38'] = Indices[38]
    hdrp1['C39'] = Indices[39]
    hdrp1['C40'] = Indices[40]
    hdrp1['C41'] = Indices[41]
    hdrp1['C42'] = Indices[42]
    
    hdrp2 = hdu[8].header
    hdrp2['PLATE'] = plate
    hdrp2['IFUDSGN'] = np.int(ifu)
    hdrp2['EXTNAME'] = 'COMBINED_DISP_MASK'
    hdrp2['CTYPE1'] = 'RA---TAN'
    hdrp2['CTYPE2'] = 'DEC--TAN'
    hdrp2['CDELT1'] = wcs_N.wcs.cdelt[0]
    hdrp2['CDELT2'] = wcs_N.wcs.cdelt[1]
    hdrp2['CRPIX1'] = cpix1
    hdrp2['CRPIX2'] = cpix2
    hdrp2['CUNIT1'] = 'deg'
    hdrp2['CUNIT2'] = 'deg'
    hdrp2['RADESYS'] = 'FK5'
    hdrp2['CRVAL1'] = crval1
    hdrp2['CRVAL2'] = crval2
    hdrp2['C0'] = Indices[0]
    hdrp2['C1'] = Indices[1]
    hdrp2['C2'] = Indices[2]
    hdrp2['C3'] = Indices[3]
    hdrp2['C4'] = Indices[4]
    hdrp2['C5'] = Indices[5]
    hdrp2['C6'] = Indices[6]
    hdrp2['C7'] = Indices[7]
    hdrp2['C8'] = Indices[8]
    hdrp2['C9'] = Indices[9]
    hdrp2['C10'] = Indices[10]
    hdrp2['C11'] = Indices[11]
    hdrp2['C12'] = Indices[12]
    hdrp2['C13'] = Indices[13]
    hdrp2['C14'] = Indices[14]
    hdrp2['C15'] = Indices[15]
    hdrp2['C16'] = Indices[16]
    hdrp2['C17'] = Indices[17]
    hdrp2['C18'] = Indices[18]
    hdrp2['C19'] = Indices[19]
    hdrp2['C20'] = Indices[20]
    hdrp2['C21'] = Indices[21]
    hdrp2['C22'] = Indices[22]
    hdrp2['C23'] = Indices[23]
    hdrp2['C24'] = Indices[24]
    hdrp2['C25'] = Indices[25]
    hdrp2['C26'] = Indices[26]
    hdrp2['C27'] = Indices[27]
    hdrp2['C28'] = Indices[28]
    hdrp2['C29'] = Indices[29]
    hdrp2['C30'] = Indices[30]
    hdrp2['C31'] = Indices[31]
    hdrp2['C32'] = Indices[32]
    hdrp2['C33'] = Indices[33]
    hdrp2['C34'] = Indices[34]
    hdrp2['C35'] = Indices[35]
    hdrp2['C36'] = Indices[36]
    hdrp2['C37'] = Indices[37]
    hdrp2['C38'] = Indices[38]
    hdrp2['C39'] = Indices[39]
    hdrp2['C40'] = Indices[40]
    hdrp2['C41'] = Indices[41]
    hdrp2['C42'] = Indices[42]

    hdr6 = hdu[9].header
    hdr6['PLATE'] = plate
    hdr6['IFUDSGN'] = np.int(ifu)
    hdr6['EXTNAME'] = 'ELINE_FLUX'
    hdr6['CTYPE1'] = 'RA---TAN'
    hdr6['CTYPE2'] = 'DEC--TAN'
    hdr6['CDELT1'] = wcs_N.wcs.cdelt[0]
    hdr6['CDELT2'] = wcs_N.wcs.cdelt[1]
    hdr6['CRPIX1'] = cpix1
    hdr6['CRPIX2'] = cpix2
    hdr6['CUNIT1'] = 'deg'
    hdr6['CUNIT2'] = 'deg'
    hdr6['RADESYS'] = 'FK5'
    hdr6['CRVAL1'] = crval1
    hdr6['CRVAL2'] = crval2
    hdr6['Unit'] = '10^(-17) erg/s/cm^2/arsec^2'
    hdr6['C0'] = 'OII-3727'
    hdr6['C1'] = 'OII-3729'
    hdr6['C2'] = 'Hthe-3798'
    hdr6['C3'] = 'Heta-3836'
    hdr6['C4'] = 'NeIII-3869'
    hdr6['C5'] = 'Hzet-3890'
    hdr6['C6'] = 'NeIII-3968'
    hdr6['C7'] = 'Heps-3971'
    hdr6['C8'] = 'Hdel-4102'
    hdr6['C9'] = 'Hgam-4341'
    hdr6['C10'] = 'HeII-4687'
    hdr6['C11'] = 'Hb-4862'
    hdr6['C12'] = 'OIII-4960'
    hdr6['C13'] = 'OIII-5008'
    hdr6['C14'] = 'HeI-5877'
    hdr6['C15'] = 'OI-6302'
    hdr6['C16'] = 'OI-6365'
    hdr6['C17'] = 'NII-6549'
    hdr6['C18'] = 'Ha-6564'
    hdr6['C19'] = 'NII-6585'
    hdr6['C20'] = 'SII-6718'
    hdr6['C21'] = 'SII-6732'
    
    hdr7 = hdu[10].header
    hdr7['PLATE'] = plate
    hdr7['IFUDSGN'] = np.int(ifu)
    hdr7['EXTNAME'] = 'ELINE_FLUX_SIGMA'
    hdr7['CTYPE1'] = 'RA---TAN'
    hdr7['CTYPE2'] = 'DEC--TAN'
    hdr7['CDELT1'] = wcs_N.wcs.cdelt[0]
    hdr7['CDELT2'] = wcs_N.wcs.cdelt[1]
    hdr7['CRPIX1'] = cpix1
    hdr7['CRPIX2'] = cpix2
    hdr7['CUNIT1'] = 'deg'
    hdr7['CUNIT2'] = 'deg'
    hdr7['RADESYS'] = 'FK5'
    hdr7['CRVAL1'] = crval1
    hdr7['CRVAL2'] = crval2
    hdr7['Unit'] = '10^(-17) erg/s/cm^2/arsec^2'
    hdr7['C0'] = 'OII-3727'
    hdr7['C1'] = 'OII-3729'
    hdr7['C2'] = 'Hthe-3798'
    hdr7['C3'] = 'Heta-3836'
    hdr7['C4'] = 'NeIII-3869'
    hdr7['C5'] = 'Hzet-3890'
    hdr7['C6'] = 'NeIII-3968'
    hdr7['C7'] = 'Heps-3971'
    hdr7['C8'] = 'Hdel-4102'
    hdr7['C9'] = 'Hgam-4341'
    hdr7['C10'] = 'HeII-4687'
    hdr7['C11'] = 'Hb-4862'
    hdr7['C12'] = 'OIII-4960'
    hdr7['C13'] = 'OIII-5008'
    hdr7['C14'] = 'HeI-5877'
    hdr7['C15'] = 'OI-6302'
    hdr7['C16'] = 'OI-6365'
    hdr7['C17'] = 'NII-6549'
    hdr7['C18'] = 'Ha-6564'
    hdr7['C19'] = 'NII-6585'
    hdr7['C20'] = 'SII-6718'
    hdr7['C21'] = 'SII-6732'
    
    hdr8 = hdu[11].header
    hdr8['PLATE'] = plate
    hdr8['IFUDSGN'] = np.int(ifu)
    hdr8['EXTNAME'] = 'ELINE_FLUX_MASK'
    hdr8['CTYPE1'] = 'RA---TAN'
    hdr8['CTYPE2'] = 'DEC--TAN'
    hdr8['CDELT1'] = wcs_N.wcs.cdelt[0]
    hdr8['CDELT2'] = wcs_N.wcs.cdelt[1]
    hdr8['CRPIX1'] = cpix1
    hdr8['CRPIX2'] = cpix2
    hdr8['CUNIT1'] = 'deg'
    hdr8['CUNIT2'] = 'deg'
    hdr8['RADESYS'] = 'FK5'
    hdr8['CRVAL1'] = crval1
    hdr8['CRVAL2'] = crval2
    hdr8['C0'] = 'OII-3727'
    hdr8['C1'] = 'OII-3729'
    hdr8['C2'] = 'Hthe-3798'
    hdr8['C3'] = 'Heta-3836'
    hdr8['C4'] = 'NeIII-3869'
    hdr8['C5'] = 'Hzet-3890'
    hdr8['C6'] = 'NeIII-3968'
    hdr8['C7'] = 'Heps-3971'
    hdr8['C8'] = 'Hdel-4102'
    hdr8['C9'] = 'Hgam-4341'
    hdr8['C10'] = 'HeII-4687'
    hdr8['C11'] = 'Hb-4862'
    hdr8['C12'] = 'OIII-4960'
    hdr8['C13'] = 'OIII-5008'
    hdr8['C14'] = 'HeI-5877'
    hdr8['C15'] = 'OI-6302'
    hdr8['C16'] = 'OI-6365'
    hdr8['C17'] = 'NII-6549'
    hdr8['C18'] = 'Ha-6564'
    hdr8['C19'] = 'NII-6585'
    hdr8['C20'] = 'SII-6718'
    hdr8['C21'] = 'SII-6732'

    hdr9 = hdu[12].header
    hdr9['PLATE'] = plate
    hdr9['IFUDSGN'] = np.int(ifu)
    hdr9['EXTNAME'] = 'ELINE_EW'
    hdr9['CTYPE1'] = 'RA---TAN'
    hdr9['CTYPE2'] = 'DEC--TAN'
    hdr9['CDELT1'] = wcs_N.wcs.cdelt[0]
    hdr9['CDELT2'] = wcs_N.wcs.cdelt[1]
    hdr9['CRPIX1'] = cpix1
    hdr9['CRPIX2'] = cpix2
    hdr9['CUNIT1'] = 'deg'
    hdr9['CUNIT2'] = 'deg'
    hdr9['RADESYS'] = 'FK5'
    hdr9['CRVAL1'] = crval1
    hdr9['CRVAL2'] = crval2
    hdr9['Unit'] = 'A'
    hdr9['C0'] = 'OII-3727'
    hdr9['C1'] = 'OII-3729'
    hdr9['C2'] = 'Hthe-3798'
    hdr9['C3'] = 'Heta-3836'
    hdr9['C4'] = 'NeIII-3869'
    hdr9['C5'] = 'Hzet-3890'
    hdr9['C6'] = 'NeIII-3968'
    hdr9['C7'] = 'Heps-3971'
    hdr9['C8'] = 'Hdel-4102'
    hdr9['C9'] = 'Hgam-4341'
    hdr9['C10'] = 'HeII-4687'
    hdr9['C11'] = 'Hb-4862'
    hdr9['C12'] = 'OIII-4960'
    hdr9['C13'] = 'OIII-5008'
    hdr9['C14'] = 'HeI-5877'
    hdr9['C15'] = 'OI-6302'
    hdr9['C16'] = 'OI-6365'
    hdr9['C17'] = 'NII-6549'
    hdr9['C18'] = 'Ha-6564'
    hdr9['C19'] = 'NII-6585'
    hdr9['C20'] = 'SII-6718'
    hdr9['C21'] = 'SII-6732'

    hdr10 = hdu[13].header
    hdr10['PLATE'] = plate
    hdr10['IFUDSGN'] = np.int(ifu)
    hdr10['EXTNAME'] = 'ELINE_EW_SIGMA'
    hdr10['CTYPE1'] = 'RA---TAN'
    hdr10['CTYPE2'] = 'DEC--TAN'
    hdr10['CDELT1'] = wcs_N.wcs.cdelt[0]
    hdr10['CDELT2'] = wcs_N.wcs.cdelt[1]
    hdr10['CRPIX1'] = cpix1
    hdr10['CRPIX2'] = cpix2
    hdr10['CUNIT1'] = 'deg'
    hdr10['CUNIT2'] = 'deg'
    hdr10['RADESYS'] = 'FK5'
    hdr10['CRVAL1'] = crval1
    hdr10['CRVAL2'] = crval2
    hdr10['Unit'] = 'A'
    hdr10['C0'] = 'OII-3727'
    hdr10['C1'] = 'OII-3729'
    hdr10['C2'] = 'Hthe-3798'
    hdr10['C3'] = 'Heta-3836'
    hdr10['C4'] = 'NeIII-3869'
    hdr10['C5'] = 'Hzet-3890'
    hdr10['C6'] = 'NeIII-3968'
    hdr10['C7'] = 'Heps-3971'
    hdr10['C8'] = 'Hdel-4102'
    hdr10['C9'] = 'Hgam-4341'
    hdr10['C10'] = 'HeII-4687'
    hdr10['C11'] = 'Hb-4862'
    hdr10['C12'] = 'OIII-4960'
    hdr10['C13'] = 'OIII-5008'
    hdr10['C14'] = 'HeI-5877'
    hdr10['C15'] = 'OI-6302'
    hdr10['C16'] = 'OI-6365'
    hdr10['C17'] = 'NII-6549'
    hdr10['C18'] = 'Ha-6564'
    hdr10['C19'] = 'NII-6585'
    hdr10['C20'] = 'SII-6718'
    hdr10['C21'] = 'SII-6732'
    
    hdr11 = hdu[14].header
    hdr11['PLATE'] = plate
    hdr11['IFUDSGN'] = np.int(ifu)
    hdr11['EXTNAME'] = 'ELINE_EW_MASK'
    hdr11['CTYPE1'] = 'RA---TAN'
    hdr11['CTYPE2'] = 'DEC--TAN'
    hdr11['CDELT1'] = wcs_N.wcs.cdelt[0]
    hdr11['CDELT2'] = wcs_N.wcs.cdelt[1]
    hdr11['CRPIX1'] = cpix1
    hdr11['CRPIX2'] = cpix2
    hdr11['CUNIT1'] = 'deg'
    hdr11['CUNIT2'] = 'deg'
    hdr11['RADESYS'] = 'FK5'
    hdr11['CRVAL1'] = crval1
    hdr11['CRVAL2'] = crval2
    hdr11['C0'] = 'OII-3727'
    hdr11['C1'] = 'OII-3729'
    hdr11['C2'] = 'Hthe-3798'
    hdr11['C3'] = 'Heta-3836'
    hdr11['C4'] = 'NeIII-3869'
    hdr11['C5'] = 'Hzet-3890'
    hdr11['C6'] = 'NeIII-3968'
    hdr11['C7'] = 'Heps-3971'
    hdr11['C8'] = 'Hdel-4102'
    hdr11['C9'] = 'Hgam-4341'
    hdr11['C10'] = 'HeII-4687'
    hdr11['C11'] = 'Hb-4862'
    hdr11['C12'] = 'OIII-4960'
    hdr11['C13'] = 'OIII-5008'
    hdr11['C14'] = 'HeI-5877'
    hdr11['C15'] = 'OI-6302'
    hdr11['C16'] = 'OI-6365'
    hdr11['C17'] = 'NII-6549'
    hdr11['C18'] = 'Ha-6564'
    hdr11['C19'] = 'NII-6585'
    hdr11['C20'] = 'SII-6718'
    hdr11['C21'] = 'SII-6732'

    hdr12 = hdu[15].header
    hdr12['EXTNAME'] = 'SWIFT/SDSS'
    hdr12['CTYPE1'] = 'RA---TAN'
    hdr12['CTYPE2'] = 'DEC--TAN'
    hdr12['CDELT1'] = wcs_N.wcs.cdelt[0]
    hdr12['CDELT2'] = wcs_N.wcs.cdelt[1]
    hdr12['CRPIX1'] = cpix1
    hdr12['CRPIX2'] = cpix2
    hdr12['CUNIT1'] = 'deg'
    hdr12['CUNIT2'] = 'deg'
    hdr12['RADESYS'] = 'FK5'
    hdr12['CRVAL1'] = crval1
    hdr12['CRVAL2'] = crval2
    hdr12['Unit'] = 'Nanomaggies'
    hdr12['C0'] = 'UVW2'
    hdr12['C1'] = 'UVW1'
    hdr12['C2'] = 'UVM2'
    hdr12['C3'] = 'SDSS u'
    hdr12['C4'] = 'SDSS g'
    hdr12['C5'] = 'SDSS r'
    hdr12['C6'] = 'SDSS i'
    hdr12['C7'] = 'SDSS z'

    hdr13 = hdu[16].header
    hdr13['EXTNAME'] = 'SWIFT/SDSS_SIGMA'
    hdr13['CTYPE1'] = 'RA---TAN'
    hdr13['CTYPE2'] = 'DEC--TAN'
    hdr13['CDELT1'] = wcs_N.wcs.cdelt[0]
    hdr13['CDELT2'] = wcs_N.wcs.cdelt[1]
    hdr13['CRPIX1'] = cpix1
    hdr13['CRPIX2'] = cpix2
    hdr13['CUNIT1'] = 'deg'
    hdr13['CUNIT2'] = 'deg'
    hdr13['RADESYS'] = 'FK5'
    hdr13['CRVAL1'] = crval1
    hdr13['CRVAL2'] = crval2
    hdr13['Unit'] = 'Nanomaggies'
    hdr13['C0'] = 'UVW2'
    hdr13['C1'] = 'UVW1'
    hdr13['C2'] = 'UVM2'
    hdr13['C3'] = 'SDSS u'
    hdr13['C4'] = 'SDSS g'
    hdr13['C5'] = 'SDSS r'
    hdr13['C6'] = 'SDSS i'
    hdr13['C7'] = 'SDSS z'

    hdr14 = hdu[17].header
    hdr14['EXTNAME'] = 'SWIFT_UVOT'
    hdr14['CTYPE1'] = 'RA---TAN'
    hdr14['CTYPE2'] = 'DEC--TAN'
    hdr14['CDELT1'] = wcs_N.wcs.cdelt[0]
    hdr14['CDELT2'] = wcs_N.wcs.cdelt[1]
    hdr14['CRPIX1'] = cpix1
    hdr14['CRPIX2'] = cpix2
    hdr14['CUNIT1'] = 'deg'
    hdr14['CUNIT2'] = 'deg'
    hdr14['RADESYS'] = 'FK5'
    hdr14['CRVAL1'] = crval1
    hdr14['CRVAL2'] = crval2
    hdr14['ABZP_W2'] = abz_W2
    hdr14['FLAMBDA_W2'] = convfact_W2
    hdr14['SKY_W2'] = skycnts_W2
    hdr14['ESKY_W2'] = eskycnts_W2
    hdr14['ABZP_W1'] = abz_W1
    hdr14['FLAMBDA_W1'] = convfact_W1
    hdr14['SKY_W1'] = skycnts_W1
    hdr14['ESKY_W1'] = eskycnts_W1
    hdr14['ABZP_M2'] = abz_M2
    hdr14['FLAMBDA_M2'] = convfact_M2
    hdr14['SKY_M2'] = skycnts_W2
    hdr14['ESKY_M2'] = eskycnts_M2
    hdr14['C0'] = 'UVW2 Counts(Not sky-subtracted)'
    hdr14['C1'] = 'UVW1 Counts(Not sky-subtracted)'
    hdr14['C2'] = 'UVM2 Counts(Not sky-subtracted)'
    hdr14['C3'] = 'UVW2 Counts Err'
    hdr14['C4'] = 'UVW1 Counts Err'
    hdr14['C5'] = 'UVM2 Counts Err'
    hdr14['C6'] = 'UVW2 Exposure'
    hdr14['C7'] = 'UVW1 Exposure'
    hdr14['C8'] = 'UVM2 Exposure'
    hdr14['C9'] = 'UVW2 Mask'
    hdr14['C10'] = 'UVW1 Mask'
    hdr14['C11'] = 'UVM2 Mask'

    hdu.writeto("/Volumes/Nikhil/Data/SWIFT/MPL-7_RP/SwiM_"+str(Line[q])+".fits",overwrite=True)

    




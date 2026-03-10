'''A rough estimation of halo mass function
'''

import numpy as np 
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import hmf
from astropy.cosmology import FlatLambdaCDM


def get_Vmax(cosmo_para, zBGG, mBGG, zlim, mlim, galarea):
    '''
    calculate Vmax based on the method in Yang et al.2021
    '''
    from astropy.cosmology import FlatLambdaCDM
    import numpy as np 

    cosmo = FlatLambdaCDM(H0 = cosmo_para['H0'], Om0 = cosmo_para['Om0'])
    h = cosmo_para['H0'] / 100
    Vratio = galarea / ((180/np.pi)**2*4*np.pi)

    dlimlow = cosmo.comoving_distance(zlim[0]).value#Mpc/h
    Vlowmax = 4/3*np.pi*dlimlow**3 * Vratio

    abM = mBGG - 5*np.log10(cosmo.luminosity_distance(zBGG).value) - 25 #calculate absolution magnitude from BCG
    dmax = 10**((mlim - abM - 25) / 5) #maximum distance from limited luminosity
    dmax_como = dmax / (1 + zBGG)

    dlimup = cosmo.comoving_distance(zlim[1]).value
    Vupmax = np.zeros(zBGG.shape[0])
    Vupmax[dmax_como < dlimup] = 4/3*np.pi*(dmax_como[dmax_como < dlimup])**3 * Vratio
    Vupmax[dmax_como >= dlimup] = 4/3*np.pi*(dlimup)**3 * Vratio

    Vmax = Vupmax - Vlowmax
    return Vmax

def cal_galarea(ra,dec):
    import healpy as hp
    import pandas as pd

    nside = 1024
    totalpixel = hp.nside2npix(nside)

    pxl = hp.ang2pix(nside, np.pi/2.-np.deg2rad(dec), np.deg2rad(ra))
    pxlc = pd.value_counts(pxl)

    gal_area = pxlc.shape[0] / totalpixel * (4*180**2/np.pi)
    print("the cover of galaxy region is %s dge2" %gal_area)

    return gal_area

def cal_Vmax(zbin,  cosmos_para, mlim, galarea):
    sel_redz = np.where((dgroup[:,4] > zbin[0]) & (dgroup[:,4] < zbin[1]))[0]
    BGGid = np.int64(dBCGinfo[sel_redz,1])
    zBGG = digal[BGGid-1,4]
    mBGG = digal[BGGid-1,5]
    
    Vmax = get_Vmax(cosmo_para, zBGG, mBGG, zbin, mlim, galarea)
    HaloM = dgroup[sel_redz, 5]
    return HaloM,Vmax

if __name__ == '__main__':
  dgroup = np.loadtxt('../../../output_magi/cosmos_group')
  dBCGinfo = np.loadtxt('../../../output_magi/cosmos_BCGinfo')
  digal = np.loadtxt('../../../output_magi/cosmos_igal')
  d2 = np.loadtxt('../../../output_magi/icosmos_2')
  d1 = np.loadtxt('../../../output_magi/icosmos_1')

  xmin = 11
  xmax = 15.5
  nbin = 15
  xM = np.linspace(xmin,xmax,nbin+1)
  nx = (xmax-xmin)/nbin/2
  xx = xM[:-1]+nx
  cosmo_para = {'H0':67.4, 'Om0':0.315}
  mlim = 26
  galarea = 10.34
  
  plt.figure(dpi = 100)
  zbins = [[0.0,0.4], [0.4,0.8]]
  for ni,zbin in enumerate(zbins):
      HaloM, Vmax = cal_Vmax(zbin, cosmo_para, mlim, galarea)
      Vmax = Vmax * 0.674**3 #(Mpc/h)^3
      ngroup,xe = np.histogram(HaloM, bins=xM, weights=1/Vmax)
      yy = ngroup/(nx*2)
      plt.plot(xx,np.log10(yy),'x', ms = 5, c = 'C%s' %ni,
              label = '%s < z < %s' %(zbin[0],zbin[1]))
  
      z_mean = np.round((zbin[0]+zbin[1])/2,2)
      h = hmf.hmf.MassFunction(z = z_mean, Mmin = 8, Mmax = 16, sigma_8 = 0.811,
          hmf_model='SMT',transfer_model = 'EH', cosmo_params = cosmo_para)
      plt.plot(np.log10(h.m), np.log10(h.dndlog10m), '-', lw = 1, c ='C%s' %ni, label = 'SMT: z = %s'%z_mean)
  
  # ax = plt.gca()
  # ax.invert_xaxis()
  plt.legend()
  plt.xlim(11,15.5)
  plt.ylim(-8,0)
  # plt.yscale('log')
  
  plt.xlabel(r'$\log M_h[M_{\odot}/h]$')
  plt.ylabel(r'$\Phi/dlogM_{\rm h}/[h^3Mpc^{-3}]$')

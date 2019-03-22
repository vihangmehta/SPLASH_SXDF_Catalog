import os
import numpy as np
import astropy.io.fits as fitsio
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from useful import calc_sn

def get_mags(catalog):

    flx_b,err_b,cov_b = catalog["FLUX_APER_supcam_b"][:,2], catalog["FLUXERR_APER_supcam_b"][:,2], catalog["COVERAGE_FLAG_supcam_b"]
    flx_z,err_z,cov_z = catalog["FLUX_APER_supcam_z"][:,2], catalog["FLUXERR_APER_supcam_z"][:,2], catalog["COVERAGE_FLAG_supcam_z"]
    flx_k,err_k,cov_k = catalog["FLUX_APER_video_ks"][:,2], catalog["FLUXERR_APER_video_ks"][:,2], catalog["COVERAGE_FLAG_video_ks"]

    sn_b = flx_b/err_b
    sn_z = flx_z/err_z
    sn_k = flx_k/err_k

    cond_cov = (cov_b == 1) & (cov_z == 1) & (cov_k == 1)
    cond_det = (sn_b>1) & (sn_z>3) & (sn_k>3) & cond_cov
    cond_non = (sn_b<1) & (sn_z>3) & (sn_k>3) & cond_cov

    mag_b,mag_z,mag_k = np.zeros((3,len(catalog))) - 99.

    mag_b[cond_det] = -2.5*np.log10(flx_b[cond_det]) + 23.93
    mag_z[cond_det] = -2.5*np.log10(flx_z[cond_det]) + 23.93
    mag_k[cond_det] = -2.5*np.log10(flx_k[cond_det]) + 23.93

    mag_b[cond_non] = -2.5*np.log10(err_b[cond_non]) + 23.93
    mag_z[cond_non] = -2.5*np.log10(flx_z[cond_non]) + 23.93
    mag_k[cond_non] = -2.5*np.log10(flx_k[cond_non]) + 23.93

    return mag_b, mag_z, mag_k, catalog["LPH_Z_BEST"], cond_det, cond_non, cond_cov

def get_star_flag(catalog):

    mag_b, mag_z, mag_k, z, cond_det, cond_non, cond_cov = get_mags(catalog)

    cond_chi = (catalog["LPH_CHI_BEST"] - catalog["LPH_CHI_STAR"]) > 0
    cond_bzk = (mag_z - mag_k) < (mag_b - mag_z) * 0.3 - 0.25

    cond_star = cond_cov &  (cond_chi & cond_bzk)
    cond_gal  = cond_cov & ~(cond_chi & cond_bzk)

    print ("Star/Galaxy classification: %i stars and %i gals (no classification for %i)" % (np.sum(cond_star),np.sum(cond_gal),np.sum(~cond_cov)))
    return cond_star, cond_gal, cond_cov

def add_sg_classification(catalog):

    cond_star, cond_gal, cond_cov = get_star_flag(catalog)
    catalog["STAR_FLAG"][cond_star] = 1
    catalog["STAR_FLAG"][cond_gal ] = 0
    catalog["STAR_FLAG"][~cond_cov] = -99
    catalog["ZPHOT"][ cond_star] = 0.00
    return catalog

def mk_plot():

    catalog = fitsio.getdata("final_cats/final_catalog_errfix_zphot.fits")
    mag_b, mag_z, mag_k, z, cond_det, cond_non, cond_cov = get_mags(catalog)
    cond_star,cond_gal,cond_cov = get_star_flag(catalog)

    fig = plt.figure(figsize=(10,9),dpi=75)
    fig.subplots_adjust(left=0.09,right=0.91,top=0.98,bottom=0.08,wspace=0.02,hspace=0.14)
    ogs = gridspec.GridSpec(1,2,width_ratios=[25,1])
    ax  = fig.add_subplot(ogs[0,0])
    cax = fig.add_subplot(ogs[0,1])
    # ax2 = fig.add_subplot(ogs[1,:])

    vmax = 4.5
    cmap = matplotlib.cm.get_cmap('RdYlBu_r')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)

    zz = np.arange(-1,7,0.1)
    ss = np.linspace(0.5,1.5,len(zz)-1)**2

    for z0,z1,s in zip (zz[:-1],zz[1:],ss):
        cond_z = (z0<=z) & (z<z1)
        im = ax.scatter(-99,-99,s=0.5,c=0,cmap=plt.cm.RdYlBu_r,vmin=0,vmax=vmax)
        ax.scatter((mag_b-mag_z)[cond_det&cond_z],(mag_z-mag_k)[cond_det&cond_z],c=cmap(norm(z[cond_det&cond_z])),s=s,alpha=0.4)
        ax.errorbar((mag_b-mag_z)[cond_non&cond_z],(mag_z-mag_k)[cond_non&cond_z],xerr=0.15,xlolims=True,ecolor=cmap(norm(0.5*(z0+z1))),linestyle='',marker='',alpha=0.4)

    ax.scatter((mag_b-mag_z)[cond_det&cond_star],(mag_z-mag_k)[cond_det&cond_star],s=2,c='k',alpha=0.4)
    ax.errorbar((mag_b-mag_z)[cond_non&cond_star],(mag_z-mag_k)[cond_non&cond_star],xerr=0.15,xlolims=True,ecolor='k',linestyle='',marker='',alpha=0.4)
    plt.colorbar(im, cax=cax)

    # ax2.hist(catalog["LPH_Z_BEST"],bins=np.arange(-1,7,0.05),color='k',alpha=0.4)
    # ax2.hist(catalog["LPH_Z_BEST"][cond_star],bins=np.arange(-1,7,0.05),color='k',alpha=0.4)

    ax.set_xlabel("$B-z$",fontsize=24)
    ax.set_ylabel("$z-K$",fontsize=24)
    ax.set_xlim(-0.8,7.8)
    ax.set_ylim(-1.5,4.0)
    cax.set_ylabel("photo-z",fontsize=24)
    # ax2.set_xlabel('z',fontsize=18)
    # ax2.set_xlim(0-0.1,6.1)

    _ = [label.set_fontsize(16) for label in ax.get_yticklabels()+ax.get_xticklabels()+cax.get_yticklabels()]
    # _ = [label.set_visible(False) for label in ax2.get_yticklabels()]

    fig.savefig("final_cats/plots/star_flag_bzk.png")

if __name__ == '__main__':

    mk_plot()
    plt.show()

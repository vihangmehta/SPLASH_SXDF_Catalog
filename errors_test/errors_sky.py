import os,sys
import numpy as np
import scipy.stats
import scipy.optimize
import matplotlib.pyplot as plt
import astropy.io.fits as fitsio
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.optimize import curve_fit

import photutils

import useful
from errors_test import get_orig_fnames

def mk_sky_phot(instr,filt):

    img = fitsio.getdata(orig_names[instr][filt]["img"])

    pixscale = proj_plane_pixel_scales(WCS(fitsio.getheader(orig_names[instr][filt]["img"])))[0] * 3600.
    apersizes = useful.apersizes / pixscale
    aperradii = apersizes / 2.

    sky_apers = fitsio.getdata('/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/errors/sky_apers_%s_%s.fits'%(instr,filt))
    orig_wcs = WCS(fitsio.getheader('/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/data/orig/mosaic_%s_%s.img.fits'%(instr,filt)))
    new_wcs  = WCS(fitsio.getheader(orig_names[instr][filt]["img"]))
    
    pos = np.array(zip(sky_apers['IMGX'],sky_apers['IMGY']))
    pos = orig_wcs.all_pix2world(pos,1)
    pos = new_wcs.all_world2pix(pos,1)

    # cond = (aperradii[-1] < pos[:,0]) & (pos[:,0] < img.shape[1]-aperradii[-1]) & \
    #        (aperradii[-1] < pos[:,1]) & (pos[:,1] < img.shape[0]-aperradii[-1])

    for i,radius in enumerate(aperradii):
        
        sys.stdout.write("\rMeasuring Photometry on sky apertures for %s %s - aperture#%i (%.2f px) ..." % (instr,filt,i+1,radius))
        sys.stdout.flush()
        
        apertures = photutils.CircularAperture(pos,r=radius)
        sky_apers['FLUX_APER'][:,i] = photutils.aperture_photometry(img, apertures)['aperture_sum']

    sys.stdout.write("done!\n")
    sys.stdout.flush()

    fitsio.writeto('sky_apers_orig_%s_%s.fits'%(instr,filt),sky_apers,overwrite=True)

def fit_flux_dist(instr,filt,fluxes,wht,zp,ax=None,c=None,verbose=True):

    def gauss(x, *p):
        mu, sigma = p
        return np.exp(-0.5*(x-mu)**2/sigma**2) / np.sqrt(2*np.pi) / sigma

    dbin = useful.get_binsize(fluxes)
    bins = np.arange(min(fluxes),max(fluxes),dbin)
    binc = 0.5*(bins[1:]+bins[:-1])
    hist = np.histogram(fluxes, bins=bins)[0]
    hist = hist / float(sum(hist)) / dbin
    gsig = np.std(fluxes)

    crit = binc[np.argmax(hist)]
    cond = (binc <= crit)
    hist, binc = hist[cond], binc[cond]

    guess =[crit,gsig]

    coeff, var_matrix = curve_fit(gauss, binc, hist, p0=guess,
                                  bounds=([min(fluxes),1e-6],[crit+5*guess[1],1e6,]))
    mag_limit = -2.5*np.log10(coeff[1]) + zp
    
    if verbose:
        print "%s %s weight: %10.4e -- Fit: [%10.3f, %10.3f] -- Limit: %8.4f (%i apers)" % (
                    instr,filt,wht,coeff[0],coeff[1],mag_limit,len(fluxes))

    if ax:
        xx = np.arange(min(fluxes),max(fluxes),dbin/10.)
        ax.plot(binc,hist,drawstyle='steps-mid',c=c,lw=1,alpha=0.8)
        ax.plot(xx,gauss(xx,*coeff),c=c,lw=1,alpha=1)
        ax.axvline(0,ls='--',lw=1,c='k')
        ax.set_xlim(min(ax.get_xlim()[0],coeff[0]-3*coeff[1]),max(ax.get_xlim()[1],coeff[0]+3*coeff[1]))

    return coeff[0], coeff[1], mag_limit

def get_bbins(instr,filt,N,sky_errors=None):

    whts = sky_errors['AVG_WHT'].flatten()
    cond = (whts!=0)
    if instr == 'uds': cond = cond & (whts>0.0001)
    whts = whts[cond]
    return scipy.stats.mstats.mquantiles(whts,np.linspace(0,1,N+1))

def bin_sky_errors(instr,filt,apersize_num,bbins,zp,sky_errors=None,verbose=True):

    if verbose: print "Aperture #%i (%.2f\")" % (apersize_num+1,useful.apersizes[apersize_num])
    fluxes, whts = sky_errors['FLUX_APER'][:,apersize_num], sky_errors['AVG_WHT'][:,apersize_num]
    
    cond = (whts!=0)
    if instr == 'uds': cond = cond & (whts>0.001)
    fluxes, whts = fluxes[cond], whts[cond]

    flux_mean, flux_err, mag_lim, bin_wht = np.zeros((4,len(bbins)-1))
    digi = np.digitize(whts,bbins) - 1

    # fig,ax = plt.subplots(1,1,figsize=(10,6),dpi=75)
    # ax.set_xlim(-1e-10,1e-10)
    cs = plt.cm.Set1(np.linspace(0,1,len(bbins)-1))
    for i in range(len(bbins)-1):
        bin_wht[i] = np.mean(whts[digi==i])
        flux_mean[i], flux_err[i], mag_lim[i] = fit_flux_dist(instr=instr,filt=filt,fluxes=fluxes[digi==i],wht=bin_wht[i],c=cs[i],zp=zp,verbose=verbose)
    # plt.show()

    return bin_wht, flux_mean, flux_err, mag_lim

def bin_sex_errors(instr,filt,apersize_num,bbins,avg_wht,sex_errors):

    flux_err = sex_errors["FLUXERR_APER"][:,apersize_num]

    cond = (flux_err!=-99.) & (avg_wht!=0)
    avg_wht,flux_err = avg_wht[cond],flux_err[cond]

    digi = np.digitize(avg_wht,bbins) - 1
    bin_wht  = np.array([np.mean(avg_wht[digi==i]) for i in range(len(bbins)-1)])
    flux_err = np.array([np.median(flux_err[digi==i]) for i in range(len(bbins)-1)])
    return bin_wht, flux_err

def get_sex_whts(catalog,instr,filt):

    radius = 3. / 2. / useful.pix_scale

    orig_wcs = WCS(fitsio.getheader('/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/data/orig/mosaic_%s_%s.img.fits'%(instr,filt)))
    new_wcs  = WCS(fitsio.getheader(orig_names[instr][filt]["img"]))
    
    pos  = zip(catalog['X_IMAGE'],catalog['Y_IMAGE'])
    pos = orig_wcs.all_pix2world(pos,1)
    pos = new_wcs.all_world2pix(pos,1)

    wht  = fitsio.getdata('/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/data/orig/mosaic_%s_%s.wht.fits'%(instr,filt))
    aperture = photutils.CircularAperture(pos,r=radius)
    avg_wht  = photutils.aperture_photometry(wht, aperture)['aperture_sum'] / (np.pi*radius**2)

    return avg_wht

def calc_fscale(zp0,zp1=23.93):
    """
    -2.5*log(f1) + zp1 = -2.5*log(f0) + zp0
    f1/f0 = 10**((zp1 - zp0) / 2.5)
    """
    fscale = 10**((zp1 - zp0) / 2.5)
    return fscale

def compare(fname='video_ks'):

    instr,filt = fname.split('_')

    fig,ax = plt.subplots(1,1,figsize=(10,6),dpi=75,tight_layout=True)

    sky_errors_swrp = fitsio.getdata('/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/errors/sky_apers_%s_%s.fits'%(instr,filt))
    # sky_errors_orig = fitsio.getdata('sky_apers_orig_%s_%s.fits'%(instr,filt))
    # sky_errors_orig["AVG_WHT"] = sky_errors_swrp["AVG_WHT"]
    nbins = len(sky_errors_swrp) / 25000 + 1

    sex_errors_swrp = fitsio.getdata('catalog_swrp_%s_%s.fits'%(instr,filt))
    sex_errors_orig = fitsio.getdata('catalog_orig_%s_%s.fits'%(instr,filt))
    sex_avg_wht_orig = get_sex_whts(catalog=sex_errors_orig,instr=instr,filt=filt)
    sex_avg_wht_swrp = get_sex_whts(catalog=sex_errors_swrp,instr=instr,filt=filt)

    bbins = get_bbins(instr=instr,filt=filt,sky_errors=sky_errors_swrp,N=nbins)
    
    bin_wht_sky,flux_mean_sky,flux_err_sky,mag_lim_sky = bin_sky_errors(instr=instr,filt=filt,sky_errors=sky_errors_swrp,apersize_num=2,bbins=bbins,zp=useful.zp)
    bin_wht_sex,flux_err_sex = bin_sex_errors(instr=instr,filt=filt,avg_wht=sex_avg_wht_swrp,sex_errors=sex_errors_swrp,apersize_num=2,bbins=bbins)

    ax.scatter(sex_avg_wht_swrp,sex_errors_swrp["FLUXERR_APER"][:,2],c='k',s=1,alpha=0.1)
    ax.plot(bin_wht_sky,flux_err_sky,c='k',lw=1.5,alpha=0.9)
    ax.plot(bin_wht_sex,flux_err_sex,c='k',lw=1.5,alpha=0.9,ls='--')

    fscale = calc_fscale(useful.orig_zp[instr][filt])
    # bin_wht_sky,flux_mean_sky,flux_err_sky,mag_lim_sky = bin_sky_errors(instr=instr,filt=filt,sky_errors=sky_errors_orig,apersize_num=2,bbins=bbins,zp=useful.orig_zp[instr][filt])
    bin_wht_sex,flux_err_sex = bin_sex_errors(instr=instr,filt=filt,avg_wht=sex_avg_wht_orig,sex_errors=sex_errors_orig,apersize_num=2,bbins=bbins)

    ax.scatter(sex_avg_wht_orig,fscale*sex_errors_orig["FLUXERR_APER"][:,2],c='r',s=1,alpha=0.1)
    # ax.plot(bin_wht_sky,fscale*flux_err_sky,c='r',lw=1.5,alpha=0.9)
    ax.plot(bin_wht_sex,fscale*flux_err_sex,c='r',lw=1.5,alpha=0.9,ls='--')

    ax.set_xscale('log')

if __name__ == '__main__':
    
    orig_names = get_orig_fnames()

    compare()
    plt.show()
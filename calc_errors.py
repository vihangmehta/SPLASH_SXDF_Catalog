import sys,os
import numpy as np
import scipy.stats
import scipy.optimize
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import useful
from mk_sky_errors import get_sky_phot

def fit_flux_dist(instr,filt,fluxes,wht,ax=None,c=None,verbose=True):

    # fluxes = np.clip(fluxes,np.median(fluxes)-5*np.std(fluxes),np.median(fluxes)+5*np.std(fluxes))

    def gauss(x, *p):
        mu, sigma = p
        return np.exp(-0.5*(x-mu)**2/sigma**2) / np.sqrt(2*np.pi) / sigma

    dbin = useful.get_binsize(fluxes)
    bins = np.arange(min(fluxes),max(fluxes),dbin)
    binc = 0.5*(bins[1:]+bins[:-1])
    hist = np.histogram(fluxes, bins=bins)[0]
    hist = hist / float(sum(hist)) / dbin
    gsig = np.std(fluxes)

    crit = binc[np.argmax(hist)] + 0.1*gsig
    cond = (binc <= crit)
    _hist, _binc = hist.copy(), binc.copy()
    hist, binc = hist[cond], binc[cond]

    guess =[crit,gsig]
    # guess = [crit,10**((25.-useful.zp)/-2.5)]

    coeff, var_matrix = curve_fit(gauss, binc, hist, p0=guess,
                                  bounds=([min(fluxes),1e-6],[crit+5*guess[1],1e6,]))
    mag_limit = -2.5*np.log10(coeff[1]) + useful.zp
    
    if verbose:
        print "%s %s weight: %10.4e -- Fit: [%10.3f, %10.3f] -- Limit: %8.4f (%i apers)" % (
                    instr,filt,wht,coeff[0],coeff[1],mag_limit,len(fluxes))

    if ax:
        xx = np.arange(min(fluxes),max(fluxes),dbin/10.)
        ax.plot(binc,hist,drawstyle='steps-mid',c=c,lw=1,alpha=0.8)
        ax.plot(_binc,_hist,drawstyle='steps-mid',c=c,lw=1,ls='--',alpha=0.8)
        ax.plot(xx,gauss(xx,*coeff),c=c,lw=1,alpha=1)
        ax.axvline(0,ls='--',lw=1,c='k')
        ax.set_xlim(min(ax.get_xlim()[0],coeff[0]-3*coeff[1]),max(ax.get_xlim()[1],coeff[0]+3*coeff[1]))

    return coeff[0], coeff[1], mag_limit

def get_bbins(instr,filt,N,sky_errors=None):

    if sky_errors is None: sky_errors = get_sky_phot(instr=instr,filt=filt)
    whts = sky_errors['AVG_WHT'].flatten()
    cond = (whts!=0)
    if instr == 'uds': cond = cond & (whts>0.0001)
    whts = whts[cond]
    return scipy.stats.mstats.mquantiles(whts,np.linspace(0,1,N+1))

def bin_sky_errors(instr,filt,apersize_num,bbins,sky_errors=None,verbose=True):

    if sky_errors is None: sky_errors = get_sky_phot(instr=instr,filt=filt)

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
        flux_mean[i], flux_err[i], mag_lim[i] = fit_flux_dist(instr=instr,filt=filt,fluxes=fluxes[digi==i],wht=bin_wht[i],c=cs[i],verbose=verbose)
    # plt.show()

    return bin_wht, flux_mean, flux_err, mag_lim

def calc_sex_errors(instr,filt,catalog,apersize_num):

    radius = 3. / 2. / useful.pix_scale # Compute the average weights for just 3" aperture

    avg_wht = catalog['AVG_WHT_%s_%s'%(instr,filt)]

    cond = (catalog['MAG_AUTO_%s_%s'%(instr,filt)] != -99.) & \
           (catalog['SE_FLAGS_%s_%s'%(instr,filt)] == 0) & \
           (catalog['COVERAGE_FLAG_%s_%s'%(instr,filt)] == 1) & \
           (catalog['FLUX_APER_%s_%s'%(instr,filt)][:,apersize_num] != -99.) & \
           (catalog['FLUXERR_APER_%s_%s'%(instr,filt)][:,apersize_num] != 0.) & \
           (catalog['MAG_APER_%s_%s'%(instr,filt)][:,apersize_num] > 10)

    _mag_lim1 = np.percentile(catalog['MAG_APER_%s_%s'%(instr,filt)][:,apersize_num][cond],25)
    _mag_lim2 = np.percentile(catalog['MAG_APER_%s_%s'%(instr,filt)][:,apersize_num][cond],75)

    cond = cond & (catalog['MAG_APER_%s_%s'%(instr,filt)][:,apersize_num] > _mag_lim1) & \
                  (catalog['MAG_APER_%s_%s'%(instr,filt)][:,apersize_num] < _mag_lim2)

    flux = catalog['FLUX_APER_%s_%s'%(instr,filt)][:,apersize_num]
    flux_err = catalog['FLUXERR_APER_%s_%s'%(instr,filt)][:,apersize_num]

    flux[~cond] = -99.
    flux_err[~cond] = -99.

    return avg_wht, flux, flux_err

def bin_sex_errors(instr,filt,apersize_num,bbins,catalog):

    avg_wht,flux,flux_err = calc_sex_errors(instr=instr,filt=filt,catalog=catalog,apersize_num=apersize_num)

    cond = (flux_err!=-99.) & (avg_wht>0)
    avg_wht,flux_err = avg_wht[cond],flux_err[cond]

    digi = np.digitize(avg_wht,bbins) - 1
    bin_wht  = np.array([np.mean(avg_wht[digi==i]) for i in range(len(bbins)-1)])
    flux_err = np.array([np.median(flux_err[digi==i]) for i in range(len(bbins)-1)])

    return bin_wht, flux_err

def mk_binned_errors(fixed=False):
    
    sky_errors = get_sky_phot(instr='hsc',filt='y')
    if not fixed:
        catalog = fitsio.getdata('final_cats/final_catalog.fits')
    else:
        catalog = fitsio.getdata('final_cats/final_catalog_errfix.fits')

    apersizes = useful.apersizes
    naper = len(apersizes)
    fnames = [x for x in useful.fnames if 'irac' not in x]

    nbins = len(sky_errors) / 30000 + 1
    errors_binned = np.recarray(len(fnames),
                                dtype=[('filt','|S9'),('wht_edges',float,(nbins+1,naper)),('aper_center',float,(nbins,naper)),
                                       ('wht_center_sky',float,(nbins,naper)),('flux_mean_sky',float,(nbins,naper)),('flux_err_sky',float,(nbins,naper)),('mag_lim_sky',float,(nbins,naper)),
                                       ('wht_center_sex',float,(nbins,naper)),('flux_err_sex',float,(nbins,naper))])

    for i,fname in enumerate(fnames):

        instr, filt = fname.split('_')
        sky_errors = get_sky_phot(instr=instr,filt=filt)
        bbins = get_bbins(instr=instr,filt=filt,sky_errors=sky_errors,N=nbins)

        errors_binned[i]['filt'] = fname
        
        for j in range(nbins): errors_binned[i]['aper_center'][j,:] = np.sqrt(np.pi) * apersizes / 2. / useful.pix_scale
        for j in range(naper):

            bin_wht_sky,flux_mean_sky,flux_err_sky,mag_lim_sky = bin_sky_errors(instr=instr,filt=filt,sky_errors=sky_errors,apersize_num=j,bbins=bbins)
            bin_wht_sex,flux_err_sex = bin_sex_errors(instr=instr,filt=filt,catalog=catalog,apersize_num=j,bbins=bbins)

            errors_binned[i]['wht_edges'][:,j]      = bbins
            errors_binned[i]['wht_center_sky'][:,j] = bin_wht_sky
            errors_binned[i]['wht_center_sex'][:,j] = bin_wht_sex
            errors_binned[i]['flux_mean_sky'][:,j]  = flux_mean_sky
            errors_binned[i]['flux_err_sky'][:,j]   = flux_err_sky
            errors_binned[i]['flux_err_sex'][:,j]   = flux_err_sex
            errors_binned[i]['mag_lim_sky'][:,j]    = mag_lim_sky

    if not fixed:
        fitsio.writeto('errors/errors_binned.fits',errors_binned,overwrite=True)
    else:
        fitsio.writeto('errors/errors_binned_errfix.fits',errors_binned,overwrite=True)

if __name__ == '__main__':
    
    mk_binned_errors()
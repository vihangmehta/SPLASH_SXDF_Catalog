import sys, os
import numpy as np
import scipy.spatial
import scipy.integrate
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
from astropy.wcs import WCS
from sklearn.cluster import KMeans

import useful
from mk_sky_errors import get_sky_phot
from calc_errors import fit_flux_dist,bin_sky_errors,get_bbins
from mk_upper_lims import CalcUpperLimits
from extract_bc03 import TemplateSED_BC03

import cosmolopy.distance as cd
import cosmolopy.constants as cc
cosmo = {'omega_M_0':0.3,'omega_lambda_0':0.7,'h':0.7}
cosmo = cd.set_omega_k_0(cosmo)
light = 2.998e18 # units: angs/sec

plt.rcParams.update({'axes.linewidth': 1.5,
                     'xtick.major.width': 1.5,
                     'ytick.major.width': 1.5,
                     'xtick.minor.width': 1.2,
                     'ytick.minor.width': 1.2,
                     'xtick.major.size': 8,
                     'ytick.major.size': 8,
                     'xtick.minor.size': 5,
                     'ytick.minor.size': 5})

def mk_mag_limit_map(aper_num):

    up_lim = CalcUpperLimits()

    for instr in ["hsc",]:#useful.instr_used_list[:-1]:
        for filt in ["g",]:#useful.filters[instr]:

            sys.stdout.write("\rProcessing %s:%s ... \033[K" % (instr,filt))
            sys.stdout.flush()

            wht,hdr = fitsio.getdata(os.path.join(cwd,"data/orig","mosaic_%s_%s.wht.fits"%(instr,filt)),header=True)
            wht_ravel = wht.ravel()
            wht_shape = wht.shape
            del wht

            mag_lim = np.zeros(wht_ravel.shape,dtype=np.float32)

            # Fix for really large arrays
            idx = np.where(wht_ravel != 0)[0]
            idx_split = np.array_split(idx,10)
            del idx
            for _idx in idx_split:
                mag_lim[_idx] = up_lim(aper=up_lim.apersizes[aper_num],wht=wht_ravel[_idx],instr=instr,filt=filt)
            ###

            # mag_lim[cond] = up_lim(aper=up_lim.apersizes[aper_num],wht=wht_ravel[cond],instr=instr,filt=filt)
            mag_lim = mag_lim.reshape(wht_shape)

            msk = fitsio.getdata(os.path.join(cwd,"data/orig","mosaic_%s_%s.msk.fits"%(instr,filt)))
            mag_lim[msk!=0] = 0
            del msk

            fitsio.writeto(os.path.join(cwd,"mag_limits","mag_lim_%s_%s_r%i.fits"%(instr,filt,aper_num+1)),data=mag_lim,header=hdr,overwrite=True)

    print "done."

def mk_irac_mag_limit_map(aper_num=1):

    instr = 'irac'
    if aper_num!=1: raise Exception("Cannot compute IRAC mag limits for any aperture other than 2\"")

    for filt in useful.filters[instr]:

        sys.stdout.write("\rProcessing %s:%s ...\033[K" % (instr,filt))
        sys.stdout.flush()

        img,hdr = fitsio.getdata("/data/highzgal/PUBLICACCESS/SPLASH/DATA/IRAC/SXDS.irac.%s.mosaic.fits"%filt,header=True)
        sky_errors = fitsio.getdata("errors/sky_apers_%s_%s.fits"%(instr,filt))

        x = sky_errors["IMGX"]
        y = sky_errors["IMGY"]
        z = -2.5 * np.log10(sky_errors["rms"]) + useful.orig_zp[instr][filt]

        tree = scipy.spatial.cKDTree(np.array([x,y]).T)

        zz = np.zeros(img.shape,dtype=np.float32)
        ix = np.array_split(np.arange(img.shape[1]),8)
        iy = np.array_split(np.arange(img.shape[0]),8)

        for j,_iy in enumerate(iy):
            for k,_ix in enumerate(ix):

                sys.stdout.write(".")
                sys.stdout.flush()

                yy,xx = np.meshgrid(_iy,_ix)
                d,i = tree.query(zip(xx.flatten(),yy.flatten()),k=20,distance_upper_bound=750,n_jobs=15)

                cond = (i!=len(x))
                _z = np.zeros(i.shape)
                _z[ cond] = z[i[cond]]
                _z[~cond] = np.NaN

                cond_nan = (np.sum(np.isfinite(_z),axis=-1) > 2)
                zz[yy.flatten()[cond_nan],xx.flatten()[cond_nan]] = np.nanmedian(_z[cond_nan],axis=-1)

        zz[~np.isfinite(img)] = np.NaN
        fitsio.writeto(os.path.join(cwd,"mag_limits","mag_lim_%s_%s_r%i.fits"%(instr,filt,aper_num+1)),data=zz,header=hdr,overwrite=True)

        print " done."

def rebin_mag_limit_maps(aper_num):

    for instr in ["hsc",]:#useful.instr_used_list:

        for filt in ["g",]:#useful.filters[instr]:

            sys.stdout.write("\rProcessing %s:%s ...\033[K" % (instr,filt))
            sys.stdout.flush()

            if instr=='irac' and aper_num!=1:
                print "Ignoring %s:%s for any aperture other than 2\"" % (instr,filt)
                continue

            mag_lim,hdr = fitsio.getdata(os.path.join(cwd,"mag_limits","mag_lim_%s_%s_r%i.fits"%(instr,filt,aper_num+1)),header=True)
            mag_lim = np.ma.masked_array(mag_lim,mask=(~np.isfinite(mag_lim))|(mag_lim==0),dtype=np.float32)

            if instr!='irac':
                factor, pixscale = 8, 0.15
            else:
                if filt in ['1','2']: mag_lim = mag_lim[:-1,:-1]
                if filt in ['3','4']: mag_lim = mag_lim[:-1,:]
                factor, pixscale = 2, 0.60

            n = np.asarray(mag_lim.shape) / factor
            mag_lim = mag_lim.reshape(n[0],factor,n[1],factor)
            mag_lim = np.ma.median(np.ma.median(mag_lim,axis=1),axis=2)
            mag_lim = mag_lim.filled(fill_value=0)

            hdr['CRPIX1'] = (hdr['CRPIX1'] - 0.5)/factor + 0.5
            hdr['CRPIX2'] = (hdr['CRPIX2'] - 0.5)/factor + 0.5
            hdr['CD1_1']  = -pixscale*factor/3600.
            hdr['CD2_2']  = +pixscale*factor/3600.

            fitsio.writeto(os.path.join(cwd,"mag_limits","mag_lim_%s_%s_r%i.rebin.fits"%(instr,filt,aper_num+1)),data=mag_lim,header=hdr,overwrite=True)

    print "done."

def mk_mag_limit_plot(aper_num):

    for instr in ["hsc",]:#useful.instr_used_list:

        for filt in ["g",]:#useful.filters[instr]:

            print instr,filt

            if instr=='irac' and aper_num!=1:
                print "Ignoring %s:%s for any aperture other than 2\"" % (instr,filt)
                continue

            fig = plt.figure(figsize=(10,8.25),dpi=75)
            ogs = gridspec.GridSpec(1,2,width_ratios=[35,1])
            ax  = fig.add_subplot(ogs[0,0])
            cax = fig.add_subplot(ogs[0,1])

            mag_lim,hdr = fitsio.getdata(os.path.join(cwd,"mag_limits","mag_lim_%s_%s_r%i.rebin.fits"%(instr,filt,aper_num+1)),header=True)
            mag_lim[mag_lim==0] = np.NaN
            mag_lim = mag_lim - 2.5*np.log10(5.)

            x = np.arange(mag_lim.shape[1])
            y = np.arange(mag_lim.shape[0])

            wcs = WCS(hdr)
            r,_ = wcs.all_pix2world(x,y[0],1)
            _,d = wcs.all_pix2world(x[0],y,1)

            im = ax.pcolormesh(r,d,mag_lim,lw=0,
                                cmap=plt.cm.Greys,
                                vmin=np.floor(np.nanmin(mag_lim)*4)/4,
                                vmax=np.ceil( np.nanmax(mag_lim)*4)/4)
            plt.colorbar(im, cax=cax)

            ax.text(0.98,0.98,"%s %s"%(instr if instr!='cfht' else 'musubi',filt),color=useful.fcolor_dict[instr][filt],fontsize=30,fontweight=600,va='top',ha='right',transform=ax.transAxes)
            ax.set_xlabel("RA [deg]",fontsize=20)
            ax.set_ylabel("Decl. [deg]",fontsize=20)
            ax.set_xlim(35.55, 33.45)
            ax.set_ylim(-6.05, -3.95)
            ax.set_xticks([35.5,35.0,34.5,34.0,33.5])
            ax.set_yticks([-4.0,-4.5,-5.0,-5.5,-6.0])
            ax.set_aspect(1.)
            _ = [tick.set_fontsize(20) for tick in ax.get_xticklabels()+ax.get_yticklabels()]

            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

            cax.set_ylabel("5$\sigma$ magnitude limit [AB]",fontsize=24)
            _ = [tick.set_fontsize(20) for tick in cax.get_yticklabels()]

            ogs.tight_layout(fig, rect=[0.02, 0, 1, 1],h_pad=0.02)
            fig.savefig("mag_limits/mag_lim_%s_%s_r%i.png"%(instr,filt,aper_num+1))
            plt.close(fig)

def get_mag_limits(mag_lim,fname):

    if   fname in ['irac_1','irac_2']: n_comp = 5
    elif fname in ['irac_3','irac_4']: n_comp = 3
    elif fname in ['cfhtls_z',]:       n_comp = 2
    else: n_comp = 1

    mag_lim.sort()
    data = mag_lim[::40].reshape(-1,1)
    k_means = KMeans(n_clusters=n_comp)
    k_means.fit(data)
    centers = k_means.cluster_centers_
    centers = np.sort(centers.flatten())

    if   fname in ['irac_1','irac_2']: centers_fin = centers[-2]
    elif fname in ['irac_3','irac_4']: centers_fin = centers[-1]
    elif fname in ['cfhtls_z',]:       centers_fin = centers[-1]
    else:
        if 'hsc' in fname: centers_fin = np.median(mag_lim[mag_lim>np.percentile(mag_lim,50)])
        else: centers_fin = np.median(mag_lim)
    return centers, centers_fin

def mk_mag_lim_hists(aper_num):

    _fnames = np.array_split(useful.fnames,4)
    fig = plt.figure(figsize=(20,10),dpi=75)
    fig.subplots_adjust(left=0.02,right=0.98,top=0.98,bottom=0.06,wspace=0.,hspace=0.17)
    ogs = gridspec.GridSpec(4,len(_fnames[0]))

    bins = np.arange(10,40,0.02)

    for i,fname in enumerate(useful.fnames):

        instr,filt = fname.split('_')
        fcolor = useful.fcolor_dict[instr][filt]

        if instr=='irac' and aper_num!=1:
            print "Ignoring %s:%s for any aperture other than 2\"" % (instr,filt)
            continue

        mag_lim = fitsio.getdata(os.path.join(cwd,"mag_limits","mag_lim_%s_%s_r%i.rebin.fits"%(instr,filt,aper_num+1)))
        mag_lim = mag_lim[mag_lim!=0]
        mag_lim = mag_lim - 2.5*np.log10(5.)

        igs = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=ogs[i])
        ax = fig.add_subplot(igs[0])

        binc = 0.5*(bins[1:]+bins[:-1])
        hist = np.histogram(mag_lim,bins=bins)[0]
        ax.plot(binc,hist,c=fcolor,lw=2)
        ax.axvline(np.median(mag_lim),c=fcolor,lw=2)

        centers,centers_fin = get_mag_limits(mag_lim,fname=fname)
        ax.vlines(centers,0,1.15*np.max(hist),colors=fcolor,linestyles='--',lw=1)
        ax.vlines(centers_fin,0,1.15*np.max(hist),colors=fcolor,linestyles='--',lw=2)
        print fname, np.median(mag_lim), centers_fin, centers

        ylim = [min(binc[hist>0.02*max(hist)])-0.25, max(binc[hist>0.02*max(hist)])+0.25]
        ax.set_xlim(ylim)

        ax.set_ylim(0,1.15*np.max(hist))
        ax.set_xlabel('Mag Limit',fontsize=12)
        ax.text(0.05,0.95,"%s %s"%(instr,filt),va='top',ha='left',fontsize=16,fontweight=800,color=fcolor,transform=ax.transAxes)
        _ = [label.set_fontsize(12)   for label in ax.get_xticklabels()]
        _ = [label.set_visible(False) for label in ax.get_yticklabels()]

    fig.savefig(os.path.join(cwd,"mag_limits","mag_lims_hists_r%i.png"%(aper_num+1)))

def get_template():

    template = TemplateSED_BC03(age=0.2, sfh='ssp', tau=None, metallicity=0.02, Av=0,
                                emlines=False, dust='calzetti', redshift=4, igm=True,
                                units='flambda', imf='chab', res='lr',
                                lya_esc=0, lyc_esc=0,
                                rootdir='/data/highzgal/mehta/galaxev/', library_version=2003,
                                workdir='temp/bc03/', cleanup=True, verbose=False)
    template.generate_sed()

    wave = template.sed["waves"]
    flux = template.sed["spec1"]
    mass = template.M_unnorm["spec1"]
    z    = template.redshift

    pivot = 3e4
    cond = (pivot-500<wave) & (wave<pivot+500)
    init_flux = scipy.integrate.simps(flux[cond]*wave[cond],wave[cond]) / scipy.integrate.simps(wave[cond],wave[cond]) * (pivot**2 / light)
    true_flux = 10**(-(23.82 + 48.6)/2.5)
    norm = true_flux / init_flux

    mass_norm = norm*4.*np.pi*(cd.luminosity_distance(z,**cosmo)*cc.Mpc_cm)**2
    mass = np.log10(mass*mass_norm)
    print "Template mass: %.2f (in log solar masses)" % mass

    flux = flux * norm
    flux = flux * (wave**2/light)
    cond = (flux>0)
    mag  = np.zeros_like(flux) + 99.
    mag[cond] = -2.5*np.log10(flux[cond]) - 48.6
    return wave/1e4, mag

def mk_pretty_plot(aper_num):

    fig,ax = plt.subplots(1,1,figsize=(15,8),dpi=75,tight_layout=True)

    fnames,fcolors = [],[]
    for instr in ['hsc','supcam','cfht','uds','x','video','cfhtls','irac']:
        if instr=='x':
            fnames.append("%s_%s"%(instr,instr))
            fcolors.append("none")
            continue
        for filt in useful.filters[instr]:
            fnames.append("%s_%s"%(instr,filt))
            fcolors.append(useful.fcolor_dict[instr][filt])

    for i,(fname,fcolor) in enumerate(zip(fnames,fcolors)):

        instr,filt = fname.split('_')

        if instr=='irac' and aper_num!=1:
            print "Ignoring %s:%s for any aperture other than 2\"" % (instr,filt)
            continue

        if instr=='x':
            ax.add_patch(Rectangle((-99,-99),0,0,lw=0,color=fcolor,alpha=0,label="%s:%s"%(instr,filt)))
            continue

        mag_lim = fitsio.getdata(os.path.join(cwd,"mag_limits","mag_lim_%s_%s_r%i.rebin.fits"%(instr,filt,aper_num+1)))
        mag_lim = mag_lim[mag_lim!=0]
        mag_lim = mag_lim - 2.5*np.log10(5.)

        centers, centers_fin = get_mag_limits(mag_lim,fname=fname)
        print fname, np.median(mag_lim), centers_fin, centers

        mag_lim = centers_fin

        filt_file = "/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/lephare/filters/%s/%s_%s.pb"%(instr,instr,filt)
        filt_waves,filt_sens = np.genfromtxt(filt_file,unpack=True)
        filt_waves = filt_waves[filt_sens>0.25*max(filt_sens)]
        filt_waves = filt_waves * 1e-4

        x0,x1 = min(filt_waves),max(filt_waves)
        yc = mag_lim
        dx,dy = x1-x0,0.06
        ax.add_patch(Rectangle((x0,yc-dy/2.),dx,dy,lw=0,color=fcolor,label="%s:%s"%(instr if instr!='cfht' else 'musubi',filt)))

    ax.set_xlabel("Wavelength [$\\mu$m]",fontsize=24)
    ax.set_ylabel("5$\\sigma$ Magnitude Limit [AB] (%i\" aperture)"%(aper_num+1),fontsize=24)
    ax.set_xscale('log')
    ax.set_xlim(0.3,10)
    if aper_num==0: ax.set_ylim(29,23.5)
    if aper_num==1: ax.set_ylim(28,22.5)
    if aper_num==2: ax.set_ylim(27.2,22)
    xticks = [0.3,0.4,0.6,0.5,0.8,1,1.5,2,3,4,5,6,8,10]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(axis="both",which="both",direction='in')
    _ = [tick.set_fontsize(24) for tick in ax.get_xticklabels()+ax.get_yticklabels()]

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    wave,mag = get_template()
    ax.plot(wave,mag,c='k',lw=1.2,alpha=0.4)

    ncol = 6 if aper_num==1 else 5
    leg = ax.legend(loc=4,fontsize=16,ncol=ncol,scatterpoints=0,markerscale=0,fancybox=True,frameon=False,handlelength=0,handletextpad=0)
    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        txt.set_color(hndl.get_facecolor())
        txt.set_fontproperties(FontProperties(size=18,weight=600))
        hndl.set_visible(False)

    fig.savefig(os.path.join(cwd,"mag_limits","mag_lims_r%i.png"%(aper_num+1)))

if __name__ == '__main__':

    cwd = "/data/highzgal/PUBLICACCESS/SPLASH/PROCESS"

    # mk_mag_limit_map(aper_num=0)
    # mk_mag_limit_map(aper_num=1)
    # mk_mag_limit_map(aper_num=2)

    # mk_irac_mag_limit_map(aper_num=1)

    # rebin_mag_limit_maps(aper_num=0)
    # rebin_mag_limit_maps(aper_num=1)
    # rebin_mag_limit_maps(aper_num=2)

    # mk_mag_limit_plot(aper_num=0)
    # mk_mag_limit_plot(aper_num=1)
    # mk_mag_limit_plot(aper_num=2)

    # mk_mag_lim_hists(aper_num=0)
    # mk_mag_lim_hists(aper_num=1)
    # mk_mag_lim_hists(aper_num=2)

    # mk_pretty_plot(aper_num=0)
    mk_pretty_plot(aper_num=1)
    mk_pretty_plot(aper_num=2)

    plt.show()

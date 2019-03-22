import os, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import astropy.io.fits as fitsio
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import make_axes_locatable

import useful

def get_orig_fnames():

    orig_fnames = {}

    for instr in useful.instr_used_list[:-1]:
        orig_fnames[instr] = {}

        for filt in useful.filters[instr]:
        
            orig_fnames[instr][filt] = {}

            img_name = os.path.join("/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/scamp/%s/premosaic_%s_%s.img.fits"%(instr,instr,filt))
            wht_name = os.path.join("/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/scamp/%s/premosaic_%s_%s.wht.fits"%(instr,instr,filt))
        
            if os.path.isfile(img_name) and os.path.isfile(wht_name):
                orig_fnames[instr][filt]["img"] = img_name
                orig_fnames[instr][filt]["wht"] = wht_name
            else:
                print "No input file found for %s:%s"%(instr,filt)

    return orig_fnames

def get_DR_fnames():

    DR_fnames = {}

    for instr in useful.instr_used_list[:-1]:
        DR_fnames[instr] = {}

        for filt in useful.filters[instr]:
        
            DR_fnames[instr][filt] = {}

            if "video" == instr:
                _filt = filt.capitalize()
                img_name = os.path.join("/data/highzgal/PUBLICACCESS/SPLASH/DATA/VIDEO/xmm1/images/xmm1_%s_maxseeing0p90_2016-04-14.fits"%_filt)
                wht_name = os.path.join("/data/highzgal/PUBLICACCESS/SPLASH/DATA/VIDEO/xmm1/images/xmm1_%s_maxseeing0p90_2016-04-14_conf.fits"%_filt)
            elif "uds" == instr:
                _filt = filt.upper()
                img_name = os.path.join("/data/highzgal/PUBLICACCESS/SPLASH/DATA/UDS/UDS-DR11-%s.mef.fits"%_filt)
                wht_name = os.path.join("/data/highzgal/PUBLICACCESS/SPLASH/DATA/UDS/UDS-DR11-%s.weight.mef.fits"%_filt)
            elif "cfhtls" == instr:
                img_name = os.path.join("/data/highzgal/PUBLICACCESS/SPLASH/DATA/CFHT/Tiles/CFHTLS_W_%s_021800-050800_T0007_MEDIAN.fits"%filt)
                wht_name = os.path.join("/data/highzgal/PUBLICACCESS/SPLASH/DATA/CFHT/Tiles/CFHTLS_W_%s_021800-050800_T0007_MEDIAN_weight.fits"%filt)
            elif "supcam" == instr:
                __filt = filt.upper() if filt in ['b','v','r'] else filt
                img_name = os.path.join("/data/highzgal/PUBLICACCESS/SPLASH/DATA//SupCam/Tiles/sxds%s1c_dr1.fits"%__filt)
                wht_name = os.path.join("/data/highzgal/PUBLICACCESS/SPLASH/MOSAICS/supcam/tile_supcam_%s_1.wht.fits"%filt)
            elif "cfht" == instr:
                img_name = os.path.join("/data/highzgal/PUBLICACCESS/SPLASH/DATA/CFHT/HSC-UD-SXDS-U-ver1.nanfix.fits")
                wht_name = os.path.join("/data/highzgal/PUBLICACCESS/SPLASH/DATA/CFHT/HSC-UD-SXDS-U-ver1.weight.fits")
            else:
                img_name = None
                wht_name = None
        
            if (img_name and wht_name) and (os.path.isfile(img_name) and os.path.isfile(wht_name)):
                DR_fnames[instr][filt]["img"] = img_name
                DR_fnames[instr][filt]["wht"] = wht_name
            else:
                print "No input file found for %s:%s"%(instr,filt)

    return DR_fnames

def setup_calls(instr,filt):

    calls = []

    try:
        
        zp = useful.orig_zp[instr][filt] if instr!='supcam' else useful.orig_zp[instr][filt][1]
        inp_name = DR_fnames[instr][filt]["img"]
        wht_name = DR_fnames[instr][filt]["wht"]
        cat_name = "catalog_DR_%s_%s.fits"%(instr,filt)

        pixscale = proj_plane_pixel_scales(WCS(fitsio.getheader(inp_name)))[0] * 3600.
        phot_aper = "-PHOT_APERTURES %s"%(",".join(map(lambda x:"%.2f"%x,(np.arange(5)+1)/pixscale)))

        call_DR   = "sextractor %s -c config/config_test.sex -PARAMETERS_NAME config/param_test.sex "\
                    "-CATALOG_NAME %s -CATALOG_TYPE FITS_1.0 -WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE %s " \
                    "-CHECKIMAGE_TYPE NONE -MAG_ZEROPOINT %.4f %s" % (inp_name,cat_name,wht_name,zp,phot_aper)
        calls.append(call_DR)

    except KeyError: pass

    zp = useful.orig_zp[instr][filt] if instr!='supcam' else useful.orig_zp[instr][filt][1]
    inp_name = orig_fnames[instr][filt]["img"]
    wht_name = orig_fnames[instr][filt]["wht"]
    cat_name = "catalog_orig_%s_%s.fits"%(instr,filt)

    pixscale = proj_plane_pixel_scales(WCS(fitsio.getheader(inp_name)))[0] * 3600.
    phot_aper = "-PHOT_APERTURES %s"%(",".join(map(lambda x:"%.2f"%x,(np.arange(5)+1)/pixscale)))

    call_orig = "sextractor %s -c config/config_test.sex -PARAMETERS_NAME config/param_test.sex "\
                "-CATALOG_NAME %s -CATALOG_TYPE FITS_1.0 -WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE %s " \
                "-CHECKIMAGE_TYPE NONE -MAG_ZEROPOINT %.4f %s" % (inp_name,cat_name,wht_name,zp,phot_aper)
    calls.append(call_orig)

    zp = useful.zp
    inp_name  = os.path.join(cwd,os.pardir,"data/orig/mosaic_%s_%s.img.fits"%(instr,filt))
    wht_name  = os.path.join(cwd,os.pardir,"data/orig/mosaic_%s_%s.wht.fits"%(instr,filt))
    cat_name  = "catalog_swrp_%s_%s.fits"%(instr,filt)
    call_swrp = "sextractor %s -c config/config_test.sex -PARAMETERS_NAME config/param_test.sex "\
                "-CATALOG_NAME %s -CATALOG_TYPE FITS_1.0 -WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE %s " \
                "-CHECKIMAGE_TYPE NONE -MAG_ZEROPOINT %.4f" % (inp_name,cat_name,wht_name,zp)
    calls.append(call_swrp)

    zp = useful.zp
    inp_name  = os.path.join(cwd,os.pardir,"data/conv/mosaic_conv_%s_%s.img.fits"%(instr,filt))
    wht_name  = os.path.join(cwd,os.pardir,"data/conv/mosaic_conv_%s_%s.wht.fits"%(instr,filt))
    cat_name  = "catalog_psfh_%s_%s.fits"%(instr,filt)
    call_psfh = "sextractor %s -c config/config_test.sex -PARAMETERS_NAME config/param_test.sex "\
                "-CATALOG_NAME %s -CATALOG_TYPE FITS_1.0 -WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE %s " \
                "-CHECKIMAGE_TYPE NONE -MAG_ZEROPOINT %.4f" % (inp_name,cat_name,wht_name,zp)
    calls.append(call_psfh)
    
    return calls

def cb_func(x):
    print x

def run_sex():

    calls = []
    for instr in useful.instr_used_list[:-1]:
        for filt in useful.filters[instr]:
            _calls = setup_calls(instr=instr,filt=filt)
            calls.extend(_calls)

    async_run = useful.AsyncFactory(useful.run, cb_func, nproc=10)
    for call in calls: async_run.call(call=call,cwd=cwd,verbose=False)
    async_run.wait()

def match_cats(fname):

    print fname
    start = time.time()

    try:
        cat_DR = fitsio.getdata("catalog_DR_%s.fits"%fname)
    except IOError:
        cat_DR = None
    cat_orig = fitsio.getdata("catalog_orig_%s.fits"%fname)
    cat_swrp = fitsio.getdata("catalog_swrp_%s.fits"%fname)
    cat_psfh = fitsio.getdata("catalog_psfh_%s.fits"%fname)

    if cat_DR is not None:

        m1,m2,d12 = useful.match_ra_dec(cat_DR['X_WORLD'],cat_DR['Y_WORLD'],cat_orig['X_WORLD'],cat_orig['Y_WORLD'])
        cond = (m2 != len(cat_orig))
        m1, m2 = m1[cond], m2[cond]
        
        cat_DR   = cat_DR[m1]
        cat_orig = cat_orig[m2]

    m1,m2,d12 = useful.match_ra_dec(cat_orig['X_WORLD'],cat_orig['Y_WORLD'],cat_swrp['X_WORLD'],cat_swrp['Y_WORLD'])
    cond = (m2 != len(cat_swrp))
    m1, m2 = m1[cond], m2[cond]
    
    if cat_DR is not None: cat_DR   = cat_DR[m1]
    cat_orig = cat_orig[m1]
    cat_swrp = cat_swrp[m2]

    m1,m3,d12 = useful.match_ra_dec(cat_orig['X_WORLD'],cat_orig['Y_WORLD'],cat_psfh['X_WORLD'],cat_psfh['Y_WORLD'])
    cond = (m3 != len(cat_psfh))
    m1, m3 = m1[cond], m3[cond]

    if cat_DR is not None: cat_DR   = cat_DR[m1]
    cat_orig = cat_orig[m1]
    cat_swrp = cat_swrp[m1]
    cat_psfh = cat_psfh[m3]

    try: print fname, len(cat_DR), len(cat_orig), len(cat_swrp), len(cat_psfh)
    except: print fname, len(cat_orig), len(cat_swrp), len(cat_psfh)

    if cat_DR is not None: fitsio.writeto("catalog_DR_%s.matched.fits"%fname,cat_DR,overwrite=True)
    fitsio.writeto("catalog_orig_%s.matched.fits"%fname,cat_orig,overwrite=True)
    fitsio.writeto("catalog_swrp_%s.matched.fits"%fname,cat_swrp,overwrite=True)
    fitsio.writeto("catalog_psfh_%s.matched.fits"%fname,cat_psfh,overwrite=True)

    return (fname, time.time() - start)

def run_matching():

    fnames = []
    for instr in useful.instr_used_list[:-1]:
        for filt in useful.filters[instr]:
            fnames.append("%s_%s"%(instr,filt))

    async_run = useful.AsyncFactory(match_cats, cb_func, nproc=10)
    for fname in fnames: async_run.call(fname=fname)
    async_run.wait()

def plot_hist(ax,xdata,ydata,binsx,binsy):

    bincx = 0.5*(binsx[1:] + binsx[:-1])
    bincy = 0.5*(binsy[1:] + binsy[:-1])
    gridy, gridx = np.meshgrid(bincy,bincx)
    hist2d = np.histogram2d(xdata,ydata,bins=[binsx,binsy])[0]
    hist2d = np.ma.masked_array(hist2d,mask=hist2d==0)
    hist2d = np.ma.log10(hist2d)
    ax.pcolormesh(gridx,gridy,hist2d,cmap=plt.cm.inferno)

def draw_plot(fname,axes,magtype):

    bins_M   = np.arange(0,40,0.1)
    bins_dM  = np.arange(-5,5,0.01)
    bins_SN  = 10**np.arange(-2,7,0.02)
    bins_dSN = 10**np.arange(-2,2,0.01)

    ax1,ax2,ax3,ax4,ax5,ax6 = axes

    try: cat_DR   = fitsio.getdata("catalog_DR_%s.matched.fits"%fname)
    except IOError: cat_DR = None
    cat_orig = fitsio.getdata("catalog_orig_%s.matched.fits"%fname)
    cat_swrp = fitsio.getdata("catalog_swrp_%s.matched.fits"%fname)
    cat_psfh = fitsio.getdata("catalog_psfh_%s.matched.fits"%fname)

    if magtype=="auto":
        mag_orig = cat_orig["MAG_AUTO"]
        mag_swrp = cat_swrp["MAG_AUTO"]
        mag_psfh = cat_psfh["MAG_AUTO"]

        sn_orig = cat_orig["FLUX_AUTO"] / cat_orig["FLUXERR_AUTO"]
        sn_swrp = cat_swrp["FLUX_AUTO"] / cat_swrp["FLUXERR_AUTO"]
        sn_psfh = cat_psfh["FLUX_AUTO"] / cat_psfh["FLUXERR_AUTO"]

    else:
        mag_orig = cat_orig["MAG_APER"][:,2]
        mag_swrp = cat_swrp["MAG_APER"][:,2]
        mag_psfh = cat_psfh["MAG_APER"][:,2]

        sn_orig = cat_orig["FLUX_APER"][:,2] / cat_orig["FLUXERR_APER"][:,2]
        sn_swrp = cat_swrp["FLUX_APER"][:,2] / cat_swrp["FLUXERR_APER"][:,2]
        sn_psfh = cat_psfh["FLUX_APER"][:,2] / cat_psfh["FLUXERR_APER"][:,2]

    if cat_DR is not None:
        
        if magtype=='auto':
            mag_DR = cat_DR["MAG_AUTO"]
            sn_DR  = cat_DR["FLUX_AUTO"] / cat_DR["FLUXERR_AUTO"]
        elif magtype=='aper':
            mag_DR = cat_DR["MAG_APER"][:,2]
            sn_DR  = cat_DR["FLUX_APER"][:,2] / cat_DR["FLUXERR_APER"][:,2]
        plot_hist(ax1,mag_DR,mag_DR-mag_orig,binsx=bins_M, binsy=bins_dM )
        plot_hist(ax2, sn_DR, sn_orig/ sn_DR,binsx=bins_SN,binsy=bins_dSN)

    plot_hist(ax3,mag_orig,mag_orig-mag_swrp,binsx=bins_M, binsy=bins_dM )
    plot_hist(ax4, sn_orig, sn_swrp/ sn_orig,binsx=bins_SN,binsy=bins_dSN)
    plot_hist(ax5,mag_swrp,mag_swrp-mag_psfh,binsx=bins_M, binsy=bins_dM )
    plot_hist(ax6, sn_swrp, sn_psfh/ sn_swrp,binsx=bins_SN,binsy=bins_dSN)

    for ax in [ax1,ax3,ax5]:
        ax.axhline(0,c='k',lw=1.2,ls='--')
        ax.set_xlim(11,28)
        ax.set_ylim(-0.8,0.8)

    for ax in [ax2,ax4,ax6]:
        ax.axhline(1,c='k',lw=1.2,ls='--')
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim(1e0,1e5)
        ax.set_ylim(10**-1,10**1)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
    
    ax3.set_ylabel("$\Delta$M")
    ax4.set_ylabel("SN ratio")
    ax6.set_xlabel("SN")
    ax5.set_xlabel('MAG_AUTO')

    _ = [label.set_visible(False) for label in ax1.get_xticklabels()+ax2.get_xticklabels()+ax3.get_xticklabels()+ax4.get_xticklabels()]

def compare_mag_sn(magtype="auto"):

    fnames = [x for x in useful.fnames if "irac" not in x]
    _fnames = np.array_split(fnames,4)
    fig = plt.figure(figsize=(18,12),dpi=75)
    fig.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05,wspace=0.4,hspace=0.23)
    ogs = gridspec.GridSpec(4,len(_fnames[0])) 

    for i,fname in enumerate(fnames):

        print fname
        igs = gridspec.GridSpecFromSubplotSpec(3,2,subplot_spec=ogs[i],wspace=0,hspace=0)
        ax1 = fig.add_subplot(igs[0,0])
        ax2 = fig.add_subplot(igs[0,1])
        ax3 = fig.add_subplot(igs[1,0],sharex=ax1,sharey=ax1)
        ax4 = fig.add_subplot(igs[1,1],sharex=ax2,sharey=ax2)
        ax5 = fig.add_subplot(igs[2,0],sharex=ax1,sharey=ax1)
        ax6 = fig.add_subplot(igs[2,1],sharex=ax2,sharey=ax2)
        axes = ax1,ax2,ax3,ax4,ax5,ax6

        fig2,axes2 = plt.subplots(3,2,figsize=(15,12),dpi=75)
        fig2.subplots_adjust(left=0.07,right=0.93,bottom=0.05,top=0.92,wspace=0.1,hspace=0.05)

        draw_plot(fname=fname,axes=axes,magtype=magtype)
        draw_plot(fname=fname,axes=axes2.flatten(),magtype=magtype)

        ax2.text(0.96,0.96,fname,fontsize=14,fontweight=600,va='top',ha='left',transform=ax1.transAxes)
        fig2.suptitle(fname,fontsize=16,fontweight=600)

        fig2.savefig("plots/compare_%s_%s.png"%(magtype,fname))
        plt.close(fig2)

    fig.savefig("plots/compare_%s.png"%magtype)

def match_video_cats(fname):

    print fname
    start = time.time()

    cat_DR = fitsio.getdata("catalog_DR_%s.fits"%fname)
    video_catalog = fitsio.getdata('/data/highzgal/PUBLICACCESS/SPLASH/DATA/VIDEO/xmm1/cats/VIDEO-xmm1_2016-04-14_fullcat.fits')
    video_catalog_errfix = fitsio.getdata('/data/highzgal/PUBLICACCESS/SPLASH/DATA/VIDEO/xmm1/cats/VIDEO-xmm1_2016-04-14_fullcat_errfix.fits')

    m1,m2,d12 = useful.match_ra_dec(cat_DR['X_WORLD'],cat_DR['Y_WORLD'],video_catalog['ALPHA_J2000'],video_catalog['DELTA_J2000'])
    cond = (m2 != len(video_catalog))
    m1, m2 = m1[cond], m2[cond]

    fitsio.writeto("catalog_DR_%s.video_matched.fits"%fname,cat_DR[m1],overwrite=True)
    fitsio.writeto("catalog_%s.video_matched.fits"%fname,video_catalog[m2],overwrite=True)
    fitsio.writeto("catalog_%s_errfix.video_matched.fits"%fname,video_catalog_errfix[m2],overwrite=True)

    return (fname, time.time() - start)

def run_video_matching():

    fnames = ["video_z","video_y","video_j","video_h","video_ks"]
    async_run = useful.AsyncFactory(match_video_cats, cb_func, nproc=5)
    for fname in fnames: async_run.call(fname=fname)
    async_run.wait()

def compare_video():

    fnames = ["video_z","video_y","video_j","video_h","video_ks"]
    _fnames = np.array_split(fnames,2)
    fig = plt.figure(figsize=(20,10),dpi=75)
    fig.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.08,wspace=0.4,hspace=0.23)
    ogs = gridspec.GridSpec(2,len(_fnames[0])) 

    for i,fname in enumerate(fnames):

        print fname
        igs = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=ogs[i],wspace=0,hspace=0)
        ax1 = fig.add_subplot(igs[0,0])
        ax2 = fig.add_subplot(igs[0,1])
        ax3 = fig.add_subplot(igs[1,0],sharex=ax1,sharey=ax1)
        ax4 = fig.add_subplot(igs[1,1],sharex=ax2,sharey=ax2)

        bins_M   = np.arange(0,40,0.1)
        bins_dM  = np.arange(-5,5,0.01)
        bins_SN  = 10**np.arange(-2,7,0.02)
        bins_dSN = 10**np.arange(-2,2,0.01)

        cat_DR   = fitsio.getdata("catalog_DR_%s.video_matched.fits"%fname)
        cat_video = fitsio.getdata("catalog_%s.video_matched.fits"%fname)
        cat_video_errfix = fitsio.getdata("catalog_%s_errfix.video_matched.fits"%fname)

        instr,filt = fname.split('_')
        mag_DR = cat_DR["MAG_AUTO"]
        mag_video = cat_video["%s_MAG_AUTO"%filt[0].upper()]
        mag_video_errfix = cat_video_errfix["%s_MAG_AUTO"%filt[0].upper()]

        cond = (cat_video["%s_MAGERR_AUTO"%filt[0].upper()] < 10) & (cat_video_errfix["%s_MAGERR_AUTO"%filt[0].upper()] < 10)
        sn_DR = (cat_DR["FLUX_AUTO"] / cat_DR["FLUXERR_AUTO"])[cond]
        sn_video = 1./(10**(cat_video["%s_MAGERR_AUTO"%filt[0].upper()][cond]/2.5) - 1)
        sn_video_errfix = 1./(10**(cat_video_errfix["%s_MAGERR_AUTO"%filt[0].upper()][cond]/2.5) - 1)

        plot_hist(ax1,mag_DR,mag_DR-mag_video,binsx=bins_M, binsy=bins_dM )
        plot_hist(ax2, sn_DR, sn_video/ sn_DR,binsx=bins_SN,binsy=bins_dSN)

        plot_hist(ax3,mag_DR,mag_DR-mag_video_errfix,binsx=bins_M, binsy=bins_dM )
        plot_hist(ax4, sn_DR, sn_video_errfix/ sn_DR,binsx=bins_SN,binsy=bins_dSN)

        ax1.text(0.96,0.96,fname,fontsize=14,fontweight=600,va='top',ha='left',transform=ax1.transAxes)

        for ax in [ax1,ax3]:
            ax.axhline(0,c='k',lw=1.2,ls='--')
            ax.set_xlim(11,27)
            ax.set_ylim(-1.4,1.4)

        for ax in [ax2,ax4]:
            ax.axhline(1,c='k',lw=1.2,ls='--')
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_xlim(1e0,1e5)
            ax.set_ylim(10**-2,10**2)
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')
        
        ax3.set_ylabel("$\Delta$M")
        ax4.set_ylabel("SN ratio")
        ax4.set_xlabel("SN")
        ax3.set_xlabel('MAG_AUTO')

        _ = [label.set_visible(False) for label in ax1.get_xticklabels()+ax2.get_xticklabels()]

    fig.savefig("plots/compare_video_auto.png")

def calc_fscale(zp0,zp1=23.93):
    """
    -2.5*log(f1) + zp1 = -2.5*log(f0) + zp0
    f1/f0 = 10**((zp1 - zp0) / 2.5)
    """
    fscale = 10**((zp1 - zp0) / 2.5)
    return fscale

def compare_pixscale():

    fnames = [x for x in useful.fnames if "irac" not in x]
    zorder = np.linspace(1,10,len(fnames))[::-1]
    
    fig,ax = plt.subplots(1,1,figsize=(10,10),dpi=75)
    fig.subplots_adjust(left=0.1,right=0.98,top=0.98,bottom=0.07)
    divider = make_axes_locatable(ax)
    dax = divider.append_axes("bottom", size="40%", pad=0.15)

    for fname,zo in zip(fnames,zorder):

        instr,filt = fname.split("_")
        fcolor = useful.fcolor_dict[instr][filt]
        inp_name = orig_fnames[instr][filt]["img"]
        pixscale = proj_plane_pixel_scales(WCS(fitsio.getheader(inp_name)))[0] * 3600.

        cat_orig = fitsio.getdata("catalog_orig_%s.matched.fits"%fname)
        cat_swrp = fitsio.getdata("catalog_swrp_%s.matched.fits"%fname)

        orig_zp = useful.orig_zp[instr][filt] if "supcam" not in fname else useful.orig_zp[instr][filt][1]
        cat_orig["FLUX_APER"]    = cat_orig["FLUX_APER"]    * calc_fscale(orig_zp)
        cat_orig["FLUXERR_APER"] = cat_orig["FLUXERR_APER"] * calc_fscale(orig_zp)

        sn_orig = cat_orig["FLUX_APER"][:,2] / cat_orig["FLUXERR_APER"][:,2]
        sn_swrp = cat_swrp["FLUX_APER"][:,2] / cat_swrp["FLUXERR_APER"][:,2]

        cond = (sn_orig >= 25.) & (sn_swrp >= 25.)
        sn_orig, sn_swrp = sn_orig[cond], sn_swrp[cond]

        ferr_orig = cat_orig["FLUXERR_APER"][:,2][cond]
        ferr_swrp = cat_swrp["FLUXERR_APER"][:,2][cond]

        ferr_med = np.median(ferr_swrp/ferr_orig)
        ferr_CIs = np.percentile(ferr_swrp/ferr_orig,[16,84])
        ferr_err = np.abs(ferr_CIs - ferr_med)[:,np.newaxis]

        sn_med = np.median(sn_swrp/sn_orig)
        sn_CIs = np.percentile(sn_swrp/sn_orig,[16,84])
        sn_err = np.abs(sn_CIs - sn_med)[:,np.newaxis]

        ax.scatter(pixscale,ferr_med,marker='o',s=75,facecolor=fcolor,edgecolor='none',label=fname.replace("_",":"),zorder=zo)
        ax.errorbar(pixscale,ferr_med,yerr=ferr_err,marker='',markersize=0,color=fcolor,capsize=5,zorder=zo)
        
        dax.scatter(pixscale,sn_med,marker='o',s=75,facecolor=fcolor,edgecolor='none',zorder=zo)
        dax.errorbar(pixscale,sn_med,yerr=sn_err,marker='',markersize=0,color=fcolor,capsize=5,zorder=zo)

        print fname, ferr_med, ferr_CIs, sn_med, sn_CIs

    ax.axhline(1,c='k',lw=1.2,ls='--')
    dax.axhline(1,c='k',lw=1.2,ls='--')
    ax.axvline(0.15,c='k',lw=1.2)
    dax.axvline(0.15,c='k',lw=1.2,label='SWARP\'d Pixscale')

    xx = np.arange(0.1,0.3,0.01)
    yy = xx/0.15
    ax.plot(xx,1./yy,c='k',ls='-')
    dax.plot(xx,yy,c='k',ls='-')

    ax.set_ylim(0.6,1.6)
    ax.set_xlim(0.11,0.23)
    dax.set_ylim(0.75,1.55)
    dax.set_xlim(0.11,0.23)

    ax.set_ylabel("$\sigma_{f,SWARP} / \sigma_{f,Orig}$",fontsize=20)
    dax.set_ylabel("$SN_{SWARP} / SN_{Orig}$",fontsize=20)
    dax.set_xlabel("Original Pixscale [\"/px]",fontsize=20)

    _ = [label.set_visible(False) for label in ax.get_xticklabels()]
    _ = [label.set_fontsize(16) for label in ax.get_xticklabels()+ax.get_yticklabels()+dax.get_xticklabels()+dax.get_yticklabels()]
    
    dax.legend(loc=2,fontsize=16,fancybox=True,frameon=False)
    leg = ax.legend(loc="upper center",fontsize=14,ncol=4,fancybox=True,frameon=False)
    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        txt.set_color(hndl.get_facecolor()[0])
        txt.set_fontproperties(FontProperties(size=14,weight=600))
        hndl.set_visible(False)

    fig.savefig("errors_swarp.png")

def compare_PSF():

    fnames = [x for x in useful.fnames if "irac" not in x]
    psfex_pars = useful.psfex_moffat_pars(basis_type="orig")
    
    fig,ax = plt.subplots(1,1,figsize=(10,6),dpi=75,tight_layout=True)

    for fname in fnames:

        instr,filt = fname.split("_")
        fcolor = useful.fcolor_dict[instr][filt]

        cat_swrp = fitsio.getdata("catalog_swrp_%s.matched.fits"%fname)
        cat_psfh = fitsio.getdata("catalog_psfh_%s.matched.fits"%fname)

        sn_swrp = cat_swrp["FLUX_AUTO"] / cat_swrp["FLUXERR_AUTO"]
        sn_psfh = cat_psfh["FLUX_AUTO"] / cat_psfh["FLUXERR_AUTO"]

        psf = psfex_pars[instr][filt][0] if instr!='cfhtls' else psfex_pars[instr][filt][5][0]
        ax.scatter(psf,np.median(sn_psfh/sn_swrp),marker='o',s=75,facecolor=fcolor,edgecolor='none',label=fname.replace("_",":"))

    ax.axhline(1,c='k',lw=1.2,ls='--')
    ax.axvline(0.7,c='k',lw=1.2,ls=':')
    ax.set_xlabel("Original PSF FWHM",fontsize=20)
    ax.set_ylabel("$SN_{PSFEx} / SN_{SWARP}$",fontsize=20)
    ax.set_ylim(0.7,2.2)
    ax.set_xlim(0.4,1.1)

    leg = ax.legend(loc=2,fontsize=14,ncol=4,fancybox=True,frameon=False)
    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        txt.set_color(hndl.get_facecolor()[0])
        txt.set_fontproperties(FontProperties(size=14,weight=600))
        hndl.set_visible(False)

    _ = [label.set_fontsize(16) for label in ax.get_xticklabels()+ax.get_yticklabels()]

    fig.savefig("errors_psfex.png")

if __name__ == '__main__':
    
    cwd = "/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/errors_test/"

    DR_fnames = get_DR_fnames()
    orig_fnames = get_orig_fnames()
    
    # run_sex()
    # run_matching()
    # run_video_matching()

    # compare_mag_sn(magtype="auto")
    # compare_mag_sn(magtype="aper")
    # compare_video()
    
    compare_pixscale()
    # compare_PSF()
    
    plt.show()
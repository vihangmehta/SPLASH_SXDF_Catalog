import os, sys, time
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

def mk_rms(inp_name,out_name,factor=1):

    sys.stdout.write("Creating weight from %s to %s ... " % (inp_name,out_name))
    sys.stdout.flush()

    img,hdr = fitsio.getdata(inp_name,header=True)
    cond = np.isfinite(img) & (img>0)
    img[cond] = 1./np.sqrt(img[cond])
    img[cond] = img[cond] * factor
    fitsio.writeto(out_name,data=img,header=hdr,overwrite=True)

    print "done"

def setup_calls(instr,filt):

    calls = []

    zp = useful.orig_zp[instr][filt] if instr!='supcam' else useful.orig_zp[instr][filt][1]
    inp_name = orig_fnames[instr][filt]["img"]
    wht_name = orig_fnames[instr][filt]["wht"]
    rms_name = "rms/orig_%s_%s.rms2.fits"%(instr,filt)
    cat_name = "catalog_rms2_orig_%s_%s.fits"%(instr,filt)
    mk_rms(wht_name,rms_name,factor=2)

    pixscale = proj_plane_pixel_scales(WCS(fitsio.getheader(inp_name)))[0] * 3600.
    phot_aper = "-PHOT_APERTURES %s"%(",".join(map(lambda x:"%.2f"%x,(np.arange(5)+1)/pixscale)))

    call_orig = "sextractor %s -c config/config_test.sex -PARAMETERS_NAME config/param_test.sex "\
                "-CATALOG_NAME %s -CATALOG_TYPE FITS_1.0 -WEIGHT_TYPE MAP_RMS -WEIGHT_IMAGE %s " \
                "-CHECKIMAGE_TYPE NONE -MAG_ZEROPOINT %.4f %s" % (inp_name,cat_name,rms_name,zp,phot_aper)
    calls.append(call_orig)

    zp = useful.zp
    inp_name  = os.path.join(cwd,os.pardir,"data/orig/mosaic_%s_%s.img.fits"%(instr,filt))
    wht_name  = os.path.join(cwd,os.pardir,"data/orig/mosaic_%s_%s.wht.fits"%(instr,filt))
    rms_name  = "rms/mosaic_%s_%s.rms2.fits"%(instr,filt)
    cat_name  = "catalog_rms2_swrp_%s_%s.fits"%(instr,filt)
    mk_rms(wht_name,rms_name,factor=2)
    
    call_swrp = "sextractor %s -c config/config_test.sex -PARAMETERS_NAME config/param_test.sex "\
                "-CATALOG_NAME %s -CATALOG_TYPE FITS_1.0 -WEIGHT_TYPE MAP_RMS -WEIGHT_IMAGE %s " \
                "-CHECKIMAGE_TYPE NONE -MAG_ZEROPOINT %.4f" % (inp_name,cat_name,rms_name,zp)
    calls.append(call_swrp)
    
    return calls

def cb_func(x):
    print x

def run_sex():

    calls = []
    for instr in useful.instr_used_list[:-1]:
        for filt in useful.filters[instr][:1]:
            _calls = setup_calls(instr=instr,filt=filt)
            calls.extend(_calls)

    async_run = useful.AsyncFactory(useful.run, cb_func, nproc=10)
    for call in calls: async_run.call(call=call,cwd=cwd,verbose=False)
    async_run.wait()

def match_cats(fname):

    print fname
    start = time.time()

    cat_orig = fitsio.getdata("catalog_rms2_orig_%s.fits"%fname)
    cat_swrp = fitsio.getdata("catalog_rms2_swrp_%s.fits"%fname)

    m1,m2,d12 = useful.match_ra_dec(cat_orig['X_WORLD'],cat_orig['Y_WORLD'],cat_swrp['X_WORLD'],cat_swrp['Y_WORLD'])
    cond = (m2 != len(cat_swrp))
    m1, m2 = m1[cond], m2[cond]
    
    cat_orig = cat_orig[m1]
    cat_swrp = cat_swrp[m2]

    print fname, len(cat_orig), len(cat_swrp)

    fitsio.writeto("catalog_rms2_orig_%s.matched.fits"%fname,cat_orig,overwrite=True)
    fitsio.writeto("catalog_rms2_swrp_%s.matched.fits"%fname,cat_swrp,overwrite=True)

    return (fname, time.time() - start)

def run_matching():

    fnames = []
    for instr in useful.instr_used_list[:-1]:
        for filt in useful.filters[instr][:1]:
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

    ax1,ax2 = axes

    cat_orig = fitsio.getdata("catalog_rms2_orig_%s.matched.fits"%fname)
    cat_swrp = fitsio.getdata("catalog_rms2_swrp_%s.matched.fits"%fname)

    if magtype=="auto":
        mag_orig = cat_orig["MAG_AUTO"]
        mag_swrp = cat_swrp["MAG_AUTO"]

        sn_orig = cat_orig["FLUX_AUTO"] / cat_orig["FLUXERR_AUTO"]
        sn_swrp = cat_swrp["FLUX_AUTO"] / cat_swrp["FLUXERR_AUTO"]

    else:
        mag_orig = cat_orig["MAG_APER"][:,2]
        mag_swrp = cat_swrp["MAG_APER"][:,2]

        sn_orig = cat_orig["FLUX_APER"][:,2] / cat_orig["FLUXERR_APER"][:,2]
        sn_swrp = cat_swrp["FLUX_APER"][:,2] / cat_swrp["FLUXERR_APER"][:,2]

    plot_hist(ax1,mag_orig,mag_orig-mag_swrp,binsx=bins_M, binsy=bins_dM )
    plot_hist(ax2, sn_orig, sn_swrp/ sn_orig,binsx=bins_SN,binsy=bins_dSN)

    ax1.axhline(0,c='k',lw=1.2,ls='--')
    ax1.set_xlim(11,28)
    ax1.set_ylim(-0.8,0.8)

    ax2.axhline(1,c='k',lw=1.2,ls='--')
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_xlim(1e0,1e5)
    ax2.set_ylim(10**-1,10**1)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    
    ax1.set_ylabel("$\Delta$M")
    ax2.set_ylabel("SN ratio")
    ax1.set_xlabel('MAG_AUTO')
    ax2.set_xlabel("SN")

def compare_mag_sn(magtype="auto"):

    fnames = [x for x in useful.fnames if "irac" not in x]

    _fnames = np.array_split(fnames,4)
    fig = plt.figure(figsize=(18,12),dpi=75)
    fig.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05,wspace=0.4,hspace=0.23)
    ogs = gridspec.GridSpec(4,len(_fnames[0])) 

    for i,fname in enumerate(fnames):

        print fname
        igs = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=ogs[i],wspace=0,hspace=0)
        ax1 = fig.add_subplot(igs[0,0])
        ax2 = fig.add_subplot(igs[0,1])
        axes = ax1,ax2

        draw_plot(fname=fname,axes=axes,magtype=magtype)

    # fig.savefig("plots/compare_%s_rms.png"%magtype)

def calc_fscale(zp0,zp1=23.93):
    """
    -2.5*log(f1) + zp1 = -2.5*log(f0) + zp0
    f1/f0 = 10**((zp1 - zp0) / 2.5)
    """
    fscale = 10**((zp1 - zp0) / 2.5)
    return fscale

def compare_pixscale():

    # fnames = [x for x in useful.fnames if "irac" not in x]
    fnames = []
    for instr in useful.instr_used_list[:-1]:
        for filt in useful.filters[instr][:1]:
            fnames.append("%s_%s"%(instr,filt))

    zorder = np.linspace(1,10,len(fnames))[::-1]
    
    fig,ax = plt.subplots(1,1,figsize=(10,10),dpi=75)
    fig.subplots_adjust(left=0.1,right=0.98,top=0.98,bottom=0.07)
    divider = make_axes_locatable(ax)
    dax = divider.append_axes("bottom", size="40%", pad=0.15)

    for fname,zo in zip(fnames,zorder):

        print fname
        instr,filt = fname.split("_")
        fcolor = useful.fcolor_dict[instr][filt]
        inp_name = orig_fnames[instr][filt]["img"]
        pixscale = proj_plane_pixel_scales(WCS(fitsio.getheader(inp_name)))[0] * 3600.

        cat_orig = fitsio.getdata("catalog_rms2_orig_%s.matched.fits"%fname)
        cat_swrp = fitsio.getdata("catalog_rms2_swrp_%s.matched.fits"%fname)

        orig_zp = useful.orig_zp[instr][filt] if "supcam" not in fname else useful.orig_zp[instr][filt][1]
        cat_orig["FLUX_APER"]    = cat_orig["FLUX_APER"]    * calc_fscale(orig_zp)
        cat_orig["FLUXERR_APER"] = cat_orig["FLUXERR_APER"] * calc_fscale(orig_zp)

        cond = (cat_orig["FLUXERR_APER"][:,2]>0) & (cat_swrp["FLUXERR_APER"][:,2]>0)
        cat_orig = cat_orig[cond]
        cat_swrp = cat_swrp[cond]

        sn_orig = cat_orig["FLUX_APER"][:,2] / cat_orig["FLUXERR_APER"][:,2]
        sn_swrp = cat_swrp["FLUX_APER"][:,2] / cat_swrp["FLUXERR_APER"][:,2]

        cond = (sn_orig >= 25.) & (sn_swrp >= 25.)
        sn_orig, sn_swrp = sn_orig[cond], sn_swrp[cond]

        ferr_orig = cat_orig["FLUXERR_APER"][:,2][cond]
        ferr_swrp = cat_swrp["FLUXERR_APER"][:,2][cond]

        ferr_med = np.nanmedian(ferr_swrp/ferr_orig)
        ferr_CIs = np.nanpercentile(ferr_swrp/ferr_orig,[16,84])
        ferr_err = np.abs(ferr_CIs - ferr_med)[:,np.newaxis]

        sn_med = np.nanmedian(sn_swrp/sn_orig)
        sn_CIs = np.nanpercentile(sn_swrp/sn_orig,[16,84])
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

    fig.savefig("errors_swarp_rms2.png")

def test():

    fname = "hsc_g"

    cat1 = fitsio.getdata("catalog_rms_swrp_%s.fits"%fname)
    cat2 = fitsio.getdata("catalog_rms2_swrp_%s.fits"%fname)

    # cat2 has fewer obj
    m1,m2,d12 = useful.match_ra_dec(cat2['X_WORLD'],cat2['Y_WORLD'],cat1['X_WORLD'],cat1['Y_WORLD'])
    cond = (m2 != len(cat1))
    m1, m2 = m1[cond], m2[cond]
    
    # cat2 = cat2[m1]
    # cat1 = cat1[m2]

    fig,ax = plt.subplots(1,1,figsize=(10,6),dpi=75,tight_layout=True)

    hist,xedges,yedges = np.histogram2d(cat1["MAG_APER"][m2,2],cat2["FLUXERR_APER"][m1,2]/cat1["FLUXERR_APER"][m2,2],bins=[np.arange(10,30,0.05),np.arange(0,5,0.001)])
    cond = (hist>0)
    hist[cond] = np.log10(hist[cond])
    ax.pcolormesh(xedges,yedges,hist.T,cmap=plt.cm.Greys,vmin=0,vmax=np.max(hist)*0.9)

    # ax.scatter(cat1["MAG_APER"][m2,2],cat2["FLUXERR_APER"][m1,2]/cat1["FLUXERR_APER"][m2,2],c='k',s=2,alpha=0.01)
    ax.axhline(2.,c='k',ls='--')

    ax.set_xlabel("MAG_APER")
    ax.set_ylabel("FLUXERR_APER ratio (2 $\\times$ rms / rms)")
    ax.set_xlim(16,29.5)
    ax.set_ylim(1.8,2.2)

    fig.savefig("test.png")

if __name__ == '__main__':
    
    cwd = "/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/errors_test/"

    orig_fnames = get_orig_fnames()
    
    # run_sex()
    # run_matching()

    # compare_mag_sn(magtype="auto")
    # compare_mag_sn(magtype="aper")
    
    # compare_pixscale()

    test()
    
    plt.show()
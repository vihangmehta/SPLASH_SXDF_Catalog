import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fitsio
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import FancyArrowPatch

import useful

errors_swarp = {"hsc":    {"g": 0.907376, "r": 0.906118, "i": 0.902605, "z": 0.900655, "y": 0.899812},
                "supcam": {"b": 0.742146, "v": 0.742253, "r": 0.742243, "i": 0.742282, "z": 0.742273},
                "uds":    {"j": 1.119700, "h": 1.120300, "k": 1.119530},
                "video":  {"z": 0.750970, "y": 0.751166, "j": 0.751554, "h": 0.751377, "ks": 0.751297},
                "cfht":   {"u": 0.837504,},
                "cfhtls": {"u": 0.812688, "g": 0.816215, "r": 0.812607, "i": 0.812421, "z": 0.816554}}

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
                print ("No input file found for %s:%s"%(instr,filt))

    return orig_fnames

def compare_pixscale():

    fnames = [x for x in useful.fnames if "irac" not in x]
    zorder = np.linspace(1,10,len(fnames))[::-1]

    fig,ax = plt.subplots(1,1,figsize=(10,10),dpi=75)
    fig.subplots_adjust(left=0.1,right=0.98,top=0.98,bottom=0.08)
    divider = make_axes_locatable(ax)
    dax = divider.append_axes("bottom", size="40%", pad=0.15)

    for fname,zo in zip(fnames,zorder):

        instr,filt = fname.split("_")
        fcolor = useful.fcolor_dict[instr][filt]
        inp_name = orig_fnames[instr][filt]["img"]
        pixscale = proj_plane_pixel_scales(WCS(fitsio.getheader(inp_name)))[0] * 3600.

        cat_orig = fitsio.getdata("errors_test/catalog_orig_%s.matched.fits"%fname)
        cat_swrp = fitsio.getdata("errors_test/catalog_swrp_%s.matched.fits"%fname)

        orig_zp = useful.orig_zp[instr][filt] if "supcam" not in fname else useful.orig_zp[instr][filt][1]
        cat_orig["FLUX_APER"]    = cat_orig["FLUX_APER"]    * useful.calc_fscale(orig_zp)
        cat_orig["FLUXERR_APER"] = cat_orig["FLUXERR_APER"] * useful.calc_fscale(orig_zp)

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

        ax.scatter(pixscale,ferr_med,marker='o',s=75,facecolor=fcolor,edgecolor='none',label="%s:%s"%(instr if instr!='cfht' else 'musubi',filt),zorder=zo)
        ax.errorbar(pixscale,ferr_med,yerr=ferr_err,marker='',markersize=0,color=fcolor,capsize=5,zorder=zo)

        dax.scatter(pixscale,sn_med,marker='o',s=75,facecolor=fcolor,edgecolor='none',zorder=zo)
        dax.errorbar(pixscale,sn_med,yerr=sn_err,marker='',markersize=0,color=fcolor,capsize=5,zorder=zo)

        print (fname, ferr_med, ferr_CIs, sn_med, sn_CIs)

    ax.axhline(1,c='k',lw=1.,ls='-',alpha=0.6)
    dax.axhline(1,c='k',lw=1.,ls='-',alpha=0.6)
    ax.axvline(0.15,c='k',lw=1.,ls='--')
    dax.axvline(0.15,c='k',lw=1.,ls='--')

    ax.annotate('New pixscale\n(0.15"/px)', xy=(0.15, 0.85), xytext=(0.145, 0.75),
                va='top', ha='right', fontsize=24,
                arrowprops=dict(facecolor='black',arrowstyle="->",connectionstyle="angle3"),)

    # an1 = ax.annotate('$\\left(\\rm\\frac{new \\ pixscale}{orig \\ pixscale}\\right)$', xy=(0.16, 0.93), xytext=(0.165, 0.73),
    #             va='top', ha='left', fontsize=26,
    #             arrowprops=dict(facecolor='black',arrowstyle="->",connectionstyle="angle3,angleA=0,angleB=90"),)

    xx = np.arange(0.1,0.3,0.01)
    yy = xx/0.15
    ax.plot(xx,1./yy,c='k',ls='-')
    dax.plot(xx,yy,c='k',ls='-')

    ax.set_ylim(0.6,1.6)
    ax.set_xlim(0.11,0.23)
    dax.set_ylim(0.75,1.55)
    dax.set_xlim(0.11,0.23)

    ax.set_ylabel("$\sigma_{f,new} / \sigma_{f,orig}$",fontsize=24)
    dax.set_ylabel("$SN_{new} / SN_{orig}$",fontsize=24)
    dax.set_xlabel("Original Pixscale [\"/px]",fontsize=24)

    _ = [label.set_visible(False) for label in ax.get_xticklabels()]
    _ = [label.set_fontsize(16) for label in ax.get_xticklabels()+ax.get_yticklabels()+dax.get_xticklabels()+dax.get_yticklabels()]

    # dax.legend(loc=2,fontsize=16,fancybox=True,frameon=False)
    leg = ax.legend(loc="upper right",fontsize=18,ncol=4,framealpha=0,handlelength=0,handletextpad=0)
    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        txt.set_fontweight(600)
        txt.set_color(hndl.get_facecolor()[0])
        hndl.set_visible(False)

    fig.savefig("errors_test/errors_swarp.png")

if __name__ == '__main__':

    orig_fnames = get_orig_fnames()
    compare_pixscale()
    plt.show()

import astropy.io.fits as fitsio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
from matplotlib.ticker import FuncFormatter
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import OrderedDict

import useful

def get_label_color_dicts():

    cmap = plt.cm.tab10
    label_dict = (("VIPERS",'VIPERS'),
                  ("UDSz"  ,'UDSz'),
                  ("C3R2"  ,'C3R2'),
                  # ("M15"   ,'Morris+15'),
                  ("SUBARU",'Subaru Comp.'),
                  ("XUDS"  ,'XUDS Comp.'))
    label_dict = OrderedDict(label_dict)
    color_dict = OrderedDict(zip(label_dict.keys(),cmap(np.linspace(0,1,10))))
    order_dict = OrderedDict(zip(label_dict.keys(),[1,3,5,6,2]))
    return label_dict, color_dict, order_dict

def get_label_cond(label,zref):

    if "SUBARU"==label: _cond = [i in useful.SUBARU_key.values() for i in zref]
    elif "XUDS"==label: _cond = [i in useful.XUDS_key.values() for i in zref]
    else: _cond = [label in i for i in zref]
    return _cond

def plot_specz(catalog,savename=None):

    label_dict, color_dict, order_dict = get_label_color_dicts()

    fig,ax = plt.subplots(1,1,figsize=(10,8),dpi=75,tight_layout=True)

    print( "%10s : %5s | %4s | %11s | %6s" % ("spec-z","N","z_med","z_range","i_med"))
    bins = np.arange(-1,8,0.1)
    for x in color_dict.keys():
        _cond = get_label_cond(x,catalog['ZSPEC_REF'])
        ax.scatter(None,None,c=color_dict[x],label=label_dict[x])
        ax.scatter(catalog["ZSPEC"][_cond],catalog["MAG_AUTO_hsc_i"][_cond],s=12,color=color_dict[x],alpha=0.5,zorder=order_dict[x])
        print( "%10s : %5i | %4.2f | [%4.2f,%4.2f] | %6.2f" % (x, np.sum(_cond), np.median(catalog["ZSPEC"][_cond]), min(catalog["ZSPEC"][_cond]), max(catalog["ZSPEC"][_cond]), np.median(catalog["MAG_AUTO_hsc_i"][_cond])))
        print( "%10s : %5i | %4.2f | [%4.2f,%4.2f] | %6.2f" % (x, np.sum(_cond), np.median(catalog["ZSPEC"][_cond]), np.percentile(catalog["ZSPEC"][_cond],5), np.percentile(catalog["ZSPEC"][_cond],95), np.median(catalog["MAG_AUTO_hsc_i"][_cond])))

    _ = [tick.set_fontsize(16) for tick in ax.get_xticklabels()+ax.get_yticklabels()]

    ax.set_ylabel("$i$-band AUTO mag [AB]",fontsize=24)
    ax.set_xlabel("spec-$z$",fontsize=24)
    ax.set_ylim(28.5,14.5)

    leg = ax.legend(loc="best",scatterpoints=1,markerscale=0,fancybox=True,frameon=False,handlelength=0,handletextpad=0)
    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        txt.set_color(hndl.get_facecolor()[0])
        txt.set_fontproperties(FontProperties(size=18,weight=600))
        hndl.set_visible(False)

    if savename: fig.savefig(savename)

def calc_sig_NMAD(zphot, zspec):

    diff = zphot - zspec
    frac = diff/(1+zspec)
    outliers = len(frac[abs(frac) > 0.15])
    residual = (diff - np.median(diff)) / (1. + zspec)
    sig_NMAD = 1.48 * np.median(abs(residual))
    return sig_NMAD, outliers / float(len(frac))

def compare_z(catalog, lph_output, apersize, chi_cut=None, z_ml=True, savename=None):

    assert np.all(catalog['ID'] == lph_output['ID'])

    # lph_output["Z_BEST"][lph_output["CHI_BEST"] > lph_output["CHI_STAR"]] = 0
    # lph_output["Z_BEST"][lph_output["CHI_BEST"] > lph_output["CHI_QSO"]] = lph_output["Z_QSO"][lph_output["CHI_BEST"] > lph_output["CHI_QSO"]]

    zspec = catalog["ZSPEC"]
    if z_ml: zphot = lph_output["Z_ML"]
    else:    zphot = lph_output["Z_BEST"]

    cond = (zphot!=-99.)
    zspec = zspec[cond]
    zphot = zphot[cond]
    catalog = catalog[cond]
    lph_output = lph_output[cond]

    def setup_fig():

        gs = gridspec.GridSpec(2, 1, height_ratios=[4,1])

        fig = plt.figure(figsize=(10.2,12.5),dpi=75)
        fig.subplots_adjust(left=0.1,right=0.98,top=0.98,bottom=0.08,hspace=0,wspace=0)
        ax  = fig.add_subplot(gs[0])
        dax = fig.add_subplot(gs[1],sharex=ax)

        zz = np.arange(0,100,0.1)
        ax.plot(zz, zz, c='k', ls='-', lw=1.0, alpha=1.0)
        ax.plot(zz,  0.15*(1+zz)+zz,lw=0.75,ls='--',c='k',alpha=0.5)
        ax.plot(zz, -0.15*(1+zz)+zz,lw=0.75,ls='--',c='k',alpha=0.5)
        ax.set_aspect(1.)
        ax.set_xlim(-0.05,6.5)
        ax.set_ylim(-0.05,6.5)
        ax.set_ylabel(r'$z_{phot}$',fontsize=28)
        _ = [label.set_visible(False) for label in ax.get_xticklabels()]
        _ = [label.set_fontsize(14) for label in ax.get_yticklabels()]

        dax.axhline(0, c='k', ls='-', lw=1.0, alpha=1.0)
        dax.axhline(-0.15,lw=0.75,ls=':',c='k', alpha=0.5)
        dax.axhline(0.15, lw=0.75,ls=':',c='k', alpha=0.5)
        dax.set_xlim(-0.05,6.5)
        dax.set_ylim(-0.45,0.45)
        dax.set_yticks([-0.4,-0.2,0.0,0.2,0.4])
        dax.set_xlabel(r'$z_{spec}$',fontsize=28)
        dax.set_ylabel(r'$\Delta z/(1+z)$',fontsize=18)
        _ = [label.set_fontsize(14) for label in dax.get_xticklabels()+dax.get_yticklabels()]

        inset = fig.add_axes([0.5,0.87,0.2,0.1])
        _ = [label.set_fontsize(12) for label in inset.get_xticklabels()+inset.get_yticklabels()]
        inset.axvline(0,lw=0.75,ls='-',c='k')
        inset.axvline(-0.15,lw=0.75,ls=':',c='k')
        inset.axvline(0.15, lw=0.75,ls=':',c='k')
        inset.set_xlim(-0.45,0.45)
        inset.set_xlabel(r'$\Delta z/(1+z)$',fontsize=16)
        inset.set_ylabel('N',fontsize=16)

        return fig, ax, dax, inset

    def plot_points(axis, daxis, zspec, zphot, fc, ec, s, alpha, lw, zorder):

        diff = zphot - zspec
        frac = diff/(1+zspec)
        for i,_z in enumerate(np.unique(zorder)):
            cond = (zorder==_z)
            _fc = fc[cond] if isinstance(fc,np.ndarray) else fc
            _ec = ec[cond] if isinstance(ec,np.ndarray) else ec
            axis.scatter(zspec[cond], zphot[cond], facecolor=_fc, edgecolor=_ec, marker='o', lw=lw, s=s, alpha=alpha, zorder=_z)
            daxis.scatter(zspec[cond], frac[cond], facecolor=_fc, edgecolor=_ec, marker='o', lw=lw, s=s, alpha=alpha, zorder=_z)
        daxis.axhline(np.median(frac), c='k', ls='--', lw=1.5)

    def plot_hist(daxis, zspec, zphot, dbin, color, ls, alpha, label=None):

        diff = zphot - zspec
        frac = diff/(1+zspec)
        bins = np.arange(-2.5-dbin/2.,2.5+dbin/2.+1e-6,dbin)
        binc = 0.5*(bins[1:] + bins[:-1])
        hist = np.histogram(frac, bins=bins)[0]
        # hist = hist/float(np.max(hist))
        daxis.plot(binc, hist, color=color, ls=ls, lw=1.25, alpha=1.0, label=label)
        daxis.set_ylim(0, max(daxis.get_xlim()[1],1.1*max(hist)))
        daxis.axvline(np.median(frac), c='k', ls='--', lw=1.5)

    fig, ax, dax1, dax2 = setup_fig()

    label_dict, color_dict, order_dict = get_label_color_dicts()
    for x in color_dict.keys():
        ax.scatter(-99, -99, facecolor=color_dict[x], edgecolor=color_dict[x], label=label_dict[x])

    colors = np.zeros((len(zspec),4))
    zorder = np.zeros(len(zspec))

    if chi_cut:

        # ax.scatter(-99, -99, facecolor='k'   , edgecolor='k', marker='o', s=50, alpha=1.0, lw=1.0, label=r'$\chi^2$ < %i'%chi_cut)
        # ax.scatter(-99, -99, facecolor='none', edgecolor='k', marker='o', s=50, alpha=1.0, lw=1.0, label=r'No $\chi^2$ Cuts')

        cond_chi  = (lph_output['CHI_BEST'] <= chi_cut)

        print()
        print( "%10s : %13s | %13s | %11s" % ("spec-z","N","sigma","out%"))
        print( "".join(['-']*55))
        for x in color_dict.keys():

            _cond = get_label_cond(x,catalog['ZSPEC_REF'])
            colors[_cond] = color_dict[x]
            zorder[_cond] = order_dict[x]

            sig_NMAD1, outliers1 = calc_sig_NMAD(zphot[_cond],zspec[_cond])
            sig_NMAD2, outliers2 = calc_sig_NMAD(zphot[_cond&cond_chi],zspec[_cond&cond_chi])
            print( "%10s : %5i (%5i) | %5.3f (%5.3f) | %4.1f (%4.1f)" % (x, np.sum(_cond & cond_chi), np.sum(_cond), sig_NMAD2, sig_NMAD1, outliers2*100, outliers1*100))

        sig_NMAD1, outliers1 = calc_sig_NMAD(zphot,zspec)
        sig_NMAD2, outliers2 = calc_sig_NMAD(zphot[cond_chi],zspec[cond_chi])
        print( "".join(['-']*55))
        print( "%10s: %5i (%5i) | %5.3f (%5.3f) | %4.1f (%4.1f)" % ("all", np.sum(cond_chi), len(zphot), sig_NMAD2, sig_NMAD1, outliers2*100, outliers1*100))
        print( "".join(['-']*55))
        print()

        plot_points(ax, dax1, zspec[~cond_chi], zphot[~cond_chi], fc='none', ec=colors[~cond_chi], s=12, lw=0.3, alpha=0.6, zorder=zorder[~cond_chi])
        plot_hist(dax2, zspec[~cond_chi], zphot[~cond_chi], dbin=0.01, color='k', ls='--', alpha=1.0)
        plot_points(ax, dax1, zspec[cond_chi], zphot[cond_chi], fc=colors[cond_chi], ec=colors[cond_chi], s=12, lw=0, alpha=0.6, zorder=zorder[cond_chi])
        plot_hist(dax2, zspec[cond_chi], zphot[cond_chi], dbin=0.01, color='k', ls='-', alpha=1.0)

        ax.text(0.03, 0.98, "%i\" aperture \n" \
                            "$\chi^2 < %i$ (all) \n" \
                            "%i (%i) galaxies \n" \
                            "Outliers : %.1f%% (%.1f%%) \n" \
                            r"$\sigma_\mathrm{NMAD}$ : %.3f (%.3f)" % (apersize, chi_cut, len(zphot[cond_chi]), len(zphot), outliers2*100, outliers1*100, sig_NMAD2, sig_NMAD1),
                            va='top', ha='left', fontsize=14, transform=ax.transAxes)

    else:

        print()
        print( "%10s : %5s | %5s | %4s" % ("spec-z","N","sigma","out%"))
        print( "".join(['-']*35))
        for x in color_dict.keys():

            _cond = get_label_cond(x,catalog['ZSPEC_REF'])
            colors[_cond] = color_dict[x]
            zorder[_cond] = order_dict[x]

            sig_NMAD, outliers = calc_sig_NMAD(zphot[_cond],zspec[_cond])
            print( "%10s : %5i | %5.3f | %4.1f" % (x, np.sum(_cond), sig_NMAD, outliers*100))

        sig_NMAD, outliers = calc_sig_NMAD(zphot, zspec)
        print( "".join(['-']*35))
        print( "%10s : %5i | %5.3f | %4.1f" % ("all", len(zphot), sig_NMAD, outliers*100))
        print( "".join(['-']*35))
        print()

        plot_points(ax, dax1, zspec, zphot, fc=colors, ec=colors, s=15, lw=0, alpha=0.6, zorder=zorder)
        plot_hist(dax2, zspec, zphot, dbin=0.01, color='k', ls='-', alpha=1.0)

        ax.text(0.03, 0.98, "%i\" aperture \n" \
                            "%i galaxies \n" \
                           r"$\sigma_\mathrm{NMAD}$=%.3f""\n" \
                            "$\eta$=%.1f%% \n" % (apersize, len(zphot), sig_NMAD, outliers*100),
                            va='top', ha='left', fontsize=18, fontweight=600, transform=ax.transAxes)

    leg = ax.legend(loc="center left",fontsize=14,scatterpoints=1,markerscale=0,fancybox=True,frameon=False,handlelength=0,handletextpad=0,bbox_to_anchor=(0.96,0.42))
    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        try:
            txt.set_color(hndl.get_facecolor()[0])
        except:
            txt.set_color('w')
            txt.set_path_effects([PathEffects.Stroke(linewidth=2,foreground=hndl.get_edgecolor()[0]),PathEffects.Normal()])
        txt.set_fontproperties(FontProperties(size=14,weight=600))
        txt.set_ha("right")
        hndl.set_visible(False)

    if savename: fig.savefig(savename)

def compare_z_w_mag(zspec, zphot, savename=None):

    cond = (zphot["Z_ML"] != -99.)
    _zspec = zspec["ZSPEC"][cond]
    _zphot = zphot["Z_ML"][cond]
    _mag_i = zspec["MAG_AUTO_hsc_i"][cond]

    fig,ax = plt.subplots(1,1,figsize=(10,10),dpi=75)
    fig.subplots_adjust(left=0.08,right=0.99,bottom=0.08,top=0.98,wspace=0,hspace=0)

    colors = plt.cm.tab10(np.linspace(0,1,10))
    mag_cuts = [{"mlims":[16,21], "label":"      m$_i$<21", "color":colors[0], "zorder":9, "alpha":0.5},
                {"mlims":[21,22], "label":"21<m$_i$<22", "color":colors[1], "zorder":8, "alpha":0.5},
                {"mlims":[22,23], "label":"22<m$_i$<23", "color":colors[2], "zorder":7, "alpha":0.5},
                {"mlims":[23,24], "label":"23<m$_i$<24", "color":colors[3], "zorder":6, "alpha":0.5},
                {"mlims":[24,28], "label":"      m$_i$>24", "color":colors[4], "zorder":5, "alpha":0.5}]

    for i,entry in enumerate(mag_cuts):

        cond = (entry['mlims'][0] < _mag_i) & (_mag_i < entry['mlims'][1])
        ax.scatter(_zspec[cond], _zphot[cond], facecolor=entry['color'], edgecolor=entry['color'], marker='o', lw=0.0, s=20, zorder=entry["zorder"], alpha=entry["alpha"])
        sig_NMAD, outliers = calc_sig_NMAD(_zphot[cond], _zspec[cond])

        ax.text(0.02, 0.98-0.035*i, "%s :  $\sigma_\mathrm{NMAD}$=%.3f, $\eta$=%.1f%%" % (
                        entry["label"],sig_NMAD,outliers*100),
                        va='top', ha='left', color=entry["color"], fontsize=22, fontweight=600, transform=ax.transAxes)

    zz = np.arange(0,100,0.1)
    ax.plot(zz, zz, c='k', ls='-', lw=1.0, alpha=1.0)
    ax.plot(zz,  0.15*(1+zz)+zz,lw=0.75,ls='--',c='k',alpha=0.5)
    ax.plot(zz, -0.15*(1+zz)+zz,lw=0.75,ls='--',c='k',alpha=0.5)
    ax.set_aspect(1.)
    ax.set_xlim(-0.05,6.5)
    ax.set_ylim(-0.05,6.5)
    ax.set_ylabel(r'$z_{phot}$',fontsize=28)
    ax.set_xlabel(r'$z_{spec}$',fontsize=28)
    _ = [label.set_fontsize(16) for label in ax.get_xticklabels()+ax.get_yticklabels()]

    if savename: fig.savefig(savename)

def plot_photz_errors(zspec, zphot, savename=None):

    cond = (zphot["Z_ML"] != -99.)
    _zspec = zspec["ZSPEC"][cond]
    _zphot = zphot["Z_ML"][cond]
    _zerr  =(zphot["Z_ML68_HIGH"] - zphot["Z_ML68_LOW"])[cond]
    _mag_i = zspec["MAG_AUTO_hsc_i"][cond]

    fig = plt.figure(figsize=(9,12),dpi=75)
    fig.subplots_adjust(left=0.12,right=0.96,bottom=0.07,top=0.96)
    gs = gridspec.GridSpec(2,1,height_ratios=[3,1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    colors = plt.cm.Set1(np.linspace(0,1,10))
    mag_cuts = [{"mlims":[16,21], "label":"16<$m_i$<21", "color":colors[0], "zorder":4},
                {"mlims":[21,22], "label":"21<$m_i$<22", "color":colors[1], "zorder":3},
                {"mlims":[22,23], "label":"22<$m_i$<23", "color":colors[2], "zorder":2},
                {"mlims":[23,24], "label":"23<$m_i$<24", "color":colors[3], "zorder":1},
                {"mlims":[24,28], "label":"24<$m_i$<28", "color":colors[4], "zorder":1}]

    bins = np.arange(0,10,0.01)
    binc = 0.5*(bins[1:]+bins[:-1])
    resi = np.abs(_zphot - _zspec) / _zerr

    hist = np.histogram(resi,bins=bins)[0]
    chist = np.insert(np.cumsum(hist),0,0) / float(len(resi)) * 100.
    ax1.plot(bins,chist,c='k',lw=2,label="Full sample")

    for i,entry in enumerate(mag_cuts):

        cond = (entry['mlims'][0] < _mag_i) & (_mag_i < entry['mlims'][1])
        hist = np.histogram(resi[cond],bins=bins)[0]
        chist = np.insert(np.cumsum(hist),0,0) / float(len(resi[cond])) * 100.
        ax1.plot(bins,chist,c=entry["color"],lw=1.5,label=entry["label"])

    ax1.axvline(1,c='k',ls='--')
    ax1.set_xlabel("$|z_{phot} - z_{spec}|$ / 1$\sigma$ error",fontsize=20)
    ax1.set_ylabel("cumulative distribution [%]",fontsize=20)
    ax1.set_xlim(0,5)
    ax1.set_ylim(0,100)
    _ = [label.set_fontsize(14) for label in ax1.get_xticklabels()+ax1.get_yticklabels()]

    leg = ax1.legend(loc=4,fontsize=16,scatterpoints=1,markerscale=0,fancybox=True,frameon=False,handlelength=0,handletextpad=0)
    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        txt.set_color(hndl.get_color())
        txt.set_fontproperties(FontProperties(size=16,weight=600))
        hndl.set_visible(False)

    zbins = np.arange(-0.1,5,0.2)
    zbinc = 0.5*(zbins[1:]+zbins[:-1])
    digi  = np.digitize(_zphot,zbins) - 1
    zbine = np.array([np.median(_zerr[digi==i]) if len(_zerr[digi==i]) else np.NaN for i in range(len(zbinc))])
    ax2.plot(zbinc, zbine,c='k',lw=1.5,label="Full Sample")
    ax2.plot(zbinc,-zbine,c='k',lw=1.5)

    for i,entry in enumerate(mag_cuts):

        cond = (entry['mlims'][0] < _mag_i) & (_mag_i < entry['mlims'][1])
        digi  = np.digitize(_zphot[cond],zbins) - 1
        __zerr = _zerr[cond]
        zbine = np.array([np.median(__zerr[digi==i]) for i in range(len(zbinc))])
        ax2.plot(zbinc, zbine,c=entry["color"],lw=1.5,label=entry["label"])
        ax2.plot(zbinc,-zbine,c=entry["color"],lw=1.5)

    ax2.axhline(0,ls='--',lw=1.2)
    ax2.set_ylim(-0.15,0.15)
    ax2.set_xlim(0,4.5)
    ax2.set_xlabel("$z_{phot}$",fontsize=20)
    ax2.set_ylabel("1$\sigma$ error",fontsize=20)
    _ = [label.set_fontsize(14) for label in ax2.get_xticklabels()+ax2.get_yticklabels()]

    if savename: fig.savefig(savename)

def plot_mass_vs_z(catalog, savename=None):

    mass = catalog["LPH_MASS_BEST"]
    photoz = catalog["LPH_Z_ML"]
    cond = (photoz == -99.)
    photoz[cond] = catalog["LPH_Z_BEST"][cond]

    fig,ax = plt.subplots(1,1,figsize=(10,6),dpi=75,tight_layout=True)
    # fig = plt.figure(figsize=(10,8),dpi=75)
    # fig.subplots_adjust(left=0.08,right=0.96,bottom=0.08,top=0.96,wspace=0,hspace=0)
    # gs = gridspec.GridSpec(2,1,height_ratios=[1,3])
    # dax = fig.add_subplot(gs[0])
    # ax  = fig.add_subplot(gs[1])

    cond = np.isfinite(mass) & (catalog['LPH_CHI_BEST_PHYS'] < 50)
    # dax.hist(photoz[cond],bins=np.arange(-1,7,0.1),color='k',alpha=0.5)

    hist,xedges,yedges = np.histogram2d(photoz[cond],mass[cond],bins=[np.arange(-1,10,0.025),np.arange(0,20,0.1)])
    cond = (hist>0)
    hist[cond] = np.log10(hist[cond])
    xc = (xedges[1:] + xedges[:-1]) * 0.5
    yc = (yedges[1:] + yedges[:-1]) * 0.5
    ax.pcolormesh(xc,yc,hist.T,cmap=plt.cm.Greys,vmin=0,vmax=np.max(hist)*0.9)

    # ax.scatter(0.0696,7.28895,facecolor='r',marker='*',s=650,lw=1,edgecolor='k')
    ax.set_xlabel('Photo-z',fontsize=24)
    ax.set_ylabel('Stellar Mass, log (M$_\star$/M$_\odot$)',fontsize=24)
    ax.set_xlim(-0.05,4.5)
    ax.set_ylim(4.8,12.2)

    # dax.set_xlim(-0.05,4.5)

    _ = [tick.set_fontsize(16) for tick in ax.get_xticklabels()+ax.get_yticklabels()]
    # _ = [tick.set_visible(False) for tick in dax.get_xticklabels()+dax.get_yticklabels()]

    if savename: fig.savefig(savename)

def plot_mass_dist(catalog,savename):

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,10),dpi=75,tight_layout=True)

    # catalog = catalog[catalog['LPH_CHI_BEST'] < 3]
    catalog = catalog[catalog['LPH_Z_BEST'] > 0]

    mass = catalog["LPH_MASS_BEST"]
    photoz = catalog["LPH_Z_ML"]
    cond = (photoz == -99.)
    photoz[cond] = catalog["LPH_Z_BEST"][cond]

    zbins = np.arange(-1,1,0.005)
    ax1.hist(np.log10(1+photoz),bins=zbins,color='k',alpha=0.5)

    mbins = np.arange(2,15,0.1)
    mbinc = 0.5*(mbins[:-1]+mbins[1:])
    zz = [0,0.25,0.5,0.75,1,1.5,2,3,4,5,6] #np.arange(0,6.1,1)
    cc = plt.cm.gist_rainbow_r(np.linspace(0.1,1,len(zz)-1))

    for z0,z1,c in zip(zz[:-1],zz[1:],cc):
        cond = (z0<=photoz) & (photoz<z1) & np.isfinite(mass)
        ax1.hist(np.log10(1+photoz[cond]),bins=zbins,color=c,alpha=0.5)
        hist = np.histogram(mass[cond],bins=mbins)[0]
        ax2.plot(mbinc,hist,color=c,lw=1.2,label="%.2f<z<%.2f"%(z0,z1))

    ax1.set_xlim(np.log10(1+np.array([-0.1,6.5])))
    ax1.set_ylabel("N")
    ax1.set_xlabel("Redshift")
    ax1.set_xticks(np.log10(1+np.arange(7)))
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x_val, tick_pos: "%.f"%(10**x_val-1)))

    ax2.set_xlim(3.8,12.5)
    ax2.set_ylabel("N")
    ax2.set_xlabel("Log Mass [M$_\odot$]")
    ax2.legend()

    _ = [tick.set_visible(False) for tick in ax1.get_yticklabels()+ax2.get_yticklabels()]

    if savename: fig.savefig(savename)

def check_differences_v15_to_v16():

    catalog_v15 = fitsio.getdata("final_cats/sxds_catalog_v1.5.fits")
    catalog_v16 = fitsio.getdata("final_cats/sxds_catalog_v1.6.fits")

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,7),dpi=75,tight_layout=True)

    bins = np.arange(-1,10,0.02)
    binc = 0.5*(bins[1:]+bins[:-1])
    hist = np.histogram2d(catalog_v15["ZPHOT"],catalog_v16["ZPHOT"],bins=[bins,bins])[0]
    im = ax1.pcolormesh(binc,bins,np.log10(hist.T),cmap=plt.cm.Greys,vmin=0,vmax=3)

    cax = fig.add_axes(make_axes_locatable(ax1).new_horizontal(size="3%",pad=0))
    fig.colorbar(mappable=im, cax=cax, orientation='vertical')

    ax1.set_xlabel("Photo-z (v1.5)")
    ax1.set_ylabel("Photo-z (v1.6)")
    ax1.set_aspect(1)
    ax1.set_xlim(0,6)
    ax1.set_ylim(0,6)

    bins = np.arange(0,20,0.05)
    binc = 0.5*(bins[1:]+bins[:-1])
    hist = np.histogram2d(catalog_v15["MASS_BEST"],catalog_v16["MASS_BEST"],bins=[bins,bins])[0]
    im = ax2.pcolormesh(binc,bins,np.log10(hist.T),cmap=plt.cm.Greys,vmin=0,vmax=2.5)

    cax = fig.add_axes(make_axes_locatable(ax2).new_horizontal(size="3%",pad=0))
    fig.colorbar(mappable=im, cax=cax, orientation='vertical')

    ax2.set_xlabel("Stellar Mass (v1.5)")
    ax2.set_ylabel("Stellar Mass (v1.6)")
    ax2.set_aspect(1)
    ax2.set_xlim(3,14)
    ax2.set_ylim(3,14)

if __name__ == '__main__':

    # catalog = fitsio.getdata("final_cats/final_catalog_errfix.extra.fits")
    # plot_specz(catalog,savename='lephare/plots/specz_dist.png')

    # catalog_zphot = useful.read_lephare_photoz("lephare/calib/catalog_errfix_r2.out")
    # compare_z(catalog[catalog_zphot['ID']-1], catalog_zphot, apersize=2, savename='lephare/plots/compare_z_calib_errfix_r2.png')
    # compare_z(catalog[catalog_zphot['ID']-1], catalog_zphot, apersize=2, savename='lephare/plots/compare_z_calib_errfix_r2.Z_BEST.png', z_ml=False)
    # compare_z_w_mag(catalog[catalog_zphot['ID']-1], catalog_zphot, savename='lephare/plots/compare_z_calib_errfix_r2.split_mag.png')
    # plot_photz_errors(catalog[catalog_zphot['ID']-1], catalog_zphot, savename='lephare/plots/compare_z_calib_errfix_r2.errors.png')

    # catalog_zphot = useful.read_lephare_photoz("lephare/calib/catalog_errfix_r2.IRAC34.out")
    # compare_z(catalog[catalog_zphot['ID']-1], catalog_zphot, apersize=2, savename='lephare/plots/compare_z_calib_errfix_r2.IRAC34.png')

    # catalog_zphot = useful.read_lephare_photoz("lephare/calib/catalog_errfix_r3.out")
    # compare_z(catalog[catalog_zphot['ID']-1], catalog_zphot, apersize=3, savename='lephare/plots/compare_z_calib_errfix_r3.png')

    # catalog_zphot = useful.read_lephare_photoz("lephare/calib/catalog_r2.out")
    # compare_z(catalog[catalog_zphot['ID']-1], catalog_zphot, apersize=2, savename='lephare/plots/compare_z_calib_r2.png')

    # catalog_zphot = useful.read_lephare_photoz("lephare/calib/catalog_r3.out")
    # compare_z(catalog[catalog_zphot['ID']-1], catalog_zphot, apersize=3, savename='lephare/plots/compare_z_calib_r3.png')

    # catalog = fitsio.getdata("final_cats/final_catalog_errfix_zphot.extra.fits")
    # plot_mass_vs_z(catalog, savename='lephare/plots/mass_vs_zphot.png')
    # plot_mass_dist(catalog, savename='lephare/plots/mass_z_dists.png')

    check_differences_v15_to_v16()
    plt.show()


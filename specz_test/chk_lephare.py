import astropy.io.fits as fitsio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
from matplotlib import transforms
from matplotlib.ticker import FuncFormatter
from matplotlib.font_manager import FontProperties
from collections import OrderedDict
from astropy.wcs import WCS

import useful
from rebin import rebin

def rebin_(img,factor):

    return rebin(img,factor=factor,mode='mean')

def logscale(img):

    mean,std = np.median(img), np.std(img)
    x0, x1, x2 = 0., mean+0.3*std, mean+0.6*std

    xo = x0
    k  = (x2 - 2*x1 + x0) / (x2 - x0)**2
    r  = np.log10( k * (x2-x0) + 1 )

    return np.log10( k * (img - xo) + 1 ) / r

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
    sizes_dict = OrderedDict(zip(label_dict.keys(),[3,15,15,35,15]))
    # order_dict = OrderedDict(zip(label_dict.keys(),[1,3,5,4,6,2]))
    # sizes_dict = OrderedDict(zip(label_dict.keys(),[5,10,10,15,30,15]))
    return label_dict, color_dict, order_dict, sizes_dict

def get_label_cond(label,zref):

    if "SUBARU"==label: _cond = [i in useful.SUBARU_key.values() for i in zref]
    elif "XUDS"==label: _cond = [i in useful.XUDS_key.values() for i in zref]
    else: _cond = [label in i for i in zref]
    return _cond

def calc_sig_NMAD(zphot, zspec):

    diff = zphot - zspec
    frac = diff/(1+zspec)
    outliers = len(frac[abs(frac) > 0.15])
    residual = (diff - np.median(diff)) / (1. + zspec)
    sig_NMAD = 1.48 * np.median(abs(residual))
    return sig_NMAD, outliers / float(len(frac))

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
    dax.axhline(-0.15,lw=0.75,ls='--',c='k', alpha=0.5)
    dax.axhline(0.15, lw=0.75,ls='--',c='k', alpha=0.5)
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

def plot_points(axis, daxis, zspec, zphot, fc, ec, s, alpha, lw, zorder, med=True):

    diff = zphot - zspec
    frac = diff/(1+zspec)
    for i,_z in enumerate(np.unique(zorder)):
        cond = (zorder==_z)
        _fc = fc[cond] if isinstance(fc,np.ndarray) else fc
        _ec = ec[cond] if isinstance(ec,np.ndarray) else ec
        axis.scatter(zspec[cond], zphot[cond], facecolor=_fc, edgecolor=_ec, marker='o', lw=lw, s=s, alpha=alpha, zorder=_z)
        daxis.scatter(zspec[cond], frac[cond], facecolor=_fc, edgecolor=_ec, marker='o', lw=lw, s=s, alpha=alpha, zorder=_z)
    if med: daxis.axhline(np.median(frac),c='k',lw=1.5,ls='--')

def plot_hist(daxis, zspec, zphot, dbin, color, ls, alpha, label=None, med=True, norm=False):

    diff = zphot - zspec
    frac = diff/(1+zspec)
    bins = np.arange(-2.5-dbin/2.,2.5+dbin/2.+1e-6,dbin)
    binc = 0.5*(bins[1:] + bins[:-1])
    hist = np.histogram(frac, bins=bins)[0]
    if norm: hist = hist/float(np.max(hist))
    daxis.plot(binc, hist, color=color, ls=ls, lw=1.25, alpha=1.0, label=label)
    if med: daxis.axvline(np.median(frac),c='k',lw=1.5,ls='--')
    daxis.set_ylim(0, max(daxis.get_ylim()[1],1.1*max(hist)))

def compare_z(catalog, lph_output, aper_text, z_ml=True, savename=None):

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

    fig, ax, dax1, dax2 = setup_fig()

    label_dict, color_dict, order_dict, sizes_dict = get_label_color_dicts()
    for x in color_dict.keys():
        ax.scatter(-99, -99, facecolor=color_dict[x], edgecolor=color_dict[x], label=label_dict[x])

    colors = np.zeros((len(zspec),4))
    zorder = np.zeros(len(zspec))

    print
    print "%10s : %5s | %5s | %4s" % ("spec-z","N","sigma","out%")
    print "".join(['-']*35)
    for x in color_dict.keys():

        _cond = get_label_cond(x,catalog['ZSPEC_REF'])
        colors[_cond] = color_dict[x]
        zorder[_cond] = order_dict[x]

        if len(zphot[_cond]) > 1:
            sig_NMAD, outliers = calc_sig_NMAD(zphot[_cond],zspec[_cond])
            print "%10s : %5i | %5.3f | %4.1f" % (x, np.sum(_cond), sig_NMAD, outliers*100)

    sig_NMAD, outliers = calc_sig_NMAD(zphot, zspec)
    print "".join(['-']*35)
    print "%10s : %5i | %5.3f | %4.1f" % ("all", len(zphot), sig_NMAD, outliers*100)
    print "".join(['-']*35)
    print

    plot_points(ax, dax1, zspec, zphot, fc=colors, ec=colors, s=15, lw=0, alpha=0.6, zorder=zorder)
    plot_hist(dax2, zspec, zphot, dbin=0.01, color='k', ls='-', alpha=1.0)

    ax.text(0.03, 0.98, "%s aperture \n" \
                        # "%i galaxies \n" \
                       r"$\sigma_\mathrm{NMAD}$=%.3f""\n" \
                        "$\eta$=%.1f%% \n" % (aper_text, sig_NMAD, outliers*100),
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

def compare_z_area(catalog,lph_output,savename=None):

    assert np.all(catalog['ID'] == lph_output['ID'])

    # lph_output["Z_BEST"][lph_output["CHI_BEST"] > lph_output["CHI_STAR"]] = 0
    # lph_output["Z_BEST"][lph_output["CHI_BEST"] > lph_output["CHI_QSO"]] = lph_output["Z_QSO"][lph_output["CHI_BEST"] > lph_output["CHI_QSO"]]

    zspec = catalog["ZSPEC"]
    zphot = lph_output["Z_ML"]

    cond = (zphot!=-99.) & (catalog["ZSPEC"]!=-99.)
    zspec = zspec[cond]
    zphot = zphot[cond]
    catalog = catalog[cond]
    lph_output = lph_output[cond]
    in_hsc = (catalog['USE_ZSPEC_FLAG']==1)

    fig, ax, dax1, dax2 = setup_fig()
    color1,color2 = plt.cm.tab10([0,0.3])

    sig_NMAD, outliers = calc_sig_NMAD(zphot[in_hsc], zspec[in_hsc])
    plot_points(ax, dax1, zspec[in_hsc], zphot[in_hsc], fc=color1, ec=color1, s=12, lw=0, alpha=0.6, zorder=1, med=False)
    plot_hist(dax2, zspec[in_hsc], zphot[in_hsc], dbin=0.01, color=color1, ls='-', alpha=1.0, med=False, norm=False)
    ax.text(0.03, 0.98, "Within HSC-UD area \n%i galaxies \n" \
                       r"$\sigma_\mathrm{NMAD}$=%.3f; " \
                        "$\eta$=%.1f%% \n" % (len(zphot[in_hsc]), sig_NMAD, outliers*100),
                        va='top', ha='left', color=color1, fontsize=14, fontweight=600, transform=ax.transAxes)

    sig_NMAD, outliers = calc_sig_NMAD(zphot[~in_hsc], zspec[~in_hsc])
    plot_points(ax, dax1, zspec[~in_hsc], zphot[~in_hsc], fc=color2, ec=color2, s=12, lw=0, alpha=0.6, zorder=2, med=False)
    plot_hist(dax2, zspec[~in_hsc], zphot[~in_hsc], dbin=0.01, color=color2, ls='-', alpha=1.0, med=False, norm=False)
    ax.text(0.03, 0.88, "Outside HSC-UD area \n%i galaxies \n" \
                       r"$\sigma_\mathrm{NMAD}$=%.3f; " \
                        "$\eta$=%.1f%% \n" % (len(zphot[~in_hsc]), sig_NMAD, outliers*100),
                        va='top', ha='left', color=color2, fontsize=14, fontweight=600, transform=ax.transAxes)

    if savename: fig.savefig(savename)

def compare_z_bump(catalog,lph_output,savename=None):

    assert np.all(catalog['ID'] == lph_output['ID'])

    zspec = catalog["ZSPEC"]
    zphot = lph_output["Z_ML"]

    cond = (zphot!=-99.) & (34.22195<catalog["RA"]) & (catalog["RA"]<34.592547) & (-5.2781681<catalog["DEC"]) & (catalog["DEC"]<-5.124545)
    zspec = zspec[cond]
    zphot = zphot[cond]
    catalog = catalog[cond]
    lph_output = lph_output[cond]
    in_M15 = np.array(get_label_cond("M15",catalog['ZSPEC_REF']))

    fig, axes = plt.subplots(2,2,figsize=(14.5,10),dpi=75,tight_layout=False,sharex=True,sharey='row',gridspec_kw={"height_ratios":[3,1]})
    fig.subplots_adjust(left=0.065,right=0.985,bottom=0.08,top=0.98,hspace=0,wspace=0.01)
    axes = axes.flatten()

    for i,(ax,dax) in enumerate(zip([axes[0],axes[1]],[axes[2],axes[3]])):

        zz = np.arange(0,100,0.1)
        ax.plot(zz, zz, c='k', ls='-', lw=1.0, alpha=1.0)
        ax.plot(zz,  0.15*(1+zz)+zz,lw=0.75,ls='--',c='k',alpha=0.5)
        ax.plot(zz, -0.15*(1+zz)+zz,lw=0.75,ls='--',c='k',alpha=0.5)
        ax.set_aspect(1.)
        ax.set_xlim(-0.05,6.5)
        ax.set_ylim(-0.05,6.5)
        ax.tick_params(axis='both',which='major',direction='in',length=5,width=1.2)
        if i==0: ax.set_ylabel(r'$z_{phot}$',fontsize=28)
        _ = [label.set_visible(False) for label in ax.get_xticklabels()]
        _ = [label.set_fontsize(14) for label in ax.get_yticklabels()]

        dax.axhline(0, c='k', ls='-', lw=1.0, alpha=1.0)
        dax.axhline(-0.15,lw=0.75,ls='--',c='k', alpha=0.5)
        dax.axhline(0.15, lw=0.75,ls='--',c='k', alpha=0.5)
        dax.set_xlim(-0.05,6.5)
        dax.set_ylim(-0.55,0.55)
        dax.set_yticks([-0.5,0.0,0.5])
        dax.tick_params(axis='both',which='major',direction='in',length=5,width=1.2)
        dax.set_xlabel(r'$z_{spec}$',fontsize=28)
        if i==0: dax.set_ylabel(r'$\Delta z/(1+z)$',fontsize=18)
        _ = [label.set_fontsize(14) for label in dax.get_xticklabels()+dax.get_yticklabels()]

    inset = fig.add_axes([0.33,0.82,0.18,0.15])
    _ = [label.set_fontsize(12) for label in inset.get_xticklabels()+inset.get_yticklabels()]
    inset.axvline(0,lw=0.75,ls='-',c='k')
    inset.axvline(-0.15,lw=0.75,ls=':',c='k')
    inset.axvline(0.15, lw=0.75,ls=':',c='k')
    inset.set_xlim(-0.45,0.45)
    inset.set_xlabel(r'$\Delta z/(1+z)$',fontsize=16)
    inset.set_yticks([])
    _ = [_.set_visible(False) for _ in inset.get_yticklabels()]

    color1,color2 = plt.cm.tab10([0,0.3])

    sig_NMAD, outliers = calc_sig_NMAD(zphot[in_M15], zspec[in_M15])
    plot_points(axes[0], axes[2], zspec[in_M15], zphot[in_M15], fc=color2, ec=color2, s=12, lw=0, alpha=0.6, zorder=1, med=False)
    plot_hist(inset, zspec[in_M15], zphot[in_M15], dbin=0.01, color=color2, ls='-', alpha=1.0, med=False, norm=True)
    axes[0].text(0.03, 0.98, "Morris+15 \n" \
                        "%i galaxies \n" \
                       r"$\sigma_\mathrm{NMAD}$=%.3f""\n" \
                        "$\eta$=%.1f%% \n" % (len(zphot[in_M15]), sig_NMAD, outliers*100),
                        va='top', ha='left', color=color2, fontsize=14, fontweight=600, transform=axes[0].transAxes)

    sig_NMAD, outliers = calc_sig_NMAD(zphot[~in_M15], zspec[~in_M15])
    plot_points(axes[1], axes[3], zspec[~in_M15], zphot[~in_M15], fc=color1, ec=color1, s=12, lw=0, alpha=0.6, zorder=2, med=False)
    plot_hist(inset, zspec[~in_M15], zphot[~in_M15], dbin=0.01, color=color1, ls='-', alpha=1.0, med=False, norm=True)
    axes[1].text(0.03, 0.98, "Other spec-z\n(within Morris+15 area) \n" \
                        "%i galaxies \n" \
                       r"$\sigma_\mathrm{NMAD}$=%.3f""\n" \
                        "$\eta$=%.1f%% \n" % (len(zphot[~in_M15]), sig_NMAD, outliers*100),
                        va='top', ha='left', color=color1, fontsize=14, fontweight=600, transform=axes[1].transAxes)

    if savename: fig.savefig(savename)

def plot_specz_coverage(catalog,savename=None):

    fig,ax = plt.subplots(1,1,figsize=(12,9.25),dpi=70,tight_layout=False)
    fig.subplots_adjust(left=0.1, top=0.98, bottom=0.08, right=0.8)

    factor = 20
    x,y = np.indices((2501,2501)) * factor
    hdr = fitsio.getheader("/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/data/orig/mosaic_hsc_y.img.fits")
    wcs = WCS(hdr)
    r,d = wcs.all_pix2world(x,y,1)

    # img = np.clip(logscale(rebin_(fitsio.getdata("/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/data/orig/mosaic_hsc_y.img.fits"),factor=factor)),0,1)
    # ax.pcolormesh(r,d,img.T,cmap=plt.cm.Greys,vmin=0,vmax=1,linewidth=0)

    ax.add_patch(mpatches.Ellipse(  xy=(34.576835,-4.8179672),width=2*3228.1658/3600.,height=2*2955.5327/3600.,
                                        edgecolor=useful.fcolor_dict['hsc']['g'],facecolor='none',lw=4,label="HSC"))
    ax.add_patch(mpatches.Rectangle(xy=(34.498508-7387.7200/3600./2.,-5.0016043-7387.7200/3600./2.),width=7387.7200/3600.,height=7387.7200/3600.,
                                        edgecolor=useful.fcolor_dict['irac']['1'],facecolor='none',lw=4,label='IRAC'))
    ax.add_patch(mpatches.Rectangle(xy=(34.448884-3385.1568/3600./2.,-5.1000118-3356.3060/3600./2.),width=3385.1568/3600.,height=3356.3060/3600.,
                                        edgecolor=useful.fcolor_dict['uds']['j'],facecolor='none',lw=4,label='UVISTA/UDS'))
    ax.add_patch(mpatches.Polygon(  xy=([(34.803238,-4.697391),(34.804941,-4.3579328),(34.194745,-4.3542229),(34.192725,-4.6936791),(33.827932,-4.6934248),(33.827302,-5.3074201),(34.198027,-5.3058556),(34.196005,-5.6471666),(34.812998,-5.647162),(34.810968,-5.2984311),(35.159337,-5.3018698),(35.155002,-4.6934411)]),closed=True,
                                        edgecolor=useful.fcolor_dict['supcam']['b'],facecolor='none',lw=4,label='SuprimeCam'))
    ax.add_patch(mpatches.Polygon(  xy=([(35.985706,-4.0849371),(34.914891,-4.0755168),(34.914993,-4.2357598),(34.778398,-4.2384915),(34.778363,-4.1556964),(33.838372,-4.1581396),(33.831747,-5.6483204),(35.058136,-5.6406718),(35.058002,-5.4830838),(35.184114,-5.4829538),(35.186878,-5.5604084),(36.115056,-5.5399199),(36.114885,-5.4704973),(37.164244,-5.4647929),(37.158309,-4.0031709),(35.996234,-4.0074955)]),closed=True,
                                        edgecolor=useful.fcolor_dict['video']['y'],facecolor='none',lw=4,label='VIDEO'))
    ax.add_patch(mpatches.Polygon(  xy=([(35.054335,-4.3211403),(34.068559,-4.3292324),(34.063159,-4.3933323),(33.843493,-4.3931722),(33.838602,-5.4413465),(33.926591,-5.4375716),(33.926425,-5.626922),(34.006673,-5.628283),(34.005358,-5.6553313),(34.994323,-5.6527556),(34.994297,-5.6179777),(35.125212,-5.6165671),(35.124213,-4.5694852),(35.051861,-4.564202)]),closed=True,
                                        edgecolor=useful.fcolor_dict['cfht']['u'],facecolor='none',lw=4,label='MUSUBI'))

    ax.set_xlabel("RA [deg]",fontsize=18)
    ax.set_ylabel("Decl. [deg]",fontsize=18)
    ax.set_xlim(35.55, 33.45)
    ax.set_ylim(-6.05, -3.95)
    ax.set_aspect(1.)

    _ = [label.set_fontsize(14) for label in ax.get_xticklabels()+ax.get_yticklabels()]

    # PLOT SPEC-Z
    catalog = catalog[catalog["ZSPEC"]!=-99.]
    colors = np.zeros((len(catalog),4))
    zorder = np.zeros( len(catalog))
    sizes  = np.zeros( len(catalog))

    label_dict, color_dict, order_dict, sizes_dict = get_label_color_dicts()
    for x in color_dict.keys():
        _cond = get_label_cond(x,catalog['ZSPEC_REF'])
        colors[_cond] = color_dict[x]
        zorder[_cond] = order_dict[x]
        sizes[ _cond] = sizes_dict[x]

    for i,_z in enumerate(np.unique(zorder)):
        cond = (zorder==_z)
        ax.scatter(catalog["RA"][cond], catalog["DEC"][cond], facecolor=colors[cond], edgecolor=colors[cond], marker='o', lw=0, s=sizes[cond], alpha=0.8, zorder=_z)

    def write_text(x,y,s,c,fs,fw,t,cs,ext=1,alpha=1):
        text = ax.text(x,y,s,color=c,transform=t,fontsize=fs,fontweight=fw,alpha=alpha)
        text.draw(cs.get_renderer())
        ex = text.get_window_extent()
        t  = transforms.offset_copy(text._transform, y=-ext*ex.height, units='dots')
        return t

    t = ax.transAxes
    canvas = ax.figure.canvas
    tx,ty = 1.01,0.78

    instrs = ['HSC','IRAC','UVISTA/UDS','SuprimeCam','VIDEO','MUSUBI']
    colors = [useful.fcolor_dict['hsc']['g'],
              useful.fcolor_dict['irac']['1'],
              useful.fcolor_dict['uds']['j'],
              useful.fcolor_dict['supcam']['b'],
              useful.fcolor_dict['video']['y'],
              useful.fcolor_dict['cfht']['u']]

    t = write_text(tx,ty,"Coverage (regions):",c="k",fs=14,fw=600,t=t,cs=canvas,ext=1.7,alpha=0.5)
    for instr,color in zip(instrs,colors):
        t = write_text(tx,ty,instr,c=color,fs=18,fw=600,t=t,cs=canvas,ext=1.2)

    t = write_text(tx,ty," ",c="w",fs=14,fw=400,t=t,cs=canvas,ext=3)
    t = write_text(tx,ty,"Spec-z (points):",c="k",fs=14,fw=600,t=t,cs=canvas,ext=1.7,alpha=0.5)
    for x in color_dict.keys():
        t = write_text(tx,ty,label_dict[x],c=color_dict[x],fs=18,fw=600,t=t,cs=canvas,ext=1.2)

    if savename: fig.savefig(savename)

def chk_z_med(catalog, catalog_zphot):

    catalog = catalog
    catalog_zphot = catalog_zphot

    cond = (catalog_zphot["Z_ML"]==-99.)
    np.savetxt("no_z_med.reg",np.vstack((catalog["RA"][cond],catalog["DEC"][cond])).T,fmt="circle(%.6f,%.6f,2\")",header="fk5",comments="")
    print np.sum(cond)

    # fig,ax = plt.subplots(1,1,figsize=(11,6),dpi=75,tight_layout=True)

    # bins = 10**np.arange(-2,10.5,0.1)
    # ax.hist(catalog_zphot["CHI_STAR"][cond], bins=bins, color='r', alpha=0.5, histtype='stepfilled')
    # ax.hist(catalog_zphot["CHI_STAR"],       bins=bins, color='r', alpha=1.0, histtype='step')
    # ax.hist(catalog_zphot["CHI_QSO"][cond],  bins=bins, color='b', alpha=0.5, histtype='stepfilled')
    # ax.hist(catalog_zphot["CHI_QSO"],        bins=bins, color='b', alpha=1.0, histtype='step')

    # ax.set_xscale('log')
    # ax.set_yscale('log')

    # fig1,axes1 = plt.subplots(4,7,figsize=(18,8),dpi=75,sharey=True)
    # fig1.subplots_adjust(wspace=0,left=0.04,right=0.98,bottom=0.05,top=0.95)
    # fig1.suptitle("Flux",fontsize=16,fontweight=600)
    # axes1 = axes1.flatten()

    # fig2,axes2 = plt.subplots(4,7,figsize=(18,8),dpi=75,sharey=True)
    # fig2.subplots_adjust(wspace=0,left=0.04,right=0.98,bottom=0.05,top=0.95)
    # fig2.suptitle("Flux error",fontsize=16,fontweight=600)
    # axes2 = axes2.flatten()

    # fig3,axes3 = plt.subplots(4,7,figsize=(18,8),dpi=75,sharey=True)
    # fig3.subplots_adjust(wspace=0,left=0.04,right=0.98,bottom=0.05,top=0.95)
    # fig3.suptitle("S/N",fontsize=16,fontweight=600)
    # axes3 = axes3.flatten()

    # for i,fname in enumerate(useful.fnames):

    #     instr,filt = fname.split('_')

    #     if instr!='irac':
    #         flux = catalog["FLUX_APER_%s"%fname][:,1]
    #         ferr = catalog["FLUXERR_APER_%s"%fname][:,1]
    #     else:
    #         flux = catalog["FLUX_TOT_%s"%fname]
    #         ferr = catalog["FLUXERR_TOT_%s"%fname]

    #         _cond = (catalog["OFFSET_FLUX"][:,2]==-99.)
    #         flux[_cond] = -99.
    #         ferr[_cond] = -99.

    #         _cond = (flux!=-99.)
    #         flux[_cond] = flux[_cond] / catalog["OFFSET_FLUX"][:,1][_cond]
    #         _cond = (ferr!=-99.)
    #         ferr[_cond] = ferr[_cond] / catalog["OFFSET_FLUX"][:,1][_cond]

        # cond_flux0 = flux>0
        # cond = cond & cond_flux0
        # bins = 10**np.arange(min(np.log10(flux[cond_flux0])),max(np.log10(flux[cond_flux0]))+0.05,0.05)
        # axes1[i].hist(flux[cond],bins=bins,color=useful.fcolor_dict[instr][filt],alpha=1.0,histtype='stepfilled')
        # axes1[i].hist(flux,      bins=bins,color=useful.fcolor_dict[instr][filt],alpha=1.0,histtype='step')
        # axes1[i].text(0.95,0.95,fname,color=useful.fcolor_dict[instr][filt],va='top',ha='right',fontsize=16,fontweight=600,transform=axes1[i].transAxes)
        # axes1[i].set_xscale("log")
        # axes1[i].set_yscale("log")

        # cond_ferr0 = ferr>0
        # cond = cond & cond_ferr0
        # bins = 10**np.arange(min(np.log10(ferr[cond_ferr0])),max(np.log10(ferr[cond_ferr0]))+0.05,0.05)
        # axes2[i].hist(ferr[cond],bins=bins,color=useful.fcolor_dict[instr][filt],alpha=1.0,histtype='stepfilled')
        # axes2[i].hist(ferr,      bins=bins,color=useful.fcolor_dict[instr][filt],alpha=1.0,histtype='step')
        # axes2[i].text(0.95,0.95,fname,color=useful.fcolor_dict[instr][filt],va='top',ha='right',fontsize=16,fontweight=600,transform=axes2[i].transAxes)
        # axes2[i].set_xscale("log")
        # axes2[i].set_yscale("log")

        # cond_ferr0 = ferr>0
        # cond = cond & cond_ferr0
        # bins = 10**np.arange(-3,5,0.05)
        # axes3[i].hist(flux[cond]/ferr[cond],bins=bins,color=useful.fcolor_dict[instr][filt],alpha=1.0,histtype='stepfilled')
        # axes3[i].hist(flux/ferr,            bins=bins,color=useful.fcolor_dict[instr][filt],alpha=1.0,histtype='step')
        # axes3[i].text(0.95,0.95,fname,color=useful.fcolor_dict[instr][filt],va='top',ha='right',fontsize=16,fontweight=600,transform=axes3[i].transAxes)
        # axes3[i].set_xscale("log")
        # axes3[i].set_yscale("log")

if __name__ == '__main__':

    catalog = fitsio.getdata("../final_cats/final_catalog_errfix.extra.fits")

    # catalog_zphot = useful.read_lephare_photoz("../lephare/calib/catalog_errfix_r2.out")
    # compare_z(catalog[catalog_zphot['ID']-1], catalog_zphot, aper_text="2\"", savename='plots/compare_z_calib_errfix_aper.png')

    # catalog_zphot = useful.read_lephare_photoz("lephare/calib/catalog_errfix_iso.out")
    # compare_z(catalog[catalog_zphot['ID']-1], catalog_zphot, aper_text="ISO", savename='plots/compare_z_calib_errfix_iso.png')

    # catalog_zphot = fitsio.getdata("../lephare/photoz/catalog_errfix_r2.out.fits")
    # compare_z_area(catalog, catalog_zphot, savename='plots/compare_z_calib_errfix_area.png')

    # catalog_zphot = useful.read_lephare_photoz("lephare/calib/catalog_errfix_r2_no_hsc.out")
    # compare_z(catalog[catalog_zphot['ID']-1], catalog_zphot, aper_text='No HSC', savename=None)

    # plot_specz_coverage(catalog,savename="plots/specz_coverage.png")

    catalog_zphot = fitsio.getdata("../lephare/photoz/catalog_errfix_r2.out.fits")
    chk_z_med(catalog, catalog_zphot)

    plt.show()

import sys
import numpy as np
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
from matplotlib.font_manager import FontProperties

import useful

def calc_opt_nir_offset(catalog,aper_num=2):

    offset_flux, offset_mag = np.zeros(len(catalog))-99., np.zeros(len(catalog))-99.
    term1,term2 = np.zeros(len(catalog)),np.zeros(len(catalog))

    for instr in useful.instr_used_list[:-1]:

        for filt in useful.filters[instr]:

            print (instr,filt)
            cond_f = (catalog[   'FLUX_AUTO_%s_%s'%(instr,filt)] > 0.0) & (catalog[   'FLUX_APER_%s_%s'%(instr,filt)][:,aper_num] > 0.0)
            # cond_f = (catalog['FLUX_AUTO_%s_%s'%(instr,filt)] / catalog['FLUXERR_AUTO_%s_%s'%(instr,filt)] > 3.0) & \
            #          (catalog['FLUX_APER_%s_%s'%(instr,filt)][:,aper_num] / catalog['FLUXERR_APER_%s_%s'%(instr,filt)][:,aper_num] > 3.0)

            diff_f,wht_f = np.zeros(len(catalog)),np.zeros(len(catalog))
            diff_f[cond_f] = catalog['FLUX_AUTO_%s_%s'%(instr,filt)][cond_f] / catalog['FLUX_APER_%s_%s'%(instr,filt)][:,aper_num][cond_f]
            wht_f[cond_f] = 1./(catalog['FLUXERR_AUTO_%s_%s'%(instr,filt)][cond_f]**2 + catalog['FLUXERR_APER_%s_%s'%(instr,filt)][:,aper_num][cond_f]**2)
            term1 += wht_f * diff_f
            term2 += wht_f

    cond_f = (term2>0)
    offset_flux[cond_f] = term1[cond_f]/term2[cond_f]

    return offset_flux

def main():

    catalog = fitsio.getdata("final_cats/final_catalog_errfix.fits")

    fnames = [x for x in useful.sorted_pivot_l if "irac" not in x]
    _fnames = np.array_split(fnames,4)
    fig = plt.figure(figsize=(18,12),dpi=75)
    fig.subplots_adjust(left=0.05,right=0.98,top=0.95,bottom=0.05,wspace=0,hspace=0.5)
    ogs = gridspec.GridSpec(4,len(_fnames[0]))

    for i,fname1 in enumerate(fnames[:-1]):

        print( fname1)

        igs = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=ogs[i],wspace=0,hspace=0)
        ax1 = fig.add_subplot(igs[0])
        ax2 = fig.add_subplot(igs[1],sharex=ax1)

        # for fname2 in [x for x in fnames if x!=fname1]:

        fname2 = fnames[i+1]
        cond = (catalog['FLUX_AUTO_%s'%fname1]>0) & (catalog['FLUX_APER_%s'%fname1][:,2]>0) & \
               (catalog['FLUX_AUTO_%s'%fname2]>0) & (catalog['FLUX_APER_%s'%fname2][:,2]>0)

        color_ap = -2.5 * np.log10(catalog["FLUX_APER_%s"%fname1][:,2][cond]/catalog["FLUX_APER_%s"%fname2][:,2][cond])
        color_au = -2.5 * np.log10(catalog["FLUX_AUTO_%s"%fname1][cond]     /catalog["FLUX_AUTO_%s"%fname2][cond]     )
        offset_1 = catalog["FLUX_AUTO_%s"%fname1][cond]/catalog["FLUX_APER_%s"%fname1][:,2][cond]
        offset_2 = catalog["FLUX_AUTO_%s"%fname2][cond]/catalog["FLUX_APER_%s"%fname2][:,2][cond]

        instr,filt = fname1.split("_")

        ax1.scatter(color_ap,offset_1,c=useful.fcolor_dict[instr][filt],s=3,alpha=0.05)
        ax2.scatter(color_au,offset_1,c=useful.fcolor_dict[instr][filt],s=3,alpha=0.05)

        ax1.text(0.02,0.97,"%s-%s"%(fname1,fname2),va='top',ha='left',fontsize=14,fontweight=600,color=useful.fcolor_dict[instr][filt],transform=ax1.transAxes)

        ax1.axhline(1,c='k',ls='--')
        ax1.set_xlim(-8.5,8.5)
        ax1.set_ylim(1e-4,1e4)
        ax1.set_yscale('log')
        ax1.set_xlabel("APER Color")
        ax1.xaxis.set_label_position('top')
        ax1.xaxis.tick_top()

        ax2.axhline(1,c='k',ls='--')
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_yscale('log')
        ax2.set_xlabel("AUTO Color")

        if i%len(_fnames[0])!=0:
            _ = [tick.set_visible(False) for tick in ax1.get_yticklabels()+ax2.get_yticklabels()]

    fig.text(0.02,0.5,"AUTO/APER flux",color='k',va='center',ha='center',fontsize=16,fontweight=600,rotation=90)
    # fig.savefig("photom_offsets.png")

def main5():

    catalog = fitsio.getdata("final_cats/final_catalog_errfix.fits")

    fnames = [x for x in useful.sorted_pivot_l if "irac" not in x]
    _fnames = np.array_split(fnames,4)
    fig = plt.figure(figsize=(15,12),dpi=75)
    fig.subplots_adjust(left=0.07,right=0.98,top=0.95,bottom=0.05,wspace=0,hspace=0.5)
    ogs = gridspec.GridSpec(4,len(_fnames[0]))

    for i,fname in enumerate(fnames):

        print( fname)
        igs = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=ogs[i],wspace=0,hspace=0)
        ax1 = fig.add_subplot(igs[0])

        cond  = (catalog['FLUX_AUTO_%s'%fname]>0) & (catalog['FLUX_APER_%s'%fname][:,2]>0) & (catalog['MAG_APER_%s'%fname][:,2]>0) & (catalog['MAGERR_APER_%s'%fname][:,2]>0)

        color1  = catalog["FLUX_RADIUS_video_ks"][:,1][cond]# - catalog["MAG_APER_%s"%fname2][:,2][cond1]
        offset1 = catalog["FLUX_AUTO_%s"%fname][cond]/catalog["FLUX_APER_%s"%fname][:,2][cond]
        offset1 /= catalog["OFFSET_FLUX"][cond][:,2]

        instr,filt = fname.split("_")

        ax1.scatter(color1,offset1,c=useful.fcolor_dict[instr][filt],s=3,alpha=0.1)
        # ax2.scatter(color2,offset2,c=useful.fcolor_dict[instr][filt],s=3,alpha=0.1)

        ax1.text(0.02,0.97,fname,va='top',ha='left',fontsize=14,fontweight=600,color=useful.fcolor_dict[instr][filt],transform=ax1.transAxes)

        ax1.axhline(1,c='k',ls='--')
        ax1.set_xlim(10,100)
        ax1.set_ylim(0,5)
        # ax1.set_yscale('log')
        ax1.set_xlabel(fname)
        # ax1.xaxis.set_label_position('top')
        # ax1.xaxis.tick_top()

        # ax2.axhline(1,c='k',ls='--')
        # ax2.set_xlim(ax1.get_xlim())
        # ax2.set_ylim(ax1.get_ylim())
        # ax2.set_yscale('log')
        # ax2.set_xlabel("%s-%s"%(fname2,fname3))

        if i%len(_fnames[0])!=0:
            _ = [tick.set_visible(False) for tick in ax1.get_yticklabels()]

    fig.text(0.02,0.5,"AUTO/APER flux",color='k',va='center',ha='center',fontsize=16,fontweight=600,rotation=90)
    # fig.savefig("photom_offsets.png")

def main6():

    _catalog = fitsio.getdata("final_cats/final_catalog_errfix.fits")

    fig,axes = plt.subplots(5,1,figsize=(12,10),dpi=75)
    fig.subplots_adjust(left=0.05,right=0.85,bottom=0.05,top=0.98,hspace=0,wspace=0)

    fnames = [x for x in useful.sorted_pivot_l if "irac" not in x]
    colors = plt.cm.gist_rainbow_r(np.linspace(0,0.95,len(fnames)))

    for ax,dbin,size_cut in zip(axes,[0.01,0.02,0.05,0.1,0.1],[0,5,10,25,50]):

        catalog = _catalog[_catalog["FLUX_RADIUS_video_ks"][:,1] > size_cut]

        for fname,color in zip(fnames,colors):

            cond = (catalog[   'FLUX_AUTO_%s'%fname]!=-99.) & (catalog[   'FLUX_APER_%s'%fname][:,2]!=-99.) & \
                   (catalog[   'FLUX_AUTO_%s'%fname]!= 0.0) & (catalog[   'FLUX_APER_%s'%fname][:,2]!= 0.0) & \
                   (catalog['FLUXERR_AUTO_%s'%fname]!=-99.) & (catalog['FLUXERR_APER_%s'%fname][:,2]!=-99.) & \
                   (catalog['FLUXERR_AUTO_%s'%fname]!= 0.0) & (catalog['FLUXERR_APER_%s'%fname][:,2]!= 0.0)

            offset_filt = catalog["FLUX_AUTO_%s"%fname][cond]/catalog["FLUX_APER_%s"%fname][:,2][cond]
            offset_orig = calc_opt_nir_offset(catalog[cond],aper_num=2)
            # offset_orig = catalog["OFFSET_FLUX"][:,2][cond]

            diff = offset_filt / offset_orig

            bins = 10**np.arange(-5-dbin/2,5+dbin/2,dbin)
            binc = 0.5*(bins[1:] + bins[:-1])
            hist = np.histogram(diff,bins=bins)[0]
            hist = hist / float(max(hist))
            ax.plot(binc,hist,color=color,lw=2,alpha=0.8,label=fname)

            _ = [label.set_visible(False) for label in ax.get_yticklabels()]
            if ax!=axes[-1]: _ = [label.set_visible(False) for label in ax.get_xticklabels()]

        ax.axvline(1,lw=0.8,ls='--')
        ax.set_xscale("log")
        ax.set_xlim(1e-1,1e1)

    ax.set_xlabel("$o_{flux,filt} / o_{flux}$")

    leg = axes[2].legend(loc="center left",fontsize=12,ncol=1,scatterpoints=0,markerscale=0,fancybox=True,frameon=False,handlelength=0,handletextpad=0,bbox_to_anchor=(1,0.5))
    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        txt.set_color(hndl.get_color())
        txt.set_fontproperties(FontProperties(size=12,weight=600))
        hndl.set_visible(False)

def main7():

    catalog = fitsio.getdata("final_cats/final_catalog_errfix.fits")

    offset_new  = calc_opt_nir_offset(catalog,aper_num=2)
    offset_orig = catalog["OFFSET_FLUX"][:,2]

    fig,ax = plt.subplots(1,1,figsize=(12,10),dpi=75,tight_layout=True)

    x = catalog["FLUX_RADIUS_video_ks"][:,1]
    diff = offset_new / offset_orig

    binsx = 10**np.arange(-3,3,0.1)
    binsy = np.arange(0,5,0.05)
    bincx = 0.5*(binsx[1:] + binsx[:-1])
    bincy = 0.5*(binsy[1:] + binsy[:-1])
    gridy, gridx = np.meshgrid(bincy,bincx)
    hist2d = np.histogram2d(x,diff,bins=[binsx,binsy])[0]
    hist2d = np.ma.masked_array(hist2d,mask=hist2d==0)
    hist2d = np.ma.log10(hist2d)
    ax.pcolormesh(gridx,gridy,hist2d,cmap=plt.cm.inferno)

    # ax.scatter(catalog["FLUX_RADIUS_video_ks"][:,1], diff, c='k', lw=0, s=3, alpha=0.5)
    ax.axhline(1,c='k',lw=0.8)
    # ax.set_yscale('log')
    ax.set_xscale('log')
    # ax.set_xlim(1e0,1e3)
    ax.set_ylim(1e-4,1e4)

def temp():

    catalog = fitsio.getdata("final_cats/final_catalog_errfix.fits")

    offset = calc_opt_nir_offset(catalog,aper_num=2)
    print( len(offset[offset==-99.]))

    cond = ~np.isfinite(offset)

    np.savetxt("temp.reg",np.vstack((catalog["RA"][cond],catalog["DEC"][cond])).T,fmt="circle(%.10f,%.10f,2\")",header='fk5',comments='')

def weird_offset_test(stars_only=False):
    """
    Test to check for objects with weirdly large offsets
    Trying to check if outlier rejection for some of the filters
    where the offset goes bonkers works or not
    """

    catalog = fitsio.getdata("final_cats/final_catalog_errfix.fits")
    final_cat = fitsio.getdata("final_cats/sxds_catalog_v1.5.fits")

    if stars_only:

        size_cut = [2.7,3.3]
        mag_cut = [20,22]

        # fig,ax = plt.subplots(1,1,figsize=(10,8),dpi=75,tight_layout=True)
        # for i in range(4):
        #     sex_cat = fitsio.getdata("final_cats/catalog{:d}_matched_hsc_g.fits".format(i+1))
        #     ax.scatter(sex_cat["FLUX_RADIUS"][:,1],sex_cat["MAG_AUTO"],c='k',s=2,alpha=0.2)
        # verts = [[size_cut[0],mag_cut[0]],[size_cut[0],mag_cut[1]],[size_cut[1],mag_cut[1]],[size_cut[1],mag_cut[0]],[size_cut[0],mag_cut[0]]]
        # patch = ax.add_patch(Polygon(verts,color='r',lw=2,alpha=0.5,closed=True))
        # ax.set_xlim(0,10)
        # ax.set_ylim(28,15)

        class_star = np.zeros(0)
        cond = np.zeros(len(catalog),dtype=bool)
        for i in range(4):
            sex_cat = fitsio.getdata("final_cats/catalog{:d}_matched_hsc_g.fits".format(i+1))
            cond_star = (mag_cut[0]<sex_cat["MAG_AUTO"]) & (sex_cat["MAG_AUTO"]<mag_cut[1]) & \
                        (size_cut[0]<sex_cat["FLUX_RADIUS"][:,1]) & (sex_cat["FLUX_RADIUS"][:,1]<size_cut[1])
            star_id = sex_cat["NUMBER"][cond_star]
            cond_num  = catalog["cutout_num"] == (i+1)
            cond_star = np.in1d(catalog["cutout_id"],star_id)
            cond = cond | (cond_num & cond_star)
        print("Selecting {:d} stars.".format(np.sum(cond)))
        catalog = catalog[cond]
        final_cat = final_cat[cond]

    fnames = []
    for instr in useful.instr_used_list[:-1]:
        for filt in useful.filters[instr]:
            fnames.append("_".join([instr,filt]))

    f_ratio,f_error = np.ma.zeros((2,len(catalog),len(fnames),5),dtype=float)
    f_ratio.mask,f_error.mask = np.zeros((2,len(catalog),len(fnames),5),dtype=bool)
    offset_flux,offset_mag = np.zeros((2,len(catalog),5)) - 99.

    # for aper_num in range(len(useful.apersizes)):
    if True:

        aper_num = 1

        for i,fname in enumerate(fnames):

            instr,filt = fname.split("_")
            sys.stdout.write("\rCalculating AUTO-APER offset for %s:%s-aper#%i ... \033[K" % (instr,filt,aper_num+1))
            sys.stdout.flush()

            # Just a simple check for if FLUX > 0
            cond_f = (catalog['FLUX_AUTO_%s_%s'%(instr,filt)] > 0.0) & (catalog[   'FLUX_APER_%s_%s'%(instr,filt)][:,aper_num] > 0.0)

            # Check if SN > 3 sigma
            # cond_f = (catalog['FLUX_AUTO_%s_%s'%(instr,filt)] / catalog['FLUXERR_AUTO_%s_%s'%(instr,filt)] > 3.0) & \
            #          (catalog['FLUX_APER_%s_%s'%(instr,filt)][:,aper_num] / catalog['FLUXERR_APER_%s_%s'%(instr,filt)][:,aper_num] > 3.0)

            f_ratio[cond_f,i,aper_num] = catalog[   'FLUX_AUTO_%s_%s'%(instr,filt)][cond_f]    / catalog[   'FLUX_APER_%s_%s'%(instr,filt)][cond_f,aper_num]
            f_error[cond_f,i,aper_num] = catalog['FLUXERR_AUTO_%s_%s'%(instr,filt)][cond_f]**2 + catalog['FLUXERR_APER_%s_%s'%(instr,filt)][cond_f,aper_num]**2
            f_ratio.mask[:,i,aper_num] = ~cond_f
            f_error.mask[:,i,aper_num] = ~cond_f

        print("done!")

        ############################
        ### ADDED to fix the problem
        ############################
        # f_ratio_median = np.ma.median(f_ratio[:,:,aper_num],axis=-1)
        # f_ratio_stdev  = np.ma.std(   f_ratio[:,:,aper_num],axis=-1)
        f_ratio_16ile,f_ratio_84ile  = np.nanpercentile(f_ratio[:,:,aper_num].filled(np.NaN),[16,84],axis=-1)

        for i,fname in enumerate(fnames):
            cond = (f_ratio_16ile > f_ratio[:,i,aper_num]) | \
                   (f_ratio[:,i,aper_num] > f_ratio_84ile)
            # cond = np.ma.abs(f_ratio_median-f_ratio[:,i,aper_num]) > f_ratio_stdev
            f_ratio.mask[cond,i,aper_num] = True
            f_error.mask[cond,i,aper_num] = True
        ############################

        offset_flux[:,aper_num] = np.ma.average(f_ratio[:,:,aper_num],weights=1/f_error[:,:,aper_num],axis=-1).filled(-99.)

        cond = (offset_flux[:,aper_num] > 0)
        offset_mag[cond,aper_num] = -2.5*np.log10(offset_flux[cond,aper_num])

        check = np.abs(offset_flux[:,aper_num]-final_cat["OFFSET_FLUX"][:,aper_num])/final_cat["OFFSET_FLUX"][:,aper_num]
        print(np.sum(check>0.1),np.sum(check>0.01),np.sum(check>0.001),np.sum(check>0.0001))

        fig,ax = plt.subplots(1,1,figsize=(10,9),dpi=75,tight_layout=True)
        ax.scatter(final_cat["OFFSET_FLUX"][:,aper_num],offset_flux[:,aper_num],s=2,c='k',alpha=0.2)
        ax.plot([1e-5,1e5],[1e-5,1e5],c='k',lw=0.8,ls='--')
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_aspect(1.)
        ax.set_xlim(1e-1,5e3)
        ax.set_ylim(1e-1,5e3)
        ax.set_xlabel("Old Offset")
        ax.set_ylabel("New Offset")

        fig,ax = plt.subplots(1,1,figsize=(20,8),dpi=75,tight_layout=True)

        bins = np.arange(0.005,5,0.005/2)
        bins = np.sort(np.concatenate((bins,-bins)))
        binc = 0.5*(bins[1:]+bins[:-1])

        for i,fname in enumerate(fnames):

            instr,filt = fname.split("_")
            median  = np.ma.median(f_ratio[:,:,aper_num],axis=-1)
            average = np.ma.average(f_ratio[:,:,aper_num],weights=1/f_error[:,:,aper_num],axis=-1)
            # compare = f_ratio[:,i,aper_num]/median
            # compare = f_ratio[:,i,aper_num]/average
            compare = f_ratio[:,i,aper_num]/offset_flux[:,aper_num]

            hist = np.histogram(np.log10(compare.compressed()),bins=bins)[0]
            hist = hist / len(catalog)
            ax.plot(binc,hist,c=useful.fcolor_dict[instr][filt],lw=1.2,label=fname.replace("_",":"))

        ax.set_yscale("log")
        ax.set_xlim(-0.1,0.1)
        ax.axvline(0,c='k',ls='--',lw=1.2)

        leg = ax.legend(fontsize=16,ncol=5,loc="best",framealpha=0,handlelength=0,handletextpad=0)

        for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
            txt.set_fontweight(600)
            instr,filt = txt.get_text().split(':')
            txt.set_color(useful.fcolor_dict[instr][filt])
            hndl.set_visible(False)

def offset_less_than_one_test():

    newcat = fitsio.getdata("final_cats/final_catalog_errfix.fits")
    # oldcat = fitsio.getdata("final_cats/old_cats_v1.5/final_catalog_errfix.fits")

    size_cut = [2.7,3.3]
    mag_cut = [20,22]

    # fig,ax = plt.subplots(1,1,figsize=(10,8),dpi=75,tight_layout=True)
    # for i in range(4):
    #     sex_cat = fitsio.getdata("final_cats/catalog{:d}_matched_hsc_g.fits".format(i+1))
    #     ax.scatter(sex_cat["FLUX_RADIUS"][:,1],sex_cat["MAG_AUTO"],c='k',s=2,alpha=0.2)
    # verts = [[size_cut[0],mag_cut[0]],[size_cut[0],mag_cut[1]],[size_cut[1],mag_cut[1]],[size_cut[1],mag_cut[0]],[size_cut[0],mag_cut[0]]]
    # patch = ax.add_patch(Polygon(verts,color='r',lw=2,alpha=0.5,closed=True))
    # ax.set_xlim(0,10)
    # ax.set_ylim(28,15)

    class_star = np.zeros(0)
    cond = np.zeros(len(newcat),dtype=bool)
    for i in range(4):
        sex_cat = fitsio.getdata("final_cats/catalog{:d}_matched_hsc_g.fits".format(i+1))
        cond_star = (mag_cut[0]<sex_cat["MAG_AUTO"]) & (sex_cat["MAG_AUTO"]<mag_cut[1]) & \
                    (size_cut[0]<sex_cat["FLUX_RADIUS"][:,1]) & (sex_cat["FLUX_RADIUS"][:,1]<size_cut[1])
        star_id = sex_cat["NUMBER"][cond_star]
        cond_num  = newcat["cutout_num"] == (i+1)
        cond_star = np.in1d(newcat["cutout_id"],star_id)
        cond = cond | (cond_num & cond_star)
    print("Selecting {:d} stars.".format(np.sum(cond)))

    fig,axes = plt.subplots(2,2,figsize=(18,10),dpi=75,tight_layout=True)
    axes = axes.flatten()

    for i,(ax,label) in enumerate(zip(axes,["1\"","2\"","3\"","4\""])):

        binsx = np.arange(0,50,0.05)
        binsy = 10**np.arange(-5,5,0.01)
        bincx = 0.5*(binsx[1:]+binsx[:-1])
        bincy = 0.5*(binsy[1:]+binsy[:-1])
        hist = np.histogram2d(newcat["MAG_AUTO_hsc_g"],newcat["OFFSET_FLUX"][:,i],bins=[binsx,binsy])[0]
        hist = np.ma.MaskedArray(hist,mask=hist<3)
        # hist = np.ma.log10(hist)
        im = ax.pcolormesh(bincx,bincy,hist.T,cmap=plt.cm.Greys,vmin=0,vmax=500)

        ax.scatter(newcat["MAG_AUTO_hsc_g"][cond],newcat["OFFSET_FLUX"][cond,i],c='r',s=10,lw=0,alpha=0.5)
        ax.axhline(1,c='k',ls='--',lw=0.5)

        ax.set_xlabel("APER HSC-g mag [{:}]".format(label),fontsize=18)
        ax.set_ylabel("FLUX_AUTO / FLUX_APER [{:}]".format(label),fontsize=18)
        ax.set_xlim(17,28)
        ax.set_ylim(5e-1,5e0)
        ax.set_yscale("log")
        [_.set_fontsize(16) for _ in ax.get_xticklabels()+ax.get_yticklabels()]

    fig.savefig("final_cats/plots/chk_new_offset.png")

def check_new_offset():

    newcat = fitsio.getdata("final_cats/final_catalog_errfix.fits")
    oldcat = fitsio.getdata("final_cats/old_cats_v1.5/final_catalog_errfix.fits")

    size_cut = [2.7,3.3]
    mag_cut = [20,22]
    cond = np.zeros(len(newcat),dtype=bool)
    for i in range(4):
        sex_cat = fitsio.getdata("final_cats/catalog{:d}_matched_hsc_g.fits".format(i+1))
        cond_star = (mag_cut[0]<sex_cat["MAG_AUTO"]) & (sex_cat["MAG_AUTO"]<mag_cut[1]) & \
                    (size_cut[0]<sex_cat["FLUX_RADIUS"][:,1]) & (sex_cat["FLUX_RADIUS"][:,1]<size_cut[1])
        star_id = sex_cat["NUMBER"][cond_star]
        cond_num  = newcat["cutout_num"] == (i+1)
        cond_star = np.in1d(newcat["cutout_id"],star_id)
        cond = cond | (cond_num & cond_star)
    print("Selecting {:d} stars.".format(np.sum(cond)))

    fig,ax = plt.subplots(1,1,figsize=(9,8),dpi=75,tight_layout=True)
    bins = 10**np.arange(-10,10,0.025)
    binc = 0.5*(bins[1:]+bins[:-1])
    hist = np.histogram2d(oldcat["OFFSET_FLUX"][:,1],newcat["OFFSET_FLUX"][:,1],bins=[bins,bins])[0]

    im = ax.pcolormesh(binc,binc,np.log10(hist.T),cmap=plt.cm.Greys,vmin=0,vmax=2,label="All objects")
    ax.scatter(oldcat["OFFSET_FLUX"][cond,1],newcat["OFFSET_FLUX"][cond,1],c='r',marker="x",s=25,alpha=0.2,label="Point-like objects (stars)")
    ax.axvline(1,c='k',ls='--',lw=0.5,alpha=0.9)
    ax.axhline(1,c='k',ls='--',lw=0.5,alpha=0.9)

    cax = fig.colorbar(im,format="10$^{%.1f}$")
    cax.set_label("N",fontsize=18)
    ax.set_xlabel("Old Flux Offset [2\"] (v1.5)",fontsize=18)
    ax.set_ylabel("New Flux Offset [2\"] (v1.6)",fontsize=18)
    ax.set_xlim(5e-2,1e3)
    ax.set_ylim(5e-1,1e3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.legend(fontsize=20)

    fig.savefig("final_cats/plots/chk_new_offset2.png")

def check_new_offset2():

    fnames = []
    for instr in useful.instr_used_list[:-1]:
        for filt in useful.filters[instr]:
            fnames.append("_".join([instr,filt]))

    newcat = fitsio.getdata("final_cats/final_catalog_errfix.fits")
    oldcat = fitsio.getdata("final_cats/old_cats_v1.5/final_catalog_errfix.fits")

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,10),dpi=75,tight_layout=True)
    ax1.text(0.99,0.97,"v1.5 catalog",fontsize=18,fontweight=600,va='top',ha='right',transform=ax1.transAxes)
    ax2.text(0.99,0.97,"v1.6 catalog",fontsize=18,fontweight=600,va='top',ha='right',transform=ax2.transAxes)

    bins = 10**np.arange(-5,5,0.025)
    binc = 0.5*(bins[1:]+bins[:-1])
    aper_num = 1

    for ax,catalog in zip([ax1,ax2],[oldcat,newcat]):

        for i,fname in enumerate(fnames):

            print(fname)
            instr,filt = fname.split("_")

            f_ratio = np.zeros(len(catalog))-999.
            cond_f          = (catalog['FLUX_AUTO_%s_%s'%(instr,filt)] > 0.0) & (catalog['FLUX_APER_%s_%s'%(instr,filt)][:,aper_num] > 0.0)
            f_ratio[cond_f] = catalog['FLUX_AUTO_%s_%s'%(instr,filt)][cond_f] / catalog['FLUX_APER_%s_%s'%(instr,filt)][cond_f,aper_num]
            compare = f_ratio/catalog["OFFSET_FLUX"][:,aper_num]

            hist = np.histogram(compare,bins=bins)[0]
            hist = hist / len(catalog)
            ax.plot(binc,hist,c=useful.fcolor_dict[instr][filt],lw=1.2,label=fname.replace("_",":"))

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim(1e-3,2e2)
        ax.set_ylim(1e-5,5e-1)
        ax.axvline(1,c='k',ls='--',lw=1.2)
        ax.set_xlabel("log($O_{filt}/O$)",fontsize=16)
        ax.set_ylabel("N",fontsize=16)

        leg = ax.legend(loc=2,fontsize=16,ncol=5,framealpha=0,handlelength=0,handletextpad=0)

        for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
            txt.set_fontweight(600)
            instr,filt = txt.get_text().split(':')
            txt.set_color(useful.fcolor_dict[instr][filt])
            hndl.set_visible(False)

    fig.savefig("final_cats/plots/chk_new_offset3.png")

if __name__ == '__main__':

    # main()
    # main5()
    # main6()
    # main7()
    # temp()
    # weird_offset_test(stars_only=False)
    # offset_less_than_one_test()
    # check_new_offset()
    check_new_offset2()
    plt.show()

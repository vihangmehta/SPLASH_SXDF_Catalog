import pdb
import numpy as np
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties

import useful

def calc_opt_nir_offset(catalog,aper_num=2):

    offset_flux, offset_mag = np.zeros(len(catalog))-99., np.zeros(len(catalog))-99.
    term1,term2 = np.zeros(len(catalog)),np.zeros(len(catalog))

    for instr in useful.instr_used_list[:-1]:
        
        for filt in useful.filters[instr]:
            
            print instr,filt
            cond_f = (catalog[   'FLUX_AUTO_%s_%s'%(instr,filt)] > 0.0) & (catalog[   'FLUX_ISO_%s_%s'%(instr,filt)] > 0.0)
            # cond_f = (catalog['FLUX_AUTO_%s_%s'%(instr,filt)] / catalog['FLUXERR_AUTO_%s_%s'%(instr,filt)] > 3.0) & \
            #          (catalog['FLUX_APER_%s_%s'%(instr,filt)][:,aper_num] / catalog['FLUXERR_APER_%s_%s'%(instr,filt)][:,aper_num] > 3.0)

            diff_f,wht_f = np.zeros(len(catalog)),np.zeros(len(catalog))
            diff_f[cond_f] = catalog['FLUX_AUTO_%s_%s'%(instr,filt)][cond_f] / catalog['FLUX_ISO_%s_%s'%(instr,filt)][cond_f]
            wht_f[cond_f] = 1./(catalog['FLUXERR_AUTO_%s_%s'%(instr,filt)][cond_f]**2 + catalog['FLUXERR_ISO_%s_%s'%(instr,filt)][cond_f]**2)
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

        print fname1

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

        print fname
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
    print len(offset[offset==-99.])
    
    cond = ~np.isfinite(offset)

    np.savetxt("temp.reg",np.vstack((catalog["RA"][cond],catalog["DEC"][cond])).T,fmt="circle(%.10f,%.10f,2\")",header='fk5',comments='')

if __name__ == '__main__':
    
    main()
    main5()
    main6()
    main7()
    # temp()
    plt.show()
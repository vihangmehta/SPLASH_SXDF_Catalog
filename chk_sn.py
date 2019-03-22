import numpy as np
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import useful

def calc_sn(mag,dmag):
    """
    Mag  = -2.5*log(S)
    dMag = -2.5*log(S) + 2.5*log(S+N)
         =  2.5*log((S+N)/S)
         =  2.5*log(1+N/S)
    """
    cond_99 = (dmag==-99.)
    cond_00 = (dmag==0)
    cond_nan = (dmag>10)
    sn = dmag * 0
    sn[cond_99] = -99.
    sn[cond_00] = 9999.
    sn[cond_nan]= -99.
    sn[~cond_00&~cond_99&~cond_nan] = 1./(10**(dmag[~cond_00&~cond_99&~cond_nan]/2.5) - 1)
    sn[np.abs(mag)==99.] = -99.
    return sn

def plot_contours(xdata,ydata,axis,color,cmap,label=None,fill=True,alpha=1):

    binsx = np.arange(10,40,0.5)
    binsy = 10**np.arange(-5,5,0.1)
    binsy = np.sort(np.concatenate((-1*binsy,binsy)))
    bincx = 0.5*(binsx[1:] + binsx[:-1])
    bincy = 0.5*(binsy[1:] + binsy[:-1])
    gridy, gridx = np.meshgrid(bincy,bincx)
    hist2d = np.histogram2d(xdata,ydata,bins=[binsx,binsy])[0]
    hist2d = np.ma.masked_array(hist2d,mask=hist2d==0)
    hist2d = np.ma.log10(hist2d)
    if fill:
        axis.contourf(gridx,gridy,hist2d,cmap=cmap,lw=0,alpha=alpha)
    else:
        axis.contour(gridx,gridy,hist2d,colors='k',linewidths=0.5,alpha=alpha)
    if label and fill:
        axis.text(0.1,0.9,label,color=color,fontsize=18,fontweight=600,transform=axis.transAxes)
    # axis.axhline(1,c='k',lw=0.5,ls='--')

def compare_sn_ext_aper():

    fig,axes = plt.subplots(3,5,figsize=(15,8),dpi=75,sharex=True,sharey=True)
    fig.subplots_adjust(left=0.05,right=0.98,bottom=0.05,top=0.96,wspace=0,hspace=0)

    colors  = ['red','blue','green','orange','purple']
    cmaps   = [plt.cm.Reds,plt.cm.Blues,plt.cm.Greens,plt.cm.Oranges,plt.cm.Purples]

    for catname,fill in zip(['final_cats/final_catalog_errfix.extra.fits','final_cats/final_catalog.extra.fits'],[True,False]):

        catalog = fitsio.getdata(catname)

        xlabels = ['MAG_APER_supcam_b','MAG_APER_supcam_v','MAG_APER_supcam_r','MAG_APER_supcam_i']
        ylabels = [   'F08_MAG_APER_b',   'F08_MAG_APER_v',   'F08_MAG_APER_r',   'F08_MAG_APER_i']

        for i,(xlabel,ylabel,c,cmap) in enumerate(zip(xlabels,ylabels,colors,cmaps)):

            snx  = calc_sn(catalog[xlabel][:,2],catalog[xlabel.replace('MAG','MAGERR')][:,2])
            sny  = calc_sn(catalog[ylabel][:,1],catalog[ylabel.replace('MAG','MAGERR')][:,1])
            cond = (snx!=0) & (snx!=-99) & (snx!=9999) & (sny!=-99) & (sny!=9999)
            plot_contours(catalog[xlabel][:,2][cond],sny[cond]/snx[cond],axis=axes[0,i],color=c,cmap=cmap,label=xlabel[9:],fill=fill)

        xlabels = ['MAG_APER_video_z','MAG_APER_video_y','MAG_APER_video_j','MAG_APER_video_h','MAG_APER_video_ks']
        ylabels = ['VIDEO_MAG_APER_z','VIDEO_MAG_APER_y','VIDEO_MAG_APER_j','VIDEO_MAG_APER_h','VIDEO_MAG_APER_ks']
        zlabels = ['VIDEO_ERRFIX_MAG_APER_z','VIDEO_ERRFIX_MAG_APER_y','VIDEO_ERRFIX_MAG_APER_j','VIDEO_ERRFIX_MAG_APER_h','VIDEO_ERRFIX_MAG_APER_ks']

        for i,(xlabel,ylabel,zlabel,c,cmap) in enumerate(zip(xlabels,ylabels,zlabels,colors,cmaps)):

            snx  = calc_sn(catalog[xlabel][:,2],catalog[xlabel.replace('MAG','MAGERR')][:,2])
            sny  = calc_sn(catalog[ylabel][:,2],catalog[ylabel.replace('MAG','MAGERR')][:,2])
            snz  = calc_sn(catalog[ylabel][:,2],catalog[zlabel.replace('MAG','MAGERR')][:,2])
            condy = (snx!=0) & (snx!=-99) & (snx!=9999) & (sny!=-99) & (sny!=9999)
            condz = (snx!=0) & (snx!=-99) & (snx!=9999) & (snz!=-99) & (snz!=9999)
            plot_contours(catalog[xlabel][:,2][condy],sny[condy]/snx[condy],axis=axes[1,i],color=c,cmap=cmap,label=xlabel[9:],fill=fill)
            plot_contours(catalog[xlabel][:,2][condz],snz[condz]/snx[condz],axis=axes[2,i],color=c,cmap=cmap,label=xlabel[9:],fill=fill)

    axes[1,0].set_xlabel('Magnitude')
    axes[1,0].set_ylabel('$\Delta$(SN)')
    axes[1,0].set_xlim(15.5,31.5)
    axes[1,0].set_ylim(1e-1,1e2)
    axes[1,0].set_yscale('log')
    
    # for ax in [ax1,ax2,ax3]:
    #     ax.legend(loc='best',fontsize=9)

def compare_sn_ext_auto():

    fig,axes = plt.subplots(3,5,figsize=(15,8),dpi=75,sharex=True,sharey=True)
    fig.subplots_adjust(left=0.05,right=0.98,bottom=0.05,top=0.96,wspace=0,hspace=0)

    colors  = ['red','blue','green','orange','purple']
    cmaps   = [plt.cm.Reds,plt.cm.Blues,plt.cm.Greens,plt.cm.Oranges,plt.cm.Purples]

    for catname,fill in zip(['final_cats/final_catalog_errfix.extra.fits','final_cats/final_catalog.extra.fits'],[True,False]):

        catalog = fitsio.getdata(catname)

        xlabels = ['MAG_AUTO_supcam_b','MAG_AUTO_supcam_v','MAG_AUTO_supcam_r','MAG_AUTO_supcam_i']
        ylabels = [   'F08_MAG_b',   'F08_MAG_v',   'F08_MAG_r',   'F08_MAG_i']

        for i,(xlabel,ylabel,c,cmap) in enumerate(zip(xlabels,ylabels,colors,cmaps)):

            snx  = calc_sn(catalog[xlabel],catalog[xlabel.replace('MAG','MAGERR')])
            sny  = calc_sn(catalog[ylabel],catalog[ylabel.replace('MAG','MAGERR')])
            cond = (snx!=0) & (snx!=-99) & (snx!=9999) & (sny!=-99) & (sny!=9999)
            plot_contours(catalog[xlabel][cond],sny[cond]/snx[cond],axis=axes[0,i],color=c,cmap=cmap,label=xlabel[9:],fill=fill)

        xlabels = ['MAG_AUTO_video_z','MAG_AUTO_video_y','MAG_AUTO_video_j','MAG_AUTO_video_h','MAG_AUTO_video_ks']
        ylabels = ['VIDEO_MAG_z','VIDEO_MAG_y','VIDEO_MAG_j','VIDEO_MAG_h','VIDEO_MAG_ks']
        zlabels = ['VIDEO_ERRFIX_MAG_z','VIDEO_ERRFIX_MAG_y','VIDEO_ERRFIX_MAG_j','VIDEO_ERRFIX_MAG_h','VIDEO_ERRFIX_MAG_ks']

        for i,(xlabel,ylabel,zlabel,c,cmap) in enumerate(zip(xlabels,ylabels,zlabels,colors,cmaps)):

            snx  = calc_sn(catalog[xlabel],catalog[xlabel.replace('MAG','MAGERR')])
            sny  = calc_sn(catalog[ylabel],catalog[ylabel.replace('MAG','MAGERR')])
            snz  = calc_sn(catalog[ylabel],catalog[zlabel.replace('MAG','MAGERR')])
            condy = (snx!=0) & (snx!=-99) & (snx!=9999) & (sny!=-99) & (sny!=9999)
            condz = (snx!=0) & (snx!=-99) & (snx!=9999) & (snz!=-99) & (snz!=9999)
            plot_contours(catalog[xlabel][condy],sny[condy]/snx[condy],axis=axes[1,i],color=c,cmap=cmap,label=xlabel[9:],fill=fill)
            plot_contours(catalog[xlabel][condz],snz[condz]/snx[condz],axis=axes[2,i],color=c,cmap=cmap,label=xlabel[9:],fill=fill)

    axes[1,0].set_xlabel('Magnitude')
    axes[1,0].set_ylabel('$\Delta$(SN)')
    axes[1,0].set_xlim(15.5,31.5)
    axes[1,0].set_ylim(1e-1,1e2)
    axes[1,0].set_yscale('log')
    
    # for ax in [ax1,ax2,ax3]:
    #     ax.legend(loc='best',fontsize=9)

def compare_sn_int(aper_num=2,fixed_version=True):

    catalog = fitsio.getdata('final_cats/final_catalog_errfix.fits')
    if not fixed_version:
        _catalog = fitsio.getdata('final_cats/final_catalog.fits')
        for x in catalog.dtype.names:
            if "FLUXERR_" in x:
                catalog[x] = _catalog[x]

    fnames = useful.fnames #[x for x in useful.fnames if "irac" not in x]
    _fnames = np.array_split(fnames,4)
    fig = plt.figure(figsize=(18,12),dpi=75)
    fig.subplots_adjust(left=0.05,right=0.98,top=0.98,bottom=0.05,wspace=0,hspace=0.1)
    ogs = gridspec.GridSpec(4,len(_fnames[0]))        

    for i,fname in enumerate(fnames):

        instr,filt = fname.split('_')
        fcolor = useful.fcolor_dict[instr][filt]

        igs = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=ogs[i],wspace=0,hspace=0,height_ratios=[1,3])
        ax2 = fig.add_subplot(igs[1])
        ax1 = fig.add_subplot(igs[0],sharex=ax2)

        if "irac" not in fname:
            flux    = catalog["FLUX_APER_%s"%fname][:,aper_num]
            fluxerr = catalog["FLUXERR_APER_%s"%fname][:,aper_num]
            mag     = catalog["MAG_APER_%s"%fname][:,aper_num]
            magerr  = catalog["MAGERR_APER_%s"%fname][:,aper_num]
        else:
            flux    = catalog["FLUX_TOT_%s"%fname]
            fluxerr = catalog["FLUXERR_TOT_%s"%fname]
            mag     = catalog["MAG_TOT_%s"%fname]
            magerr  = catalog["MAGERR_TOT_%s"%fname]

        cond_flux = (flux>0)
        cond_non  = (magerr==-1)

        bins = np.arange(0,50,0.1)
        binc = 0.5*(bins[1:]+bins[:-1])
        dbin = np.diff(bins)
        hist1 = np.histogram(-2.5*np.log10(flux[cond_flux]) + useful.zp,bins=bins)[0]
        hist1 = hist1 / float(np.max(hist1))
        ax1.bar(binc,hist1,width=dbin,color=fcolor,lw=0)
        hist2 = np.histogram(mag[cond_non],bins=bins)[0]
        hist2 = hist2 / float(np.max(hist2))
        ax1.step(binc,hist2,where='mid',color='k',lw=1.2)

        ax1.set_ylim(0,1.1*np.max((hist1,hist2)))
        ax1.set_xlim(19.5,28.5)
        ax1.text(0.05,0.95,"%s %s"%(instr,filt),va='top',ha='left',fontsize=14,fontweight=400,color=fcolor,transform=ax1.transAxes)

        plot_contours(-2.5*np.log10(flux[cond_flux]) + useful.zp,flux[cond_flux]/fluxerr[cond_flux],
                        axis=ax2,color=fcolor,cmap=plt.cm.Greys,label=None,fill=True,alpha=0.8)

        ulims = mag[cond_non]
        ax2.axvspan(*np.percentile(ulims,[16,84]),color=fcolor,alpha=0.5)
        ax2.axvline(np.median(ulims),lw=1,color=fcolor)
        ax2.axvline(np.min(ulims),lw=0.5,color=fcolor,ls='--')
        ax2.axvline(np.max(ulims),lw=0.5,color=fcolor,ls='--')

        ax2.set_ylim(1e-1,1e2)
        ax2.axhline(3,c='k',lw=0.5,ls='--')
        ax2.set_yscale('log')

        if i>=len(fnames)-len(_fnames[0]):ax2.set_xlabel('Magnitude',fontsize=9)
        if i%len(_fnames[0])==0: ax2.set_ylabel('SN',fontsize=9)
        else: _ = [label.set_visible(False) for label in ax2.get_yticklabels()]

        _ = [label.set_visible(False) for label in ax1.get_xticklabels()+ax1.get_yticklabels()]

    # fig.savefig('errors/plots/err_analysis_errfix.png')

if __name__ == '__main__':
    
    # compare_sn_ext_aper()
    # compare_sn_ext_auto()
    compare_sn_int()
    plt.show()
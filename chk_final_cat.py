import os,sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.io.fits as fitsio

import useful
from mk_coverage_area import get_coverage_area
from collections import OrderedDict
from mk_mag_lim import get_mag_limits

mag_lims_r2 = {"hsc_g"    : 26.81,"hsc_r"    : 26.31,"hsc_i"    : 26.06,"hsc_z"    : 25.50,"hsc_y"    : 24.77,
               "supcam_b" : 27.32,"supcam_v" : 26.98,"supcam_r" : 26.78,"supcam_i" : 26.55,"supcam_z" : 25.81,
               "uds_j"    : 25.55,"uds_h"    : 25.00,"uds_k"    : 25.26,
               "video_z"  : 25.27,"video_y"  : 24.83,"video_j"  : 24.32,"video_h"  : 24.00,"video_ks" : 23.67,
               "cfht_u"   : 27.35,
               "cfhtls_u" : 25.72,"cfhtls_g" : 26.02,"cfhtls_r" : 25.39,"cfhtls_i" : 24.94,"cfhtls_z" : 23.81,
               "irac_1"   : 25.39,"irac_2"   : 25.13,"irac_3"   : 23.04,"irac_4"   : 22.90}

def get_zorder():
    _i = 50
    while _i >= 0:
        yield _i
        _i -= 1

def plot_photometry(cat_dir,catalog,mag_lims=False,number_counts=False):

    fig,axes = plt.subplots(4,1,figsize=(10,12),dpi=75,tight_layout=False,sharex=True)
    fig.subplots_adjust(left=0.11,right=0.84,bottom=0.07,top=0.98,wspace=0.0,hspace=0.0)

    dbin = 0.3
    bins = np.arange(10,35,dbin)
    binc = 0.5*(bins[1:]+bins[:-1])

    zorder = get_zorder()

    for instr in useful.instr_used_list[:-1]:
        for filt in useful.filters[instr]:

            print (instr,filt)

            if instr in ["hsc",]: ax = axes[0]
            if instr in ["cfht","supcam","uds"]: ax = axes[1]
            if instr in ["video",]: ax = axes[2]
            if instr in ["cfhtls",]: ax = axes[3]

            fname = "%s_%s"%(instr,filt)

            mag_prefix = "MAG_AUTO" if instr!="irac" else "MAG_TOT"
            mags = catalog["%s_%s_%s"%(mag_prefix,instr,filt)]
            merr = catalog["%s_%s_%s"%(mag_prefix.replace("MAG","MAGERR"),instr,filt)]
            cond = (merr > 0)
            mags = mags[cond]

            hist = np.histogram(mags,bins=bins)[0]
            hist = hist / dbin
            if number_counts: hist = hist / get_coverage_area(instr,filt)
            ax.plot(binc,hist,lw=2.5,color=useful.fcolor_dict[instr][filt],zorder=zorder.next())
            axes[1].plot(-99,-99,lw=2.5,color=useful.fcolor_dict[instr][filt],label="%s:%s"%(instr if instr!='cfht' else 'musubi',filt))

            ax.set_yscale("log")
            ax.set_xlim(17.9,28.9)

            if number_counts: ax.set_ylim(1e3,1.5e5)
            else: ax.set_ylim(1e3,8e5)

            _ = [tick.set_fontsize(16) for tick in ax.get_xticklabels()+ax.get_yticklabels()]

            ax.tick_params(axis='both',which='major',length=8,width=1.2)
            ax.tick_params(axis='both',which='minor',length=5,width=1.)

            # if mag_lims:
            #       ax.axvline(mag_lims_r2[fname],lw=1.5,ls='--',color=useful.fcolor_dict[instr][filt])
            #     mag_lim = fitsio.getdata(os.path.join(cwd,"mag_limits","mag_lim_%s_%s_r%i.rebin.fits"%(instr,filt,2)))
            #     mag_lim = mag_lim[mag_lim!=0]
            #     mag_lim = mag_lim - 2.5*np.log10(5.)
            #     centers, centers_fin = get_mag_limits(mag_lim,fname=fname)
            #     ax.axvline(centers_fin,lw=1.5,ls='--',color=useful.fcolor_dict[instr][filt])

    if number_counts:
        fig.text(0.03,0.5,"N$_{gal}$ mag$^{-1}$ deg$^{-1}$",fontsize=25,ha='center',va='center',rotation=90)
    else:
        fig.text(0.03,0.5,"N$_{gal}$ mag$^{-1}$",fontsize=25,ha='center',va='center',rotation=90)

    for ax in axes:
        ax.tick_params(axis="both",direction='in',top="on",right="on")

    axes[-1].set_xlabel("Magnitude [AB]",fontsize=25)
    leg = axes[1].legend(fontsize=20,ncol=1,loc="center left",framealpha=0,
                     handlelength=0,handletextpad=0,markerscale=0,
                     bbox_to_anchor=[0.98,0])
    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        txt.set_fontweight(600)
        instr,filt = txt.get_text().split(':')
        txt.set_color(useful.fcolor_dict[instr if instr!='musubi' else 'cfht'][filt])
        hndl.set_visible(False)

    if number_counts: fig.savefig(os.path.join(cat_dir,'plots','number_counts.png'))
    else: fig.savefig(os.path.join(cat_dir,'plots','dist_photometry.png'))

# def plot_number_counts(cat_dir,catalog):

#     fig,axes = plt.subplots(7,5,figsize=(16,10),dpi=75,sharex=True,sharey=True,tight_layout=False)
#     fig.subplots_adjust(left=0.05,right=0.98,bottom=0.05,top=0.96,wspace=0.0,hspace=0.0)

#     bins = np.arange(10,35,0.5)
#     binc = 0.5*(bins[1:]+bins[:-1])
#     dbin = bins[1:] - bins[:-1]

#     for j,instr in enumerate(useful.instr_used_list):

#         for i,filt in enumerate(useful.filters[instr]):

#             fname = "%s_%s"%(instr,filt)
#             mag_prefix = "MAG_AUTO" if instr!="irac" else "MAG_TOT"
#             hist = np.histogram(catalog["%s_%s_%s"%(mag_prefix,instr,filt)],bins)[0]
#             hist = hist.astype(float) / dbin / get_coverage_area(instr,filt)
#             axes[j,i].plot(binc,hist,marker='o',markersize=5,color=useful.fcolor_dict[instr][filt])

#             axes[j,i].text(0.05,0.95,"%s:%s"%(instr,filt),color=useful.fcolor_dict[instr][filt],fontsize=18,fontweight=600,va='top',ha='left',transform=axes[j,i].transAxes)
#             axes[-1,i].set_xlabel("Magnitude")
#             axes[j,i].set_xlim(17,30)
#             axes[j,i].set_yscale('log')
#             axes[j,i].set_ylim(5e2,1e5)

#         _ = [axes[j,_].set_visible(False) for _ in range(i+1,5)]

#     fig.text(0.02,0.5,"N$_{gal}$ 0.5mag$^{-1}$ deg$^{-1}$",va='center',ha='center',rotation=90)
#     fig.savefig(os.path.join(cat_dir,'plots','number_counts.png'))

def check_photometry_int(cat_dir,catalog):

    print ("Plotting internal photometry comparison ... ",)

    sel_aper = 2
    apers    = [' [1"]', ' [2"]', ' [3"]', ' [4"]', ' [5"]']
    _xlabels = ['cfht_u',  'hsc_r'   ,'hsc_i'   ,'hsc_z'  ,'hsc_y'  ,'uds_j'  ,'uds_h' ]
    _ylabels = ['cfhtls_u','supcam_r','supcam_i','video_z','video_y','video_j','video_h']

    fig,_axes = plt.subplots(len(_xlabels),2,figsize=(16,10),dpi=75,sharex=True,sharey='row',tight_layout=False)
    fig.subplots_adjust(left=0.05,right=0.98,bottom=0.05,top=0.96,wspace=0.0,hspace=0.0)

    for axes,mag in zip(_axes.T,['MAG_AUTO','MAG_APER']):

        xlabels = ['%s_%s' % (mag,x) for x in _xlabels]
        ylabels = ['%s_%s' % (mag,y) for y in _ylabels]
        xcovlbs = ['COVERAGE_FLAG_%s' % x for x in _xlabels]
        ycovlbs = ['COVERAGE_FLAG_%s' % y for y in _ylabels]

        if 'APER' in mag:
            axes[0].set_title(mag.upper()+apers[sel_aper],fontsize=18,fontweight=800)
        else:
            axes[0].set_title(mag.upper(),fontsize=18,fontweight=800)

        for ax,xlabel,ylabel,xcovlbl,ycovlbl in zip(axes,xlabels,ylabels,xcovlbs,ycovlbs):

            divider = make_axes_locatable(ax)
            dax = divider.append_axes("right", size="20%", pad=0.0, sharey=ax)
            _ = [label.set_visible(False) for label in dax.get_xticklabels()]
            _ = [label.set_visible(False) for label in dax.get_yticklabels()]

            _catalog = catalog[(catalog[xcovlbl]==1) & (catalog[ycovlbl]==1)]
            if 'APER' in mag:
                x = _catalog[xlabel][:,sel_aper]
                y = _catalog[ylabel][:,sel_aper]
            else:
                x = _catalog[xlabel]
                y = _catalog[ylabel]
            cond = (np.abs(x)!=99.) & (np.abs(y)!=99.) & (x<35) & (y<35)
            diff = x - y

            cond_med = cond & (x<23)
            med, std = np.median(diff[cond_med]), np.std(diff[cond])

            binsx = np.arange(0,35,0.1)
            binsy = np.arange(-25,25,0.1)
            bincx = 0.5*(binsx[1:] + binsx[:-1])
            bincy = 0.5*(binsy[1:] + binsy[:-1])
            gridy, gridx = np.meshgrid(bincy,bincx)
            hist2d = np.histogram2d(x[cond],diff[cond],bins=[binsx,binsy])[0]
            hist2d = np.ma.masked_array(hist2d,mask=hist2d==0)
            hist2d = np.ma.log10(hist2d)
            ax.contourf(gridx,gridy,hist2d,cmap=plt.cm.inferno)

            text = "%s vs. %s" % (":".join(xlabel.split('_')[2:]),":".join(ylabel.split('_')[2:])) + "\nMedian offset: %.4f" % med
            ax.text(0.02,0.05,text,va='bottom',ha='left',transform=ax.transAxes)
            ax.axhline(0,c='k',ls='--',lw=0.5)
            ax.set_xlabel(mag+" #1")
            ax.set_ylabel("$\Delta$M")
            ax.set_ylim(-3,3)
            ax.set_xlim(11,30)

            bins = np.arange(-10,10,0.05)
            binc = 0.5*(bins[1:]+bins[:-1])
            hist = np.histogram(diff[cond],bins=bins)[0]
            hist = np.ma.masked_array(hist,mask=hist==0)
            hist = np.ma.log10(hist)
            dax.fill_betweenx(binc,0,hist,lw=0,color='k',alpha=0.5)
            dax.axhline(0,c='k',ls='--',lw=0.5)
            dax.set_xlim(np.ma.max(hist)-3,np.ma.max(hist)+0.1)

    fig.savefig(os.path.join(cat_dir,'plots','check_photometry_internal.png'))
    print ("done.")

def check_photometry_ext(cat_dir,catalog):

    uds_comp = (('UDS_MAG_u',    [  None ,     None ,    None ,  None ,'cfht_u','cfhtls_u',     None ]),
                ('UDS_MAG_b',    [  None ,'supcam_b',    None ,  None ,   None ,     None ,     None ]),
                ('UDS_MAG_v',    [  None ,'supcam_v',    None ,  None ,   None ,     None ,     None ]),
                ('UDS_MAG_r',    ['hsc_r','supcam_r',    None ,  None ,   None ,     None ,     None ]),
                ('UDS_MAG_i',    ['hsc_i','supcam_i',    None ,  None ,   None ,     None ,     None ]),
                ('UDS_MAG_z',    ['hsc_z','supcam_z','video_z',  None ,   None ,     None ,     None ]),
                ('UDS_MAG_j',    [  None ,     None ,'video_j','uds_j',   None ,     None ,     None ]),
                ('UDS_MAG_h',    [  None ,     None ,'video_h','uds_h',   None ,     None ,     None ]),
                ('UDS_MAG_k',    [  None ,     None ,    None ,'uds_k',   None ,     None ,     None ]),
                ('UDS_MAG_IRAC1',[  None ,     None ,    None ,  None ,   None ,     None ,  'irac_1']),
                ('UDS_MAG_IRAC2',[  None ,     None ,    None ,  None ,   None ,     None ,  'irac_2']))

    uds2cls_comp = (('UDS_S2CLS_MAG_APER_u',    [  None ,     None ,    None ,  None ,'cfht_u','cfhtls_u',    None ]),
                    ('UDS_S2CLS_MAG_APER_b',    [  None ,'supcam_b',    None ,  None ,   None ,     None ,    None ]),
                    ('UDS_S2CLS_MAG_APER_v',    [  None ,'supcam_v',    None ,  None ,   None ,     None ,    None ]),
                    ('UDS_S2CLS_MAG_APER_r',    ['hsc_r','supcam_r',    None ,  None ,   None ,     None ,    None ]),
                    ('UDS_S2CLS_MAG_APER_i',    ['hsc_i','supcam_i',    None ,  None ,   None ,     None ,    None ]),
                    ('UDS_S2CLS_MAG_APER_z',    ['hsc_z','supcam_z','video_z',  None ,   None ,     None ,    None ]),
                    ('UDS_S2CLS_MAG_APER_j',    [  None ,     None ,'video_j','uds_j',   None ,     None ,    None ]),
                    ('UDS_S2CLS_MAG_APER_h',    [  None ,     None ,'video_h','uds_h',   None ,     None ,    None ]),
                    ('UDS_S2CLS_MAG_APER_k',    [  None ,     None ,    None ,'uds_k',   None ,     None ,    None ]),)

    video_comp = (('VIDEO_MAG_z' ,['hsc_z','video_z' ,  None ]),
                  ('VIDEO_MAG_y' ,['hsc_y','video_y' ,  None ]),
                  ('VIDEO_MAG_j' ,[  None ,'video_j' ,'uds_j']),
                  ('VIDEO_MAG_h' ,[  None ,'video_h' ,'uds_h']),
                  ('VIDEO_MAG_ks',[  None ,'video_ks',  None ]))

    video_aper_comp = (('VIDEO_MAG_APER_z' ,['hsc_z','video_z' ,  None ]),
                       ('VIDEO_MAG_APER_y' ,['hsc_y','video_y' ,  None ]),
                       ('VIDEO_MAG_APER_j' ,[  None ,'video_j' ,'uds_j']),
                       ('VIDEO_MAG_APER_h' ,[  None ,'video_h' ,'uds_h']),
                       ('VIDEO_MAG_APER_ks',[  None ,'video_ks',  None ]))

    f08_auto_comp = (('F08_MAG_b',[  None ,'supcam_b',    None ]),
                     ('F08_MAG_v',[  None ,'supcam_v',    None ]),
                     ('F08_MAG_r',['hsc_r','supcam_r',    None ]),
                     ('F08_MAG_i',['hsc_i','supcam_i',    None ]),
                     ('F08_MAG_z',['hsc_z','supcam_z','video_z']))

    f08_aper_comp = (('F08_MAG_APER_b',[  None ,'supcam_b',    None ]),
                     ('F08_MAG_APER_v',[  None ,'supcam_v',    None ]),
                     ('F08_MAG_APER_r',['hsc_r','supcam_r',    None ]),
                     ('F08_MAG_APER_i',['hsc_i','supcam_i',    None ]),
                     ('F08_MAG_APER_z',['hsc_z','supcam_z','video_z']))

    uds_comp = OrderedDict(uds_comp)
    uds2cls_comp = OrderedDict(uds2cls_comp)
    video_comp = OrderedDict(video_comp)
    video_aper_comp = OrderedDict(video_aper_comp)
    f08_auto_comp = OrderedDict(f08_auto_comp)
    f08_aper_comp = OrderedDict(f08_aper_comp)

    for comp_cat,ncols,savename in zip([uds_comp,uds2cls_comp,video_comp,video_aper_comp,f08_auto_comp,f08_aper_comp],
                                       [7,7,3,3,3,3],
                                       ['UDS','UDS-2CLS','VIDEO','VIDEO_APER','F08_AUTO','F08_APER']):

        sys.stdout.write("\rPlotting photometry comparison with external catalog: %s ... "%savename)
        sys.stdout.flush()

        fig,_axes = plt.subplots(len(comp_cat),ncols,figsize=(3.5*ncols,len(comp_cat)),dpi=75,sharex=True,sharey='row',tight_layout=False)
        fig.subplots_adjust(left=0.05,right=0.98,bottom=0.05,top=0.9,wspace=0.0,hspace=0.0)
        fig.suptitle("Comparison with %s" % savename,fontsize=18,fontweight=800)

        for axes,comp_filt in zip(_axes,comp_cat):

            for ncol,(ax,filt) in enumerate(zip(axes,comp_cat[comp_filt])):

                divider = make_axes_locatable(ax)
                dax = divider.append_axes("right", size="20%", pad=0.0, sharey=ax)
                _ = [label.set_visible(False) for label in dax.get_xticklabels()+dax.get_yticklabels()]

                if filt is None:
                    ax.text(0.5,0.5,"No overlap",va='center',ha='center',fontsize=14,transform=ax.transAxes)
                    continue

                print (comp_filt.split('_')[-1])
                _axes[0,ncol].set_title(filt.split('_')[0].upper(),fontsize=15,fontweight=600)
                axes[0].set_ylabel(comp_filt.split('_')[-1],fontsize=15,fontweight=600,rotation=0)

                if '2CLS' in savename or 'F08_APER' in savename:
                    xlabel,ylabel = comp_filt,'MAG_APER_%s'%filt
                    x = catalog[xlabel][:,1] # [0,1] is [2",3"]
                    y = catalog[ylabel][:,2] # [0,1,2,3,4] is [1",2",3",4",5"]
                elif 'VIDEO_APER' in savename:
                    xlabel,ylabel = comp_filt,'MAG_APER_%s'%filt
                    x = catalog[xlabel][:,4] # [0,1,2,3,4] is [1",2",3",4",5"]
                    y = catalog[ylabel][:,2] # [0,1,2,3,4] is [1",2",3",4",5"]
                else:
                    xlabel,ylabel = comp_filt,'MAG_AUTO_%s'%filt
                    if 'irac' in filt: ylabel = "MAG_TOT_%s"%filt
                    x = catalog[xlabel]
                    y = catalog[ylabel]


                cond = (catalog['COVERAGE_FLAG_%s'%filt].astype(bool)) & \
                       (np.abs(x)!=99.) & (np.abs(y)!=99.) & (np.abs(x)!=999.) & (np.abs(y)!=999.) & (x<35) & (y<35)
                x,y  = x[cond],y[cond]
                diff = x - y

                cond_med = (x<np.percentile(x,0.25))
                med, std = np.median(diff[cond_med]), np.std(diff)

                binsx = np.arange(0,35,0.1)
                binsy = np.arange(-25,25,0.05)
                bincx = 0.5*(binsx[1:] + binsx[:-1])
                bincy = 0.5*(binsy[1:] + binsy[:-1])
                gridy, gridx = np.meshgrid(bincy,bincx)
                hist2d = np.histogram2d(x,diff,bins=[binsx,binsy])[0]
                hist2d = np.ma.masked_array(hist2d,mask=hist2d==0)
                hist2d = np.ma.log10(hist2d)
                ax.pcolormesh(gridx,gridy,hist2d,cmap=plt.cm.inferno)

                ax.text(0.02,0.05,"Offset: %.4f"%med,va='bottom',ha='left',transform=ax.transAxes)
                ax.axhline(0,c='k',ls='--',lw=0.5)
                #ax.set_xlabel("MAG#1")
                #ax.set_ylabel("$\Delta$M")
                ax.set_ylim(-5*std,+5*std)
                ax.set_xlim(11,30)

                bins = np.arange(-10,10,0.05)
                binc = 0.5*(bins[1:]+bins[:-1])
                hist = np.histogram(diff,bins=bins)[0]
                hist = np.ma.masked_array(hist,mask=hist==0)
                hist = np.ma.log10(hist)
                dax.fill_betweenx(binc,0,hist,lw=0,color='k',alpha=0.5)
                dax.axhline(0,c='k',ls='--',lw=0.5)
                dax.set_xlim(np.ma.max(hist)-3,np.ma.max(hist)+0.1)

                ax.set_ylim(-1.5,1.5)

        fig.savefig(os.path.join(cat_dir,'plots','check_photometry_external_%s.png' % savename))
        print ("done.")

def check_astrometry_ext(cat_dir,catalog):

    print ("Plotting astrometry comparison with PS1 catalog ... ")

    fig,axes = plt.subplots(4,1,figsize=(10,10),tight_layout=True)

    axes[0].scatter(catalog['RA'] ,(catalog['RA'] -catalog['PS1_RA' ])*3600,c='r',lw=0,s=3,alpha=0.1)
    axes[1].scatter(catalog['RA'] ,(catalog['DEC']-catalog['PS1_DEC'])*3600,c='g',lw=0,s=3,alpha=0.1)
    axes[2].scatter(catalog['DEC'],(catalog['RA'] -catalog['PS1_RA' ])*3600,c='g',lw=0,s=3,alpha=0.1)
    axes[3].scatter(catalog['DEC'],(catalog['DEC']-catalog['PS1_DEC'])*3600,c='b',lw=0,s=3,alpha=0.1)

    axes[0].axhline(0,c='k',lw=1,ls='--')
    axes[1].axhline(0,c='k',lw=1,ls='--')
    axes[2].axhline(0,c='k',lw=1,ls='--')
    axes[3].axhline(0,c='k',lw=1,ls='--')

    axes[0].set_xlabel('RA')
    axes[1].set_xlabel('RA')
    axes[2].set_xlabel('DEC')
    axes[3].set_xlabel('DEC')

    axes[0].set_ylabel('$\Delta$(RA) ["]')
    axes[1].set_ylabel('$\Delta$(DEC) ["]')
    axes[2].set_ylabel('$\Delta$(RA) ["]')
    axes[3].set_ylabel('$\Delta$(DEC) ["]')

    for ax in axes: ax.set_ylim(-0.5,0.5)

    fig.savefig(os.path.join(cat_dir,'plots','check_astrometry_PS1.png'))
    print ("done.")

if __name__ == '__main__':

    cwd = '/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/'
    cat_dir = os.path.join(cwd,'final_cats')

    # catalog = fitsio.getdata(os.path.join(cat_dir,'final_catalog.fits'))
    catalog = fitsio.getdata(os.path.join(cat_dir,'final_catalog.extra.fits'))
    # catalog = fitsio.getdata(os.path.join(cat_dir,'sxds_catalog_v1.5.fits'))

    # plot_photometry(cat_dir,catalog)
    # plot_photometry(cat_dir,catalog,mag_lims=True,number_counts=True)
    # plot_number_counts(cat_dir,catalog)

    # check_photometry_int(cat_dir,catalog)
    # check_photometry_ext(cat_dir,catalog)
    # check_astrometry_ext(cat_dir,catalog)

    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fitsio
from astropy.table import Table
from mpl_toolkits.axes_grid1 import make_axes_locatable

from useful import match_ra_dec

def match():

    catalog = fitsio.getdata("final_cats/final_catalog_errfix.fits")
    # catalog = fitsio.getdata("final_cats/old_cats_v1.5/final_catalog_errfix.fits")

    archive = Table.read("../CATALOGS/IRAC_Archive_SourceListQuery.tbl",format='ipac')
    spuds   = Table.read("../CATALOGS/IRAC_SpUDS_catalog.tbl",format='ipac')

    matched = np.recarray(len(catalog),dtype=[("ID",int),("RA",float),("DEC",float),
                                         ("FLUX_TOT_irac_1",float),("FLUXERR_TOT_irac_1",float),
                                         ("FLUX_TOT_irac_2",float),("FLUXERR_TOT_irac_2",float),
                                         ("FLUX_TOT_irac_3",float),("FLUXERR_TOT_irac_3",float),
                                         ("FLUX_TOT_irac_4",float),("FLUXERR_TOT_irac_4",float),
                                         ("FLUX_ARX_irac_1",float),("FLUXERR_ARX_irac_1",float),
                                         ("FLUX_ARX_irac_2",float),("FLUXERR_ARX_irac_2",float),
                                         ("FLUX_ARX_irac_3",float),("FLUXERR_ARX_irac_3",float),
                                         ("FLUX_ARX_irac_4",float),("FLUXERR_ARX_irac_4",float),
                                         ("FLUX_SpUDS_irac_1",float),("FLUXERR_SpUDS_irac_1",float),
                                         ("FLUX_SpUDS_irac_2",float),("FLUXERR_SpUDS_irac_2",float),
                                         ("FLUX_SpUDS_irac_3",float),("FLUXERR_SpUDS_irac_3",float),
                                         ("FLUX_SpUDS_irac_4",float),("FLUXERR_SpUDS_irac_4",float)])

    matched["ID"]                 = catalog["ID"]
    matched["RA"]                 = catalog["RA"]
    matched["DEC"]                = catalog["DEC"]
    matched["FLUX_TOT_irac_1"]    = catalog["FLUX_TOT_irac_1"]
    matched["FLUXERR_TOT_irac_1"] = catalog["FLUXERR_TOT_irac_1"]
    matched["FLUX_TOT_irac_2"]    = catalog["FLUX_TOT_irac_2"]
    matched["FLUXERR_TOT_irac_2"] = catalog["FLUXERR_TOT_irac_2"]
    matched["FLUX_TOT_irac_3"]    = catalog["FLUX_TOT_irac_3"]
    matched["FLUXERR_TOT_irac_3"] = catalog["FLUXERR_TOT_irac_3"]
    matched["FLUX_TOT_irac_4"]    = catalog["FLUX_TOT_irac_4"]
    matched["FLUXERR_TOT_irac_4"] = catalog["FLUXERR_TOT_irac_4"]

    m1,m2,d12 = match_ra_dec(catalog['RA'],catalog['DEC'],archive['ra'],archive['dec'])
    cond = (m2 != len(archive))
    m1, m2 = m1[cond], m2[cond]
    print ('IRAC Archive: Matched %i out of %i sources (%i available)' % (len(m1), len(catalog), len(archive)))

    matched["FLUX_ARX_irac_1"][m1]    = archive["i1_f_ap1"][m2]
    matched["FLUXERR_ARX_irac_1"][m1] = archive["i1_df_ap1"][m2]
    matched["FLUX_ARX_irac_2"][m1]    = archive["i2_f_ap1"][m2]
    matched["FLUXERR_ARX_irac_2"][m1] = archive["i2_df_ap1"][m2]
    matched["FLUX_ARX_irac_3"][m1]    = archive["i3_f_ap1"][m2]
    matched["FLUXERR_ARX_irac_3"][m1] = archive["i3_df_ap1"][m2]
    matched["FLUX_ARX_irac_4"][m1]    = archive["i4_f_ap1"][m2]
    matched["FLUXERR_ARX_irac_4"][m1] = archive["i4_df_ap1"][m2]

    m1,m2,d12 = match_ra_dec(catalog['RA'],catalog['DEC'],spuds['ra'],spuds['dec'])
    cond = (m2 != len(spuds))
    m1, m2 = m1[cond], m2[cond]
    print ('SpUDS catalog: %i out of %i sources (%i available)' % (len(m1), len(catalog), len(spuds)))

    matched["FLUX_SpUDS_irac_1"][m1]    = spuds["flux_ap_36"][m2]
    matched["FLUXERR_SpUDS_irac_1"][m1] = spuds["uncf_ap_36"][m2]
    matched["FLUX_SpUDS_irac_2"][m1]    = spuds["flux_ap_45"][m2]
    matched["FLUXERR_SpUDS_irac_2"][m1] = spuds["uncf_ap_45"][m2]
    matched["FLUX_SpUDS_irac_3"][m1]    = spuds["flux_ap_56"][m2]
    matched["FLUXERR_SpUDS_irac_3"][m1] = spuds["uncf_ap_56"][m2]
    matched["FLUX_SpUDS_irac_4"][m1]    = spuds["flux_ap_80"][m2]
    matched["FLUXERR_SpUDS_irac_4"][m1] = spuds["uncf_ap_80"][m2]

    fitsio.writeto("irac_phot/chk_irac_phot.fits",matched,overwrite=True)
    # fitsio.writeto("final_cats/old_cats_v1.5/chk_irac_phot.fits",matched,overwrite=True)

def plot():

    matched = fitsio.getdata("irac_phot/chk_irac_phot.fits")
    # matched = fitsio.getdata("final_cats/old_cats_v1.5/chk_irac_phot.fits")

    for suffix in ["ARX","SpUDS"]:

        fig,axes = plt.subplots(4,1,figsize=(10,12),dpi=75)
        fig.subplots_adjust(left=0.06,right=0.98,bottom=0.05,top=0.98,wspace=0,hspace=0)

        for i,ax in enumerate(axes):

            filt = str(i+1)

            x = matched["FLUX_TOT_irac_%i"%(i+1)]
            y = matched["FLUX_%s_irac_%i"%(suffix,i+1)]
            cond = (x>0) & (y>0)
            x,y = x[cond],y[cond]

            x = -2.5*np.log10(x) + 23.93
            y = -2.5*np.log10(y) + 23.93

            diff = x - y
            cond_med = (x<np.percentile(x,0.5))
            med, std = np.median(diff), np.std(diff)

            binsx = np.arange(0,35,0.2)
            binsy = np.arange(-25,25,0.1)
            bincx = 0.5*(binsx[1:] + binsx[:-1])
            bincy = 0.5*(binsy[1:] + binsy[:-1])
            gridy, gridx = np.meshgrid(bincy,bincx)
            hist2d = np.histogram2d(x,diff,bins=[binsx,binsy])[0]
            hist2d = np.ma.masked_array(hist2d,mask=hist2d==0)
            hist2d = np.ma.log10(hist2d)
            ax.pcolormesh(gridx,gridy,hist2d,cmap=plt.cm.inferno)

            ax.text(0.02,0.95,"Offset: %.4f"%med,va='top',ha='left',fontsize=16,fontweight=400,transform=ax.transAxes)
            ax.axhline(0,c='k',ls='--',lw=0.5)
            ax.set_ylim(-5*std,+5*std)
            ax.set_xlim(13,27.5)

            bins = np.arange(-10,10,0.05)
            binc = 0.5*(bins[1:]+bins[:-1])
            hist = np.histogram(diff,bins=bins)[0]
            hist = np.ma.masked_array(hist,mask=hist==0)
            hist = np.ma.log10(hist)
            divider = make_axes_locatable(ax)
            dax = divider.append_axes("right", size="20%", pad=0.0, sharey=ax)
            dax.fill_betweenx(binc,0,hist,lw=0,color='k',alpha=0.5)
            dax.axhline(0,c='k',ls='--',lw=0.5)
            dax.set_xlim(0,np.ma.max(hist)+0.2)
            _ = [label.set_visible(False) for label in dax.get_xticklabels()+dax.get_yticklabels()]

            ax.set_ylim(-1.8,2.1)

        axes[-1].set_xlabel("MAG_TOT")

    fig.savefig("irac_phot/chk_irac_phot_{:}.png".format(suffix))
    # fig.savefig("final_cats/old_cats_v1.5/chk_irac_phot_{:}.png".format(suffix))

if __name__ == '__main__':

    # match()
    plot()
    plt.show()

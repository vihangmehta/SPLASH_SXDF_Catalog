import sys,os, warnings
import numpy as np
import astropy.units as u
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse,Circle
from matplotlib.lines import Line2D
from collections import OrderedDict
from astropy.table import Table, Column
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, FK5
from astropy.nddata.utils import Cutout2D
from astropy.utils.exceptions import AstropyWarning

import useful

warnings.simplefilter('ignore',category=AstropyWarning)

leff = [('hsc_g'   ,0.4724),('hsc_r'   ,0.6226),('hsc_i'   ,0.7722),('hsc_z'   ,0.8917),('hsc_y'   ,1.0032),
        ('supcam_b',0.4374),('supcam_v',0.5448),('supcam_r',0.6509),('supcam_i',0.7676),
        ('uds_j'   ,1.2556),('uds_h'   ,1.6496),('uds_k'   ,2.2356),('video_z' ,0.8779),
        ('video_y' ,1.0211),('video_j' ,1.2541),('video_h' ,1.6464),('video_ks',2.1488),
        ('cfht_u'  ,0.3746),('cfhtls_u',0.3811),
        ('irac_1'  ,3.5573),('irac_2'  ,4.5049),('irac_3'  ,5.7386),('irac_4'  ,7.9274)]
leff2 = leff + [('ib_527'  ,0.5261),]

leff.sort(key=lambda x:x[1])
leff = OrderedDict(leff)

leff2.sort(key=lambda x:x[1])
leff2 = OrderedDict(leff2)

def mk_cutout(instr,filt,pos,size):

    if instr!="ib":
        img,hdr = fitsio.getdata("data/orig/mosaic_%s_%s.img.fits"%(instr,filt),header=True)
    else:
        img,hdr = fitsio.getdata("../DATA/IB/OBJECT.527.1.fits",header=True)
        
    wcs = WCS(hdr)
    len_arc_sec = 1./useful.pix_scale

    pos = SkyCoord(*(pos*u.deg),frame='fk5')
    cutout = Cutout2D(img,pos,size,wcs=wcs)

    _stamp = cutout.data[int(np.floor(size/4.)):int(np.ceil(size*3./4.)),int(np.floor(size/4.)):int(np.ceil(size*3./4.))]
    vmin = np.median(_stamp) - 3*np.std(_stamp)
    vmax = np.median(_stamp) + 3*np.std(_stamp)
    return cutout.data, vmin, vmax, len_arc_sec

def mk_stamps(catalog,savename,size=51,leff=leff):

    with PdfPages(savename) as pdf:

        for entry in catalog:

            n_filt = len(leff)
            fig,ax = plt.subplots(1,n_filt,figsize=(n_filt,1/0.7),dpi=75)
            fig.subplots_adjust(left=0.05/n_filt,right=1-0.05/n_filt,bottom=0.05,top=0.7,wspace=0,hspace=0)
            fig.suptitle("ID: %i" % entry["ID"],fontsize=14,fontweight=800)

            for i,fname in enumerate(leff):

                instr,filt = fname.split('_')
                mag = entry['MAG_AUTO_%s'%fname] if instr!='irac' else entry['MAG_TOT_%s'%fname]
                sys.stdout.write("\rPlotting stamp for %i - %s:%s ... \033[K" % (entry['ID'],instr,filt))
                sys.stdout.flush()

                cutout,vmin,vmax,len_arc_sec = mk_cutout(instr=instr,filt=filt,pos=(entry['RA'],entry['DEC']),size=size)
                ax[i].pcolormesh(cutout,cmap=plt.cm.Greys,vmin=vmin,vmax=vmax,rasterized=True)
                ax[i].set_title("%s:%s"%(instr,filt),fontsize=12,fontweight=800)
                ax[i].text(0.95,0.95,"%.2f"%mag,
                            color='b',ha='right',va='top',fontsize=12,fontweight=500,transform=ax[i].transAxes,
                            path_effects=[PathEffects.withStroke(linewidth=0.5,foreground="b")])
                ax[i].add_line(Line2D([0.05*size,0.05*size+len_arc_sec],[0.05*size,0.05*size],lw=3,c='blue'))
                ax[i].add_line(Line2D([size/2.,size/2.],[size/2-0.2*size,size/2.-0.35*size],lw=2,c='red'))
                ax[i].add_line(Line2D([size/2.+0.2*size,size/2.+0.35*size],[size/2.,size/2.],lw=2,c='red'))
                ax[i].xaxis.set_visible(False)
                ax[i].yaxis.set_visible(False)

            pdf.savefig(fig)
            plt.close(fig)

    print "done!"

if __name__ == '__main__':
    
    catalog = fitsio.getdata('final_cats/final_catalog.fits')
    cond = np.array([True]*len(catalog))
    for fname in leff:
        mag = entry['MAG_AUTO_%s'%fname] if instr!='irac' else entry['MAG_TOT_%s'%fname]
        cond = cond & (mag!=-99.)
    catalog = catalog[cond][np.random.randint(len(catalog[cond]),size=10)]
    mk_stamps(catalog,savename='stamps.pdf')

    # sample = Table(np.genfromtxt("/data/highzgal/mehta/misc/mehdi/catalog_215.dat", dtype=[("RA",float),("DEC",float),("MAG_AUTO_ib_527",float)], usecols=[0,1,6]))
    # m1, m2, d12 = useful.match_ra_dec(sample['RA'],sample['DEC'],catalog['RA'],catalog['DEC'])
    # for x in [_ for _ in catalog.dtype.names if "MAG_AUTO_" in _]:
    #     mag = np.zeros(len(sample)) - 99.
    #     mag[m1] = catalog[x][m2]
    #     sample.add_column(Column(name=x,data=mag))
    # sample.add_column(Column(name="ID",data=np.arange(len(sample))+1))
    # mk_stamps(sample,savename='stamps_mehdi.pdf',leff=leff2)
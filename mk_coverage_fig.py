import numpy as np
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from astropy.wcs import WCS

from useful import fcolor_dict
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

def main(mk_rgb=False):

    fig,ax = plt.subplots(1,1,figsize=(10,10),dpi=150,tight_layout=True)

    factor = 20
    x,y = np.indices((2501,2501)) * factor
    hdr = fitsio.getheader("data/orig/mosaic_hsc_y.img.fits")
    wcs = WCS(hdr)
    r,d = wcs.all_pix2world(x,y,1)
    
    if mk_rgb:
        rgb = np.zeros((2500,2500,3))
        rgb[:,:,0] = np.clip(logscale(rebin_(fitsio.getdata("data/orig/mosaic_hsc_y.img.fits"),factor=factor)),0,1)
        rgb[:,:,1] = np.clip(logscale(rebin_(fitsio.getdata("data/orig/mosaic_hsc_r.img.fits"),factor=factor)),0,1)
        rgb[:,:,2] = np.clip(logscale(rebin_(fitsio.getdata("data/orig/mosaic_hsc_g.img.fits"),factor=factor)),0,1)
        color_tuple = rgb.transpose((1,0,2)).reshape((rgb.shape[0]*rgb.shape[1],rgb.shape[2]))
        im = ax.pcolormesh(r,d,rgb[:,:,0],color=color_tuple,linewidth=0)
        im.set_array(None)
    else:
        img = np.clip(logscale(rebin_(fitsio.getdata("data/orig/mosaic_hsc_y.img.fits"),factor=factor)),0,1)
        ax.pcolormesh(r,d,img.T,cmap=plt.cm.Greys,vmin=0,vmax=1,linewidth=0)

    ax.add_patch(mpatches.Ellipse(  xy=(34.576835,-4.8179672),width=2*3228.1658/3600.,height=2*2955.5327/3600.,edgecolor=fcolor_dict['hsc']['g'],facecolor='none',lw=4,label="HSC"))
    # ax.add_patch(mpatches.Rectangle(xy=(34.498573-10382.886/3600./2.,-5.1297314-10300.155/3600./2.),width=10382.886/3600.,height=10300.155/3600.,edgecolor=fcolor_dict['cfhtls']['u'],facecolor='none',lw=4,label='CFHT-LS'))
    ax.add_patch(mpatches.Rectangle(xy=(34.498508-7387.7200/3600./2.,-5.0016043-7387.7200/3600./2.),width=7387.7200/3600.,height=7387.7200/3600.,edgecolor=fcolor_dict['irac']['1'],facecolor='none',lw=4,label='IRAC'))
    ax.add_patch(mpatches.Rectangle(xy=(34.448884-3385.1568/3600./2.,-5.1000118-3356.3060/3600./2.),width=3385.1568/3600.,height=3356.3060/3600.,edgecolor=fcolor_dict['uds']['j'],facecolor='none',lw=4,label='UVISTA/UDS'))
    ax.add_patch(mpatches.Polygon(  xy=([(34.803238,-4.697391),(34.804941,-4.3579328),(34.194745,-4.3542229),(34.192725,-4.6936791),(33.827932,-4.6934248),(33.827302,-5.3074201),(34.198027,-5.3058556),(34.196005,-5.6471666),(34.812998,-5.647162),(34.810968,-5.2984311),(35.159337,-5.3018698),(35.155002,-4.6934411)]),closed=True,edgecolor=fcolor_dict['supcam']['b'],facecolor='none',lw=4,label='SuprimeCam'))
    ax.add_patch(mpatches.Polygon(  xy=([(35.985706,-4.0849371),(34.914891,-4.0755168),(34.914993,-4.2357598),(34.778398,-4.2384915),(34.778363,-4.1556964),(33.838372,-4.1581396),(33.831747,-5.6483204),(35.058136,-5.6406718),(35.058002,-5.4830838),(35.184114,-5.4829538),(35.186878,-5.5604084),(36.115056,-5.5399199),(36.114885,-5.4704973),(37.164244,-5.4647929),(37.158309,-4.0031709),(35.996234,-4.0074955)]),closed=True,edgecolor=fcolor_dict['video']['y'],facecolor='none',lw=4,label='VIDEO'))
    ax.add_patch(mpatches.Polygon(  xy=([(35.054335,-4.3211403),(34.068559,-4.3292324),(34.063159,-4.3933323),(33.843493,-4.3931722),(33.838602,-5.4413465),(33.926591,-5.4375716),(33.926425,-5.626922),(34.006673,-5.628283),(34.005358,-5.6553313),(34.994323,-5.6527556),(34.994297,-5.6179777),(35.125212,-5.6165671),(35.124213,-4.5694852),(35.051861,-4.564202)]),closed=True,edgecolor=fcolor_dict['cfht']['u'],facecolor='none',lw=4,label='MUSUBI'))

    ax.set_xlabel("RA [deg]",fontsize=18)
    ax.set_ylabel("Decl. [deg]",fontsize=18)
    ax.set_xlim(35.55, 33.45)
    ax.set_ylim(-6.05, -3.95)
    ax.set_aspect(1.)

    _ = [label.set_fontsize(14) for label in ax.get_xticklabels()+ax.get_yticklabels()]

    leg = ax.legend(loc="lower center",fontsize=18,ncol=3,scatterpoints=0,markerscale=0,fancybox=True,frameon=False,handlelength=0,handletextpad=0)
    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        txt.set_color(hndl.get_edgecolor())
        txt.set_fontproperties(FontProperties(size=24,weight=600))
        hndl.set_visible(False)

    if mk_rgb: fig.savefig("SXDS_coverage_rgb.png")
    else: fig.savefig("SXDS_coverage.png")

if __name__ == '__main__':
    
    main(mk_rgb=True)
    main(mk_rgb=False)
    # plt.show()

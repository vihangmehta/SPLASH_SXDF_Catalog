import os,sys
import numpy as np
import astropy.io.fits as fitsio
import astropy.units as u
import astropy.coordinates as coord
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sfd

plt.rcParams.update({'font.size':18,'font.weight':400,
                    #'mathtext.default':'regular',
                     'axes.linewidth': 2.0,
                     'xtick.major.width': 1.5,
                     'ytick.major.width': 1.5,
                     'xtick.minor.width': 1.2,
                     'ytick.minor.width': 1.2,
                     'xtick.major.size': 8,
                     'ytick.major.size': 8,
                     'xtick.minor.size': 5,
                     'ytick.minor.size': 5})

class Gal_Extinction():

    def __init__(self):

        self.Rv = 3.1    # Av/E(B-V)
        self.center_rd = [34.5, -5.00]
        self.center_Av = self.calc_Av(ra=self.center_rd[0],dec=self.center_rd[1])

        self.Al_Av = np.genfromtxt('lephare/ext/SPLASH-SXDS.extinc',usecols=[0,2],dtype=[('filt','<U8'),('Al_Av',float)])
        self.fnames = self.Al_Av['filt']
        self.Alambda_Av = {}
        for fname in self.fnames:
            _fname = fname.lower().replace("-","_")
            self.Alambda_Av[_fname] = self.Al_Av[self.Al_Av['filt']==fname][0]

    def get_Av(self):

        ebv = 0.0209 # Schlegel+98
        # ebv = 0.0179 # Schlafly+11
        Av = ebv * self.Rv
        return Av

    def calc_EBV(self,ra,dec):

        if isinstance(ra, float): ra  = np.array([ra,])
        if isinstance(dec,float): dec = np.array([dec,])
        coords = coord.SkyCoord(ra=ra*u.degree,dec=dec*u.degree)
        ebv = sfd.ebv(coords)
        return ebv

    def calc_Av(self,ra=None,dec=None,ebv=None):
        if ebv is None: Av = self.calc_EBV(ra,dec) * self.Rv
        else: Av = ebv * self.Rv
        return Av

    def calc_Alambda(self,instr,filt,Av):

        factor_mag  = self.Alambda_Av["%s_%s"%(instr,filt)]['Al_Av'] * Av
        factor_flux = 10**(factor_mag/-2.5)
        return factor_mag, factor_flux

    def remove_gal_ext(self,mag,flux,instr,filt,ra=None,dec=None,Av=None):

        if Av is None: Av = self.calc_Av(ra=ra,dec=dec)
        factor_mag, factor_flux = self.calc_Alambda(instr,filt,Av=Av)

        cond = (np.abs(mag) != 99.)
        mag[ cond] -= factor_mag[cond]

        cond = (np.abs(flux) != 99.)
        flux[cond] /= factor_flux[cond]

        return mag, flux

    def __repr__(self):

        pprint  = "\n*** Galactic Extinction for SXDS ***"
        pprint += "\n    Center: [%f,%f] \n" % (self.center_rd[0],self.center_rd[1])
        pprint += "\n    E(B-V): %f        " % (self.center_Av / self.Rv)
        pprint += "\n        Av: %f    \n\n" % self.center_Av
        pprint += "%10s -- %8s [%10s]   \n" % ("Filter","Alambda","Flux_fctr")
        pprint += "".join(['-']*40)+"\n"

        for fname in self.fnames:
            instr,filt = fname.lower().split('-')
            factor_mag, factor_flux = self.calc_Alambda(instr,filt,Av=self.center_Av)
            pprint += "%10s -- %8.4f [%10.4f]\n" % (fname,factor_mag,factor_flux)

        return pprint

def mk_plot():

    gal_ext = Gal_Extinction()
    catalog = fitsio.getdata('final_cats/sxds_catalog_v1.5.fits')

    ra = np.linspace(np.min(catalog["RA"]),np.max(catalog["RA"]),1000)
    dec = np.linspace(np.min(catalog["DEC"]),np.max(catalog["DEC"]),1000)
    dec,ra = np.meshgrid(dec,ra)
    Av = gal_ext.calc_Av(ra=ra.flatten(),dec=dec.flatten()).reshape(ra.shape)

    fig = plt.figure(figsize=(10.5,12),dpi=75)
    fig.subplots_adjust(left=0.05,right=0.95,bottom=0.08,top=0.98,hspace=0.2)
    gs  = gridspec.GridSpec(2, 1, height_ratios=[5,2])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.2)

    data = ax1.pcolormesh(ra,dec,Av,cmap=plt.cm.RdYlBu_r)
    cbar = plt.colorbar(mappable=data,cax=cax,orientation='vertical')
    cbar.set_label('$A_V$',fontsize=20)

    Av = gal_ext.calc_Av(ra=catalog["RA"],dec=catalog["DEC"])
    ax2.hist(Av,bins=np.arange(0.02,0.1,0.0002),color='k',lw=0,alpha=0.4)
    ax2.axvline(gal_ext.get_Av(),c='k',lw=1.5,alpha=0.8)

    cond = (0.01<Av) & (Av<0.1)
    print ("Field avg.:", gal_ext.get_Av())
    print ("Scatter:", np.max(Av[cond]) - np.min(Av[cond]))

    ax1.set_xlabel('RA [deg]',fontsize=20)
    ax1.set_ylabel('DEC [deg]',fontsize=20)
    ax1.set_aspect(1.)
    ax1.set_xlim(35.55, 33.45)
    ax1.set_ylim(-6.05, -3.95)
    ax1.set_xticks([35.5,35.0,34.5,34.0,33.5])
    ax1.set_yticks([-4.0,-4.5,-5.0,-5.5,-6.0])

    ax2.set_xlim(0.054,0.086)
    ax2.set_xlabel('$A_V$',fontsize=20)
    ax2.set_yticks([])

    fig.savefig('final_cats/plots/gal_extinction.png')

if __name__ == '__main__':

    # gal_ext = Gal_Extinction()
    # print gal_ext

    mk_plot()
    plt.show()

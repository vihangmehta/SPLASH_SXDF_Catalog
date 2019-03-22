import os,sys
import numpy as np
import scipy.interpolate
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

import useful
from mk_sky_errors import get_sky_phot

def plane(pars,x,y):
    _x = np.sqrt(x)
    _y = 1./np.sqrt(y)
    return pars[0] + pars[1]*_x + pars[2]*_x*_x + pars[3]*_y + pars[4]*_y*_y

def mnmz_func(pars,x,y,z):
    return np.sum((z - plane(pars, x, y))**2)

class CalcUpperLimits():

    def __init__(self,cat_dir='final_cats/'):

        self.get_fnames()
        self.get_errors()
        self.get_par_lims()

    def get_fnames(self):

        fcond = np.array([("irac" not in x) for x in useful.fnames]).astype(bool)
        self.fnames       = np.array(useful.fnames)[fcond]
        self.apersizes    = np.sqrt(np.pi*(useful.apersizes / useful.pix_scale / 2.)**2)
        self.colors       = np.array(['m','r','g','b','c'])
        self.func_corr_list = {}

    def get_errors(self):

        bin_errors = fitsio.getdata('errors/errors_binned.fits')

        self.sky_errors = {}
        self.bin_errors = {}

        for fname in self.fnames:

            instr,filt = fname.split('_')
            self.sky_errors[fname] = get_sky_phot(instr=instr,filt=filt)
            self.bin_errors[fname] = bin_errors[bin_errors['filt']==fname][0]

    def get_par_lims(self):

        sex_errors = fitsio.getdata('errors/errors_sex.fits')
        self.wht_lims = {}
        for fname in self.fnames:
            cond = (sex_errors['AVG_WHT_%s'%fname]!=0)
            self.wht_lims[fname] = [np.min(sex_errors['AVG_WHT_%s'%fname][cond]),\
                                    np.max(sex_errors['AVG_WHT_%s'%fname][cond])]

    def mk_func(self,instr,filt,full=False):

        fname = '_'.join([instr,filt])

        apers = self.apersizes
        apers = np.array([apers]*len(self.bin_errors[fname]['wht_center_sky']))
        whts  = self.bin_errors[fname]['wht_center_sky']
        vals  = self.bin_errors[fname]['mag_lim_sky']

        # Some custom interpolation for a large gap in the weight grid
        if instr=='cfht':
            _whts = 10**(0.5*(np.log10(whts[1,:])+np.log10(whts[2,:])))
            apers = np.insert(apers,2,apers[0,:],axis=0)
            whts  = np.insert(whts,2,_whts,axis=0)
            vals  = np.insert(vals,2,0.5*(vals[1,:]+vals[2,:]),axis=0)

        # pars = scipy.optimize.minimize(mnmz_func,x0=[1,1,1,1,1],args=(apers,whts,vals),bounds=[(-1e3,1e3),(-1e3,1e3),(-1e3,1e3),(-1e3,1e3),(-1e3,1e3)])['x']
        # func = lambda aper,wht,grid: plane(pars,aper,wht)
        # print (pars)

        func = scipy.interpolate.RectBivariateSpline(self.apersizes,np.average(whts,axis=-1),vals.T,kx=1,ky=1,s=0,
                                                        bbox=[self.apersizes[0],
                                                              self.apersizes[-1],
                                                              np.min(self.bin_errors[fname]['wht_edges']),
                                                              np.max(self.bin_errors[fname]['wht_edges'])])

        if full:
            return func, {"x":apers,"y":whts,"z":vals}
        return func

    def calc_residuals(self,instr,filt):

        func, func_dict = self.mk_func(instr=instr,filt=filt,full=True)

        gx = func_dict["x"]
        gy = func_dict["y"]
        gz = np.reshape(self.__call__(gx.ravel(),gy.ravel(),instr=instr,filt=filt), gx.shape)

        resi = np.sum((func_dict["z"]-gz)**2)
        print (resi)

    def __call__(self,aper,wht,instr,filt):

        fname = '_'.join([instr,filt])

        if fname not in self.func_corr_list:
            self.func_corr_list[fname] = self.mk_func(instr=instr,filt=filt)

        cond = (wht==0)
        np.clip(aper,self.apersizes[0],self.apersizes[-1],out=np.asarray(aper))
        np.clip(wht,np.min(self.bin_errors[fname]['wht_edges']),
                    np.max(self.bin_errors[fname]['wht_edges']),out=np.asarray(wht))

        res = self.func_corr_list[fname](aper,wht,grid=False)
        res[cond] = -99.
        return res

    def mk_plot3D(self,instr=None,filt=None):

        fig = plt.figure()
        ax  = fig.add_subplot(111,projection='3d')

        fname = '_'.join([instr,filt])
        instr,filt = fname.split('_')

        self.calc_residuals(instr,filt)

        func, func_dict = self.mk_func(instr=instr,filt=filt,full=True)
        ax.plot_wireframe(func_dict["x"],np.log10(func_dict["y"]),func_dict["z"],color='k',alpha=0.8)

        # gx,gy = np.mgrid[self.apersizes[0]:self.apersizes[-1]:50j,
        #                  np.log10(np.min(func_dict["y"])): \
        #                  np.log10(np.max(func_dict["y"])):50j]
        gx,gy = np.mgrid[self.apersizes[0]:self.apersizes[-1]:50j,
                         np.log10(np.min(self.bin_errors["%s_%s"%(instr,filt)]['wht_edges'])): \
                         np.log10(np.max(self.bin_errors["%s_%s"%(instr,filt)]['wht_edges'])):50j]
        gy = 10**gy
        gz = np.reshape(self.__call__(gx.ravel(),gy.ravel(),instr=instr,filt=filt), gx.shape)
        ax.plot_surface(gx,np.log10(gy),gz,cmap=plt.cm.inferno,alpha=0.5,lw=0,antialiased=False)
        ax.set_ylim(np.log10([0.5*np.min(func_dict["y"]),1.5*np.max(func_dict["y"])]))

        ax.set_title("%s %s"%(instr,filt),fontsize=16,fontweight=800,color='k')
        ax.set_xlabel('Apersize')
        ax.set_ylabel('Log Weight')

if __name__ == '__main__':

    cwd = "/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/"

    ul = CalcUpperLimits()

    for fname in ul.fnames:
        instr,filt = fname.split("_")
        ul.mk_plot3D(instr=instr,filt=filt)
        plt.show()

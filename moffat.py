import os
import numpy as np
import scipy.optimize
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
from mpl_toolkits.mplot3d import Axes3D

import useful

quad_args = {'epsabs':1e-10,'epsrel':1e-10,'limit':250}

def gaussian(r,pars):

    if   len(pars) == 2:
        norm,sig = pars
    elif len(pars) == 1:
        sig  = pars
        norm = 1
    else: raise Exception("Incorrect parameter args in moffat().")
    
    sig = sig / useful.pix_scale
    return np.exp(-0.5*r**2/sig**2) / 2 / np.pi / sig**2

def moffat(r,pars):

    if   len(pars) == 3:
        norm,theta,beta = pars
    elif len(pars) == 2:
        theta,beta = pars
        norm = 1
    else: raise Exception("Incorrect parameter args in moffat().")

    theta = theta / useful.pix_scale
    alpha = theta/(2.*np.sqrt(2.**(1./beta)-1))
    I_0 = (beta-1)*(np.pi*alpha**2)**(-1)
    I_r = I_0*(1+(r/alpha)**2)**(-beta)
    return I_r * norm

def _moffat_int(x,pars):
    
    integrand = lambda r: moffat(r,pars) * 2 * np.pi * r
    return scipy.integrate.quad(integrand,0,x,**quad_args)[0]

moffat_int = np.vectorize(_moffat_int,excluded=['pars'])

def mk_model(pars,N,modelfn):

    x = y = np.arange(N) - N/2
    y,x = np.meshgrid(y,x)
    r = np.sqrt(x**2 + y**2)
    if   modelfn=='moffat':
        z = moffat(r,pars)
    elif modelfn=='gaussian':
        z = gaussian(r,pars)
    return z

def solver(pars,data,modelfn):

    model = mk_model(pars,N=data.shape[0],modelfn=modelfn)
    chi2 = np.ma.sum((data-model)**2/data)
    return chi2 * 1e8

def solver2(pars,data,modelfn):

    model = mk_model(pars,N=data[0].shape[0],modelfn=modelfn)
    chi2 = np.array([np.ma.sum((_data-model)**2/_data) for _data in data])
    return np.sum(chi2) * 1e8

def fit_model(data,modelfn):

    if isinstance(data,list):
        data = [_data / useful.get_psf_fluxes(_data,[1.5/useful.pix_scale,])[0] for _data in data]
    else:
        data = data / useful.get_psf_fluxes(data,[1.5/useful.pix_scale,])[0]

    if   modelfn == 'moffat':
        #x0, bounds = [1.0,1.0,1.0], [[0.0001,10],[0.01,10],[0.01,10]]
        x0, bounds = [0.8,3.5], [[0.01,100],[0.01,100]]
    elif modelfn == 'gaussian':
        #x0, bounds = [1.0,5.0], [[0.0001,10],[1,20]]
        x0, bounds = [5.0], [[1,20],]

    if isinstance(data,list):
        return scipy.optimize.minimize(solver2,x0=x0,bounds=bounds,args=(data,modelfn))['x']
    else:
        return scipy.optimize.minimize(solver,x0=x0,bounds=bounds,args=(data,modelfn))['x']

def mk_masked_psf(psf):

    N = psf.shape[0]
    x = y = np.arange(N) - N/2
    y,x = np.meshgrid(y,x)
    r = np.sqrt(x**2 + y**2)
    mask = (r >= 2.0/useful.pix_scale)
    return np.ma.masked_array(psf,mask=mask)

def main(psf_dir,title,psfex_target,target=None,modelfn='moffat'):

    psf_list = useful.get_PSF_list(psf_dir)
    split_list = np.array_split(psf_list,3)
    psfex_moffat_pars = useful.psfex_moffat_pars(basis_type=psf_dir)

    psfs = [mk_masked_psf(fitsio.getdata(os.path.join(psf_dir,psf_name))[0][0][0]) for psf_name in psf_list]

    pars_psfex_best = psfex_target
    model_psfex_best = mk_model(pars_psfex_best,N=psfs[0].shape[0],modelfn=modelfn)

    if target: pars_best = target
    else:      pars_best = fit_model(psfs,modelfn)
    model_best = mk_model(pars_best,N=psfs[0].shape[0],modelfn=modelfn)
    vmin,vmax = 0,0.2*np.max(model_best)
    print "Basis: %s and %s model" % (psf_dir,modelfn)
    print " Overall %10s -- %s%s" % ('',','.join(["%.2f"%_ for _ in pars_best]), ' (fixed)' if target else '')

    lolim,hilim = -8,8

    fig = plt.figure(figsize=(16,12),dpi=75)
    fig.subplots_adjust(left=0.05,right=0.95,top=0.9,bottom=0.05)
    fig.suptitle(title,fontsize=20,fontweight=600)
    ggs = gridspec.GridSpec(1,3,wspace=0.1,hspace=0.1)
    
    for j,_psf_list in enumerate(split_list):

        gs = gridspec.GridSpecFromSubplotSpec(len(split_list[0]),5,subplot_spec=ggs[j],wspace=0.0,hspace=0.0)

        for i,_psf in enumerate(_psf_list):

            instr,filt = _psf.split('.')[0].split('_')[-2],_psf.split('.')[0].split('_')[-1]
            psf = mk_masked_psf(fitsio.getdata(os.path.join(psf_dir,_psf))[0][0][0])
            HLR = fitsio.getheader(os.path.join(psf_dir,_psf),1)["PSF_FWHM"]

            pars = fit_model(psf,modelfn)
            model = mk_model(pars,N=psf.shape[0],modelfn=modelfn)
            resi = (psf - model) / np.ma.sqrt(psf) * 100
            resi_best = (psf - model_best) / np.ma.sqrt(psf) * 100

            pars_psfex = psfex_moffat_pars[instr][filt]
            model_psfex = mk_model(pars_psfex,N=psf.shape[0],modelfn=modelfn)
            resi_psfex = (psf - model_psfex) / np.ma.sqrt(psf) * 100
            resi_psfex_best = (psf - model_psfex_best) / np.ma.sqrt(psf) * 100
            print "Best Fit: %6s %2s -- %s [PSFEx: %s]" % (instr,filt,','.join(["%.2f"%_ for _ in pars]),','.join(["%.2f"%_ for _ in pars_psfex]))

            ax1 = plt.Subplot(fig,gs[i,0])
            ax2 = plt.Subplot(fig,gs[i,1])
            ax3 = plt.Subplot(fig,gs[i,2])
            ax4 = plt.Subplot(fig,gs[i,3])
            ax5 = plt.Subplot(fig,gs[i,4])

            for ax in [ax1,ax2,ax3,ax4,ax5]:
                fig.add_subplot(ax)
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.set_xlim([psf.shape[0]*(0.5-0.2),psf.shape[0]*(0.5+0.2)])
                ax.set_ylim([psf.shape[0]*(0.5-0.2),psf.shape[0]*(0.5+0.2)])

            ax1.imshow(psf,            origin='lower',interpolation='none',cmap=plt.cm.Greys,vmin=vmin,vmax=vmax)
            im = ax2.imshow(resi,      origin='lower',interpolation='none',cmap=plt.cm.RdYlBu_r,vmin=lolim,vmax=hilim)
            ax3.imshow(resi_best,      origin='lower',interpolation='none',cmap=plt.cm.RdYlBu_r,vmin=lolim,vmax=hilim)
            ax4.imshow(resi_psfex,     origin='lower',interpolation='none',cmap=plt.cm.RdYlBu_r,vmin=lolim,vmax=hilim)
            ax5.imshow(resi_psfex_best,origin='lower',interpolation='none',cmap=plt.cm.RdYlBu_r,vmin=lolim,vmax=hilim)

            ax3.set_title('%s %s -- HLR: %.2f"' % (instr,filt,HLR*useful.pix_scale),fontsize=14,fontweight=600)
            ax2.text(0.5,0.95,','.join(["%.2f"%_ for _ in pars]),
                         ha='center',va='top',fontsize=11,fontweight=400,color='k',transform=ax2.transAxes)
            ax3.text(0.5,0.95,','.join(["%.2f"%_ for _ in pars_best]),
                         ha='center',va='top',fontsize=11,fontweight=400,color='k',transform=ax3.transAxes)
            ax4.text(0.5,0.95,','.join(["%.2f"%_ for _ in pars_psfex]),
                         ha='center',va='top',fontsize=11,fontweight=400,color='k',transform=ax4.transAxes)
            ax5.text(0.5,0.95,','.join(["%.2f"%_ for _ in pars_psfex_best]),
                         ha='center',va='top',fontsize=11,fontweight=400,color='k',transform=ax5.transAxes)

    cbax = fig.add_axes([0.68,0.1,0.25,0.02])
    cbar = fig.colorbar(mappable=im,cax=cbax,orientation='horizontal')
    cbar.set_label("Frac. Residuals [%] $\\left( \\frac{data - model}{\\sqrt{data}} \cdot 100 \\right)$")
    
    if modelfn=='moffat':
        fig.savefig(os.path.join(psf_dir,"plots/moffat.png"))

if __name__ == '__main__':

    main(psf_dir='psfex/orig_pixel/',title=' Pre-matching (PIXEL basis)',psfex_target=[0.7,2.8])
    #main(psf_dir='psfex/orig_gauss/',title=' Pre-matching (GAUSS-LAGUERRE basis)',psfex_target=[0.7,2.8])
    #main(psf_dir='psfex/conv_pixel/',title='Post-matching (PIXEL basis)',psfex_target=[0.7,2.8])
    #main(psf_dir='psfex/conv_gauss/',title='Post-matching (GAUSS-to-GAUSS basis)',psfex_target=[0.7,2.8])

    plt.show()

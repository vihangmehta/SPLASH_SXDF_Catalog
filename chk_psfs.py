import os
import argparse
import numpy as np
import scipy.integrate
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import useful
from moffat import *

apers = np.logspace(np.log10(0.1/0.15),np.log10(6/0.15),50)
radii = apers/2.

def get_colors(psf_dir,psf_list):
    
    fwhm = np.array([fitsio.getheader(os.path.join(psf_dir,fname),1)["PSF_FWHM"] for fname in psf_list])

    for fname in psf_list:
        instr,filt = fname.split('.')[0].split('_')[-2],fname.split('.')[0].split('_')[-1]
        _fwhm = fitsio.getheader(os.path.join(psf_dir,fname),1)["PSF_FWHM"]
        print "%6s %2s: %6.2f" % (instr,filt,_fwhm)

    isort = np.argsort(fwhm)
    colors = plt.cm.jet(np.linspace(0,1,len(fwhm)))
    sorted_colors = colors[isort]

    cdict = {}
    for i,fname in enumerate(psf_list):
        instr,filt = fname.split('.')[0].split('_')[-2],fname.split('.')[0].split('_')[-1]
        cdict["%s_%s"%(instr,filt)] = sorted_colors[i]
        cdict["%s_%s_fwhm"%(instr,filt)] = fwhm[i]

    return cdict

def plot_PSFs(psf_dir,title=''):

    psf_list = useful.get_PSF_list(psf_dir)
    
    N = int(np.sqrt(len(psf_list)))+1
    r,c = len(psf_list)/N+1,N
    fig,axes = plt.subplots(r,c,figsize=(2*c,2*r),dpi=75)
    fig.subplots_adjust(left=0.05,right=0.95,bottom=0.02,top=0.92,wspace=0,hspace=0)
    fig.suptitle(title,fontsize=20,fontweight=600)
    axes = axes.flatten()
    
    for ax in axes:
        ax.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    for fname,ax in zip(psf_list,axes):

        instr,filt = fname.split('.')[0].split('_')[-2],fname.split('.')[0].split('_')[-1]
        _psf = fitsio.open(os.path.join(psf_dir,fname))
        try: psf = _psf[1].data[0][0][0]
        except IndexError: psf = _psf[0].data
        n = psf.shape[0]
        psf = psf[n/3:2*n/3,n/3:2*n/3]

        ax.set_visible(True)
        ax.imshow(psf,cmap=plt.cm.Greys,interpolation='none',origin='lower',vmin=0, vmax=0.5*np.max(psf))
        ax.text(0.05,0.95,"%s:%s"%(instr,filt),ha='left',va='top',fontsize=14,color='k',fontweight=600,
                                    path_effects=[PathEffects.withStroke(linewidth=1.5,foreground="w")],
                                    transform=ax.transAxes)

    fig.savefig("%s/plots/PSFs.png" % psf_dir)

def plot_COG(psf_dir,title='',target=None,model=False):

    # psf_list = useful.get_PSF_list(psf_dir)
    
    instrs = useful.instr_used_list[:-1] if "conv" in psf_dir else useful.instr_used_list[:-2]

    psf_list = []
    for instr in instrs:
        for filt in useful.filters[instr]:
            psf_list.append("%s_%s"%(instr,filt))

    psfex_moffat_pars = useful.psfex_moffat_pars(basis_type=psf_dir)

    fig,[[ax1,ax2],[tax,dax]] = plt.subplots(2,2,figsize=(15,10),dpi=75)
    fig.subplots_adjust(left=0.08,right=0.96,bottom=0.06,top=0.9,wspace=0.15,hspace=0.3)
    fig.suptitle(title,fontsize=20,fontweight=600)
    dax.set_visible(False)

    tax.axhline(1,color='k',ls=':')
    tax.set_xlim(0,13.5)
    tax.set_xlabel('Radius [px]')
    tax.set_ylim(0.6,1.8)
    tax.set_ylabel('F(r)/F$_T$(R)')
    taxx = tax.twiny()
    taxx.set_xlim(useful.pix_scale*tax.get_xlim()[0], useful.pix_scale*tax.get_xlim()[1])
    taxx.set_xlabel('Radius [arcsec]')

    tarflux = moffat_int(radii,pars=target)

    for fname in psf_list:

        instr,filt = fname.split('.')[0].split('_')[-2],fname.split('.')[0].split('_')[-1]
        color = useful.fcolor_dict[instr][filt]
        # _psf = fitsio.open(os.path.join(psf_dir,fname))
        # try: psf = _psf[1].data[0][0][0]
        # except IndexError: psf = _psf[0].data

        # if model:
        #     fit_pars = fit_model(mk_masked_psf(psf),modelfn='moffat')
        #     fluxes = moffat_int(radii,pars=fit_pars)
        # else:
        #     fluxes = useful.get_psf_fluxes(psf,radii)
        #     fluxes = fluxes / useful.get_psf_fluxes(psf,[2.0/useful.pix_scale,])[0]

        # mags = useful.calc_m_from_f(fluxes)
        # ax1.plot(radii,fluxes        ,color=color,lw=0.8,ls='--',alpha=0.8)
        # ax2.plot(radii,mags          ,color=color,lw=0.8,ls='--',alpha=0.8)
        # tax.plot(radii,fluxes/tarflux,color=color,lw=0.8,ls='--',alpha=0.8)

        pars = psfex_moffat_pars[instr][filt]
        fluxes = moffat_int(radii,pars=pars)
        mags = useful.calc_m_from_f(fluxes)
        ax1.plot(radii,fluxes        ,color=color,lw=1.5,label="%s:%s"%(instr,filt),alpha=1)
        ax2.plot(radii,mags          ,color=color,lw=1.5,label="%s:%s"%(instr,filt),alpha=1)
        tax.plot(radii,fluxes/tarflux,color=color,lw=1.5,label="%s:%s"%(instr,filt),alpha=1)

    if target:
        fluxes = moffat_int(radii,pars=target)
        mags = useful.calc_m_from_f(fluxes)
        ax1.plot(radii,fluxes,color='k',lw=4,ls='--',label='Moffat(%.1f,%.1f)' % (target[0],target[1]),alpha=0.8)
        ax2.plot(radii,mags  ,color='k',lw=4,ls='--',label='Moffat(%.1f,%.1f)' % (target[0],target[1]),alpha=0.8)

    ax1.axhline(0,color='k',ls=':')
    ax1.axhline(1,color='k',ls=':')
    ax2.axhline(0,color='k',ls=':')

    ax1.set_ylim(-0.05,1.05)
    ax2.set_ylim(-0.2,2.49)
    ax1.set_ylabel('F(r)/F$_{tot}$')
    ax2.set_ylabel('m(r)-m$_{tot}$')

    ax1.set_xlim(0,13.5)
    ax2.set_xlim(1,20)
    ax2.set_xscale('log')
    
    xtcks = [1,2,3,5,8,10,15,20]
    ax2.set_xticks(xtcks)
    ax2.set_xticklabels(xtcks)

    for ax in [ax1,ax2]:
        axx = ax.twiny()
        axx.set_xlim(useful.pix_scale*ax.get_xlim()[0], useful.pix_scale*ax.get_xlim()[1])
        axx.set_xlabel('Radius [arcsec]')
        ax.set_xlabel('Radius [px]')
        if ax==ax2:
            axx.set_xscale('log')
            xtcks = [0.15,0.2,0.3,0.4,0.5,0.8,1.0,1.5,2.0,3.0]
            axx.set_xticks(xtcks)
            axx.set_xticklabels(xtcks)

    leg = ax2.legend(fontsize=18,ncol=3,loc=10,
                     handlelength=0,handletextpad=0,
                     bbox_to_anchor=[0.5,-0.8])
    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        txt.set_fontweight(600)
        if "Moffat" not in txt.get_text():
            instr,filt = txt.get_text().split(':')
            txt.set_color(useful.fcolor_dict[instr][filt])
        hndl.set_visible(False)

    fig.savefig("%s/plots/curve_of_growth.png" % psf_dir)

def mk_pretty_plot(target=[0.7,2.8]):

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(9,10),dpi=75,sharex=True,sharey=True)
    fig.subplots_adjust(left=0.11,right=0.8,bottom=0.08,top=0.93,hspace=0)

    for ax in [ax1,ax2]:

        ax.set_xlim(0,10.9)
        ax.set_xlabel('Radius [px]',fontsize=24)
        ax.set_ylim(0.49,1.95)
        ax.set_ylabel('F(r)/F$_T$(R)',fontsize=24)
        ax.tick_params(axis="x",direction='in',top="on")
        ax.tick_params(axis="y",direction='in',right="on")
        ax.tick_params(axis="both",which="major",length=8,width=1.2)
        ax.tick_params(axis="both",which="minor",length=5,width=1)
        _ = [label.set_fontsize(16) for label in ax.get_xticklabels()+ax.get_yticklabels()]

    dax = ax1.twiny()
    dax.set_xlim(useful.pix_scale*np.array(ax2.get_xlim()))
    dax.set_xlabel('Radius [arcsec]',fontsize=20)
    dax.tick_params(axis="x",direction='in')
    dax.tick_params(axis="both",which="major",length=8,width=1.2)
    dax.tick_params(axis="both",which="minor",length=5,width=1)
    _ = [label.set_fontsize(16) for label in dax.get_xticklabels()+dax.get_yticklabels()]

    tarflux = moffat_int(radii,pars=target)
    
    for version,ax,text in zip(["orig","conv"],[ax1,ax2],["Before PSF homogenization","After PSF homogenization"]):

        instrs = useful.instr_used_list[:-2] if version=="orig" else useful.instr_used_list[:-1]
        psfex_moffat_pars = useful.psfex_moffat_pars(basis_type=version)

        for instr in instrs:
            for filt in useful.filters[instr]:
                fcolor = useful.fcolor_dict[instr][filt]
                pars = psfex_moffat_pars[instr][filt]
                fluxes = moffat_int(radii,pars=pars)
                mags = useful.calc_m_from_f(fluxes)
                ax.plot(radii,fluxes/tarflux,color=fcolor,lw=1.5,label="%s:%s"%(instr if instr!='cfht' else 'musubi',filt))

        if version=="orig":
            for instr in ["cfhtls",]:
                for filt in useful.filters[instr]:
                    for i in np.arange(9)+1:
                        fcolor = useful.fcolor_dict[instr][filt]
                        pars = psfex_moffat_pars[instr][filt][i]
                        fluxes = moffat_int(radii,pars=pars)
                        mags = useful.calc_m_from_f(fluxes)
                        if i==1: ax.plot(radii,fluxes/tarflux,color=fcolor,lw=0.2,label="%s:%s"%(instr,filt))
                        else:    ax.plot(radii,fluxes/tarflux,color=fcolor,lw=0.2)

        ax.text(0.98,0.95,text,va='top',ha='right',fontsize=18,fontweight=600,transform=ax.transAxes)

    ax1.axvline(1/useful.pix_scale,c='k',ls='--',lw=1.2)
    ax2.axvline(1/useful.pix_scale,c='k',ls='--',lw=1.2)

    ax1.axhline(1,color='k',lw=1.2)
    ax2.axhline(1,color='k',lw=1.2)
    # ax1.axhline(1.1,c='k',ls=':',lw=1.2)
    # ax1.axhline(0.9,c='k',ls=':',lw=1.2)
    # ax2.axhline(1.1,c='k',ls=':',lw=1.2)
    # ax2.axhline(0.9,c='k',ls=':',lw=1.2)

    leg = ax1.legend(fontsize=18,ncol=1,loc="center left",framealpha=0,
                     handlelength=0,handletextpad=0,
                     bbox_to_anchor=[1,0])

    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        txt.set_fontweight(600)
        instr,filt = txt.get_text().split(':')
        txt.set_color(useful.fcolor_dict[instr if instr!='musubi' else 'cfht'][filt])
        hndl.set_visible(False)

    fig.savefig("psfex/curve_of_growth.png")

if __name__ == '__main__':
    
    # psf_list = useful.get_PSF_list(psf_dir='psfex/orig_pixel/',normed=False)
    # cdict = get_colors(psf_dir='psfex/orig/',psf_list=psf_list)

    # plot_COG(psf_dir='psfex/orig/',title=' Pre-homogenization',target=[0.7,2.8])
    # plot_COG(psf_dir='psfex/conv/',title='Post-homogenization',target=[0.7,2.8])

    # plot_PSFs(psf_dir='psfex/orig_pixel/',title=' Pre-matching (PIXEL basis)')
    # plot_PSFs(psf_dir='psfex/orig_gauss/',title=' Pre-matching (GAUSS-LAGUERRE basis)')
    # plot_PSFs(psf_dir='psfex/conv_pixel/',title='Post-matching (PIXEL-to-GAUSS basis)')
    # plot_PSFs(psf_dir='psfex/conv_gauss/',title='Post-matching (GAUSS-to-GAUSS basis)')

    mk_pretty_plot()

    plt.show()
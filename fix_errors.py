import sys,os
import numpy as np
import scipy.interpolate
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties
from sklearn.cluster import KMeans

import useful
from mk_upper_lims import CalcUpperLimits

class FixErrors():

    def __init__(self,catalog=None):

        self.up_lim  = CalcUpperLimits()
        self.catalog = catalog if catalog is not None else fitsio.getdata("final_cats/final_catalog.fits")
        self.fnames  = [x for x in useful.fnames if "irac" not in x]
        self.linapersizes = np.sqrt(np.pi*(useful.apersizes / useful.pix_scale / 2.)**2)

        self.func_corr_list = {}
        self.offsets = {}
        self.n_comp  = {}
        self.labels  = {}
        self.k_means = {}

        self.trim_catalog()
        self.get_sizes()
        self.get_ncomp()
        self.classify()

    def __call__(self,aper,wht,instr,filt,fixed_aper=False):

        aper = np.atleast_1d(aper)
        wht  = np.atleast_1d(wht )

        if not fixed_aper: aper = np.clip(aper,self.size_lims[0],self.size_lims[1])

        fname = self.fname(instr,filt)
        if fname in self.k_means:
            labels = self.k_means[fname].predict(wht.reshape(-1,1))
        else:
            labels = np.zeros(wht.shape)

        if fname not in self.func_corr_list:
            self.func_corr_list[fname] = self.mk_offset_func(instr=instr,filt=filt)
        res = self.func_corr_list[fname](aper)

        return res[labels,range(res.shape[-1])]

    def fname(self,instr,filt):
        return "_".join([instr,filt])

    def trim_catalog(self):

        keep_cols = ["ID","X_IMAGE","Y_IMAGE","ISOAREAF_IMAGE","A_IMAGE","B_IMAGE","KRON_RADIUS"]
        for fname in self.fnames:
            keep_cols = keep_cols + ["MAG_AUTO_%s"%fname,
                                     "MAG_APER_%s"%fname,
                                     "FLUX_APER_%s"%fname,
                                     "FLUXERR_APER_%s"%fname,
                                     "SE_FLAGS_%s"%fname,
                                     "COVERAGE_FLAG_%s"%fname,
                                     "AVG_WHT_%s"%fname]
        self.catalog = useful.view_fields(self.catalog,keep_cols)

    def get_sizes(self,plot=False):

        isoarea  = self.catalog['ISOAREAF_IMAGE']
        autoarea = np.pi*self.catalog['A_IMAGE']*self.catalog['KRON_RADIUS'] * \
                         self.catalog['B_IMAGE']*self.catalog['KRON_RADIUS']
        self.npix_iso  = np.sqrt(isoarea)
        self.npix_auto = np.sqrt(autoarea)
        self.npix_aper = self.linapersizes
        self.npix_all  = np.concatenate([self.npix_auto,self.npix_iso])

        bins = np.arange(0,1000,0.1)
        binc = 0.5*(bins[1:]+bins[:-1])
        hist_all  = np.histogram(self.npix_all ,bins=bins)[0]
        chist_all  = np.cumsum(hist_all)  / float(sum(hist_all))
        self.size_lims = scipy.interpolate.interp1d(chist_all ,binc)([0.05,0.95])

        if plot:

            fig,ax = plt.subplots(1,1,figsize=(10,8),dpi=75,tight_layout=True)

            ax.plot(binc,chist_all ,c='k',lw=2,label='ALL')
            ax.vlines(self.npix_aper,0,1,colors='k',lw=2,label='APER')
            ax.hlines([0.05,0.95],bins[0],bins[-1],colors='k',linestyles='--',lw=1.2)
            ax.vlines(self.size_lims,0,1,colors='k',linestyles='--',lw=2)

            ax.set_xlabel("Linear Aperture size [px]",fontsize=14)
            ax.set_xlabel("Cumulative dist.",fontsize=14)

            ax.set_xlim(-5,50)
            ax.set_ylim(-0.1,1.1)

    def get_ncomp(self):

        for fname in self.fnames:

            instr,filt = fname.split("_")

            if   instr == 'cfht':
                self.n_comp[fname] = 3
            elif instr in ["supcam",] or fname=="video_j":
                self.n_comp[fname] = 2
            else:
                self.n_comp[fname] = 1

    def classify(self):

        for fname in self.fnames:

            sys.stdout.write("\rClassifying %s ... \033[K" % fname)
            sys.stdout.flush()

            instr,filt = fname.split("_")
            self.labels[fname] = np.zeros(len(self.catalog),dtype=int) - 99.
            ratio,whts = self.calc_ratio(aper_num=2,instr=instr,filt=filt,return_whts=True)
            cond = (whts>0.)

            if self.n_comp[fname]:
                data = whts[cond].reshape(-1,1)
                self.k_means[fname] = KMeans(n_clusters=self.n_comp[fname])
                self.k_means[fname].fit(data)
                self.labels[fname][cond] = self.k_means[fname].predict(data)
            else:
                self.k_means = None
                self.labels[fname][cond] = 0

        print ("done.")

    def calc_ratio(self,aper_num,instr,filt,return_whts=False,return_all=False):

        cond = (self.catalog[     'MAG_AUTO_%s_%s'%(instr,filt)] != -99.) & \
               (self.catalog[     'SE_FLAGS_%s_%s'%(instr,filt)] ==   0 ) & \
               (self.catalog['COVERAGE_FLAG_%s_%s'%(instr,filt)] ==   1 ) & \
               (self.catalog[     'MAG_APER_%s_%s'%(instr,filt)][:,aper_num] >   15.) & \
               (self.catalog[    'FLUX_APER_%s_%s'%(instr,filt)][:,aper_num] != -99.) & \
               (self.catalog[ 'FLUXERR_APER_%s_%s'%(instr,filt)][:,aper_num] >    0.) & \
               (self.catalog[      'AVG_WHT_%s_%s'%(instr,filt)] !=   0 )

        mag_lim = [np.percentile(self.catalog['MAG_APER_%s_%s'%(instr,filt)][:,aper_num][cond],25),
                   np.percentile(self.catalog['MAG_APER_%s_%s'%(instr,filt)][:,aper_num][cond],75)]

        cond = cond & \
               (self.catalog[     'MAG_APER_%s_%s'%(instr,filt)][:,aper_num] > mag_lim[0]) & \
               (self.catalog[     'MAG_APER_%s_%s'%(instr,filt)][:,aper_num] < mag_lim[1])

        whts = self.catalog["AVG_WHT_%s_%s"%(instr,filt)]
        mlim_aper = self.up_lim(aper=self.up_lim.apersizes[aper_num],wht=whts[cond],instr=instr,filt=filt)
        flux_err_sky = 10**((mlim_aper - useful.zp) / -2.5)
        flux_err_sex = self.catalog["FLUXERR_APER_%s_%s"%(instr,filt)][:,aper_num][cond]

        ratio = np.zeros(len(self.catalog)) - 99.
        ratio[cond] = flux_err_sky / flux_err_sex

        if return_all: return ratio,whts,flux_err_sky,flux_err_sex
        if return_whts: return ratio,whts
        return ratio

    def calc_offset(self,aper_num,instr,filt):

        fname = self.fname(instr,filt)
        if fname not in self.fnames:
            raise Exception("Invalid instr/filt combination: %s/%s"%(instr,filt))

        ratio,whts = self.calc_ratio(aper_num=aper_num,instr=instr,filt=filt,return_whts=True)
        cond = (ratio!=-99.)
        offsets = np.zeros(self.n_comp[fname])
        for i in range(self.n_comp[fname]):
            offsets[i] = np.median(ratio[cond & (self.labels[fname]==i)])
        return offsets

    def calc_offsets(self,instr,filt):

        fname = self.fname(instr,filt)
        if fname in self.offsets: return self.offsets[fname]
        self.offsets[fname] = np.zeros((self.n_comp[fname],len(self.linapersizes)))
        for aper_num in range(len(self.linapersizes)):
            self.offsets[fname][:,aper_num] = self.calc_offset(aper_num=aper_num,instr=instr,filt=filt)
        return self.offsets[fname]

    def mk_offset_func(self,instr,filt):

        fname = self.fname(instr,filt)
        if fname not in self.offsets:
            self.offsets[fname] = self.calc_offsets(instr=instr,filt=filt)
        return scipy.interpolate.interp1d(self.linapersizes,self.offsets[fname],kind="linear",fill_value='extrapolate')

    def print_offsets(self):

        with open("errors/offsets.txt","w") as f:

            f.write("%6s"%"Aper#")
            for fname in self.fnames:
                instr,filt = fname.split("_")
                self.calc_offsets(instr=instr,filt=filt)
                spacing = "%"+str(self.n_comp[fname]*10 + (self.n_comp[fname]-1) + 2)+"s"
                f.write(spacing%fname)
            f.write("\n")

            for aper_num in range(len(self.linapersizes)):
                f.write("%6i"%(aper_num+1))
                for fname in self.fnames:
                    f.write('      [%s]' % ','.join(map(lambda x: "%7.4f" % x, self.offsets[fname][:,aper_num])))
                f.write("\n")

    def mk_plot0(self,aper_num):
        """
        Compare SExtractor and Sky errors as a func of weight
        """
        _fnames = np.array_split(self.fnames,4)
        fig = plt.figure(figsize=(20,12),dpi=75)
        fig.subplots_adjust(left=0.04,right=0.98,top=0.98,bottom=0.05,wspace=0.23,hspace=0.23)
        ogs = gridspec.GridSpec(4,len(_fnames[0]))

        for i,fname in enumerate(self.fnames):

            sys.stdout.write("\rPlot #0: %s \033[K" % fname)
            sys.stdout.flush()

            instr,filt = fname.split('_')
            fcolor = useful.fcolor_dict[instr][filt]

            igs = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=ogs[i],wspace=0,hspace=0,height_ratios=[1,3])
            ax1 = fig.add_subplot(igs[1])
            ax2 = fig.add_subplot(igs[0],sharex=ax1)

            _ = [label.set_fontsize(10)   for label in ax1.get_xticklabels()+ax1.get_yticklabels()+ax2.get_xticklabels()+ax2.get_yticklabels()]
            _ = [label.set_visible(False) for label in ax2.get_xticklabels()+ax2.get_yticklabels()]

            ax1.set_xlabel('Weight',fontsize=12)
            ax1.set_ylabel('$\sigma$',fontsize=12)

            ax1.text(0.95,0.95,"%s:%s"%(instr,filt),va='top',ha='right',fontsize=16,fontweight=800,color=fcolor,transform=ax1.transAxes)

            ratio,whts,flux_err_sky,flux_err_sex = self.calc_ratio(aper_num=aper_num,instr=instr,filt=filt,return_all=True)
            cond = (ratio!=-99.)
            ax1.scatter(whts[cond],flux_err_sky,c='b',s=3,alpha=0.02)
            ax1.scatter(whts[cond],flux_err_sex,c='r',s=3,alpha=0.02)
            ax2.hist(whts[cond],bins=1000,color=fcolor)

            ax1.set_yscale("log")

        print()
        fig.savefig("errors/plots/err_analysis_new_r%i_0.png"%(aper_num+1))

    def mk_plot1(self,aper_num):
        """
        Plot correction factor as a func of weight
        """
        _fnames = np.array_split(self.fnames,4)
        fig = plt.figure(figsize=(20,12),dpi=75)
        fig.subplots_adjust(left=0.04,right=0.98,top=0.98,bottom=0.05,wspace=0.23,hspace=0.23)
        ogs = gridspec.GridSpec(4,len(_fnames[0]))

        for i,fname in enumerate(self.fnames):

            sys.stdout.write("\rPlot #1: %s \033[K" % fname)
            sys.stdout.flush()

            instr,filt = fname.split('_')
            fcolor = useful.fcolor_dict[instr][filt]
            colors = ['b','r','g']

            igs = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=ogs[i],wspace=0,hspace=0,width_ratios=[5,2])
            ax2 = fig.add_subplot(igs[1])
            ax1 = fig.add_subplot(igs[0],sharey=ax2)

            _ = [label.set_visible(False) for label in ax2.get_xticklabels()+ax2.get_yticklabels()]
            _ = [label.set_fontsize(10)   for label in ax1.get_xticklabels()+ax1.get_yticklabels()]

            ax1.set_xlabel('Weight',fontsize=12)
            ax1.set_ylabel('$\sigma_{sky} / \sigma_{SE}$',fontsize=12)

            ax1.text(0.05,0.95,"%s:%s"%(instr,filt),va='top',ha='left',fontsize=16,fontweight=800,color=fcolor,transform=ax1.transAxes)

            ratio,whts = self.calc_ratio(aper_num=aper_num,instr=instr,filt=filt,return_whts=True)
            cond = (ratio!=-99.)
            ax1.scatter(whts[cond],ratio[cond],c=fcolor,s=3,alpha=0.02)
            ylim = ax1.get_ylim()

            bins = np.arange(0,10,0.01)
            binc = 0.5*(bins[1:]+bins[:-1])
            dbin = bins[1:] - bins[:-1]

            for i,c in zip(range(self.n_comp[fname]),colors):
                _cond = cond & (self.labels[fname]==i)
                hist = np.histogram(ratio[_cond],bins=bins)[0]
                ax2.barh(binc,hist,height=dbin,color=c,alpha=0.3)

            offsets = self.calc_offsets(instr=instr,filt=filt)[:,aper_num]
            ax1.hlines(offsets,*ax1.get_xlim(),colors=colors[:len(offsets)],linestyles='--',lw=0.5)
            ax2.hlines(offsets,*ax2.get_xlim(),colors=colors[:len(offsets)],linestyles='--',lw=0.5)

            ax1.axhline(1,c='k',ls='-',lw=1.)
            ax2.axhline(1,c='k',ls='-',lw=1.)

            # ax1.set_xscale("log")
            ax2.set_ylim(ylim)

        print()
        fig.savefig("errors/plots/err_analysis_new_r%i_1.png"%(aper_num+1))

    def mk_plot2(self,aper_num):
        """
        Plot the classification of weights and ratios
        """
        _fnames = np.array_split(self.fnames,4)
        fig = plt.figure(figsize=(20,12),dpi=75)
        fig.subplots_adjust(left=0.04,right=0.98,top=0.95,bottom=0.05,wspace=0.23,hspace=0.55)
        ogs = gridspec.GridSpec(4,len(_fnames[0]))

        for i,fname in enumerate(self.fnames):

            sys.stdout.write("\rPlot #2: %s \033[K" % fname)
            sys.stdout.flush()

            instr,filt = fname.split('_')
            fcolor = useful.fcolor_dict[instr][filt]
            colors = ['b','r','g']

            igs = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=ogs[i],wspace=0,hspace=0)
            ax1 = fig.add_subplot(igs[0])
            ax2 = fig.add_subplot(igs[1])

            ax1.text(0.95,0.95,"%s:%s"%(instr,filt),va='top',ha='right',fontsize=16,fontweight=800,color=fcolor,transform=ax1.transAxes)
            ax1.set_ylim(1,2)
            ax2.set_ylim(1,2)

            ratio,whts = self.calc_ratio(aper_num=aper_num,instr=instr,filt=filt,return_whts=True)
            cond = (ratio!=-99.)
            for i,c in zip(range(self.n_comp[fname]),colors):
                _cond = cond & (self.labels[fname]==i)
                hist = ax1.hist( whts[_cond],bins=500,color=c,alpha=1,lw=1.5,histtype='step')[0]
                ax1.set_ylim(1,max(ax1.get_ylim()[1],1.1*max(hist)))
                hist = ax2.hist(ratio[_cond],bins=500,color=c,alpha=1,lw=1.5,histtype='step')[0]
                ax2.set_ylim(1,max(ax2.get_ylim()[1],1.1*max(hist)))

            offsets = self.calc_offsets(instr=instr,filt=filt)[:,aper_num]
            ax2.vlines(offsets,*ax2.get_ylim(),colors=colors[:len(offsets)],linestyles='--',lw=0.5)
            ax2.axvline(1,c='k',ls='-',lw=1.)

            ax1.set_yscale("log")
            ax2.set_yscale("log")

            ax1.xaxis.tick_top()
            ax1.set_xlabel('Weight',fontsize=12)
            ax1.xaxis.set_label_position('top')
            ax2.set_xlabel('Ratio',fontsize=12)
            _ = [label.set_fontsize(10) for label in ax1.get_xticklabels()+ax1.get_yticklabels()+ax2.get_xticklabels()+ax2.get_yticklabels()]

        print()
        fig.savefig("errors/plots/err_analysis_new_r%i_2.png"%(aper_num+1))

    def mk_plot3(self):

        _fnames = np.array_split(self.fnames,4)
        fig = plt.figure(figsize=(20,12),dpi=75)
        fig.subplots_adjust(left=0.04,right=0.98,top=0.95,bottom=0.05,wspace=0.23,hspace=0.23)
        ogs = gridspec.GridSpec(4,len(_fnames[0]))

        for i,fname in enumerate(self.fnames):

            sys.stdout.write("\rPlot #3: %s \033[K" % fname)
            sys.stdout.flush()

            instr,filt = fname.split('_')
            fcolor = useful.fcolor_dict[instr][filt]
            colors = ['b','r','g']

            igs = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=ogs[i],wspace=0,hspace=0)
            ax = fig.add_subplot(igs[0])

            ax.text(0.05,0.95,"%s:%s"%(instr,filt),va='top',ha='left',fontsize=16,fontweight=800,color=fcolor,transform=ax.transAxes)

            offsets = self.calc_offsets(instr=instr,filt=filt)
            xx = np.arange(self.size_lims[0],self.size_lims[-1],0.1)
            yy = self.mk_offset_func(instr,filt)(xx)

            for i,c in zip(range(self.n_comp[fname]),colors):
                ax.scatter(self.linapersizes,offsets[i,:],s=20,color=c)
                ax.plot(xx,yy[i,:],c=c,lw=1.2)

            ax.axhline(1,c='k',ls='-',lw=1.)

            ax.set_xlabel('Linear Aperture size [px]',fontsize=12)
            ax.set_ylabel('Offset',fontsize=12)
            _ = [label.set_fontsize(10) for label in ax.get_xticklabels()+ax.get_yticklabels()]

        print()
        fig.savefig("errors/plots/err_analysis_new_3.png")

if __name__ == '__main__':

    # main()

    fix = FixErrors()
    # fix.get_sizes(plot=True)
    fix.print_offsets()

    # for i in range(5):
    #     fix.mk_plot0(aper_num=i)
    #     fix.mk_plot1(aper_num=i)
    #     fix.mk_plot2(aper_num=i)

    # fix.mk_plot3()

    # plt.show()

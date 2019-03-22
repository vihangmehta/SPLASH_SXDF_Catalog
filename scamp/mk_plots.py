import numpy as np
import matplotlib.pyplot as plt

import useful

def main():

    scamp_pars = np.genfromtxt("scamp_summary.txt",
                    dtype=[("instr",np.object),("dr_b",float),("dd_b",float),("sr_b",float),("sd_b",float),
                                               ("dr_a",float),("dd_a",float),("sr_a",float),("sd_a",float)])

    cond = ['SupCam-' in _ for _ in scamp_pars["instr"]]
    supcam_entries = scamp_pars[cond]

    ientry_sc = np.where(scamp_pars["instr"]=='SupCam')[0][0]
    for x in ["dr_b","dd_b","sr_b","sd_b","dr_a","dd_a","sr_a","sd_a"]:
        scamp_pars[ientry_sc][x] = np.mean(supcam_entries[x])

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(9,8),dpi=75,tight_layout=False,sharey=True)
    fig.subplots_adjust(left=0.1,right=0.96,bottom=0.15,top=0.96,wspace=0,hspace=0)

    xlabels = scamp_pars["instr"][:6]

    for i,instr in enumerate(xlabels):

        c = useful.fcolor_dict[instr.lower()][useful.filters[instr.lower()][0]]

        ax1.plot([i,i],[scamp_pars["sr_b"][i],scamp_pars["sr_a"][i]],lw=1.5,ls='--',c=c,zorder=1)
        ax1.scatter(i,scamp_pars["sr_b"][i],marker='s',s=100,facecolor='w',edgecolor=c,zorder=2)
        ax1.scatter(i,scamp_pars["sr_a"][i],marker='s',s=100,facecolor=c,edgecolor=c,zorder=2)

        ax2.plot([i,i],[scamp_pars["sd_b"][i],scamp_pars["sd_a"][i]],lw=1.5,ls='--',c=c,zorder=1)
        ax2.scatter(i,scamp_pars["sd_b"][i],marker='s',s=100,facecolor='w',edgecolor=c,zorder=2)
        ax2.scatter(i,scamp_pars["sd_a"][i],marker='s',s=100,facecolor=c,edgecolor=c,zorder=2)

    ax2.set_xlim(-0.5,5.5)
    ax2.set_ylim(0.095,0.205)
    ax2.set_xticks(range(len(xlabels)))
    ax2.set_xticklabels(xlabels,rotation=45,fontsize=18,fontweight=600)

    _ = [tick.set_color(useful.fcolor_dict[instr.lower()][useful.filters[instr.lower()][0]]) for instr,tick in zip(xlabels,ax2.get_xticklabels())]

    ax1.set_ylabel("$\sigma_{RA}$ [arcsec]",fontsize=18)
    ax2.set_ylabel("$\sigma_{DEC}$ [arcsec]",fontsize=18)

def main2():

    scamp_pars = np.genfromtxt("scamp_summary.txt",
                    dtype=[("instr",np.object),("dr_b",float),("dd_b",float),("sr_b",float),("sd_b",float),
                                               ("dr_a",float),("dd_a",float),("sr_a",float),("sd_a",float)])

    cond = ['SupCam-' in _ for _ in scamp_pars["instr"]]
    supcam_entries = scamp_pars[cond]

    ientry_sc = np.where(scamp_pars["instr"]=='SupCam')[0][0]
    for x in ["dr_b","dd_b","sr_b","sd_b","dr_a","dd_a","sr_a","sd_a"]:
        scamp_pars[ientry_sc][x] = np.mean(supcam_entries[x])

    fig,ax = plt.subplots(1,1,figsize=(10,6),dpi=75,tight_layout=True)

    xlabels = scamp_pars["instr"][:6]

    for i,instr in enumerate(xlabels):

        c = useful.fcolor_dict[instr.lower()][useful.filters[instr.lower()][0]]
        offset = 0.15

        ax.plot([i-offset,i-offset],[scamp_pars["sr_b"][i],scamp_pars["sr_a"][i]],lw=1.5,ls='--',c=c,zorder=1)
        ax.scatter(i-offset,scamp_pars["sr_b"][i],marker='s',s=100,facecolor='w',edgecolor=c,zorder=2)
        ax.scatter(i-offset,scamp_pars["sr_a"][i],marker='s',s=100,facecolor=c,edgecolor=c,zorder=2)

        ax.plot([i+offset,i+offset],[scamp_pars["sd_b"][i],scamp_pars["sd_a"][i]],lw=1.5,ls=':',c=c,zorder=1)
        ax.scatter(i+offset,scamp_pars["sd_b"][i],marker='o',s=100,facecolor='w',edgecolor=c,zorder=2)
        ax.scatter(i+offset,scamp_pars["sd_a"][i],marker='o',s=100,facecolor=c,edgecolor=c,zorder=2)

    ax.set_xlim(-0.5,5.5)
    ax.set_ylim(0.095,0.205)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels([x if x!='CFHT' else 'MUSUBI' for x in xlabels],rotation=45,fontsize=18,fontweight=600)

    _ = [tick.set_color(useful.fcolor_dict[instr.lower()][useful.filters[instr.lower()][0]]) for instr,tick in zip(xlabels,ax.get_xticklabels())]

    ax.set_ylabel("$\sigma_{RA}$ or $\sigma_{DEC}$ [arcsec]",fontsize=18)

    fig.savefig("astrometry_residuals.png")

if __name__ == '__main__':

    main2()
    plt.show()
import os
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects
import matplotlib.ticker as ticker

import useful

plt.rcParams.update({'font.size':18,'font.weight':400,
                    #'mathtext.default':'regular',
                     'axes.linewidth': 1.25,
                     'xtick.major.width': 1.25,
                     'ytick.major.width': 1.25,
                     'xtick.minor.width': 1.,
                     'ytick.minor.width': 1.,
                     'xtick.major.size': 10,
                     'ytick.major.size': 10,
                     'xtick.minor.size': 7,
                     'ytick.minor.size': 7,
                     'xtick.direction' : "inout"})

# min_wave = {'video_z.pb2': 7500,'video_y.pb2': 8500,'video_j.pb2':10000,'video_h.pb2':13000,'video_ks.pb2':17000}
# max_wave = {'video_z.pb2':10000,'video_y.pb2':12250,'video_j.pb2':15000,'video_h.pb2':20000,'video_ks.pb2':25000}

min_wave = {'video_z.pb2': 7500,'video_y.pb2': 8500,'video_j.pb2':10000,'video_h.pb2':13000,'video_ks.pb2':18500}
max_wave = {'video_z.pb2':10000,'video_y.pb2':12250,'video_j.pb2':15000,'video_h.pb2':20000,'video_ks.pb2':24500}

def mk_transmission_curves():

    video_dir = 'lephare/filters/video/'
    video_curves = ['video_z.pb2','video_y.pb2','video_j.pb2','video_h.pb2','video_ks.pb2']

    colors = plt.cm.gist_rainbow_r(np.linspace(0,0.9,len(video_curves)))
    fig,ax = plt.subplots(1,1,figsize=(10,6),dpi=75,tight_layout=True)

    for curve,c in zip(video_curves,colors):

        wave,sens = np.genfromtxt(os.path.join(video_dir,curve),unpack=True)

        # _wave = np.linspace(min_wave[curve],max_wave[curve],500)
        _wave = np.arange(min_wave[curve],max_wave[curve],10)
        print len(_wave)
        _sens = scipy.interpolate.interp1d(wave,sens)(_wave) / 100.
        np.savetxt(os.path.join(video_dir,curve.replace('.pb2','.pb')),
                    np.vstack((_wave,_sens)).T,
                    fmt='%15.4f%20.6e',header='VIDEO-%s'%(curve.split('.')[0].split('_')[-1]))
        useful.force_symlink(src=os.path.join(cwd,video_dir,curve.replace('.pb2','.pb')),dst=os.path.join(lephare_dir,curve.replace('.pb2','.pb')))

        ax.plot( wave, sens/100.,c=c,lw=1.5,label=curve)
        ax.plot(_wave,_sens,c='k',lw=1.0,ls='--')

    ax.set_xlabel('Wavelength [$\\AA$]')
    ax.set_ylabel('Throughput')

def plot_transmission_curves():

    pivot_l = {'hsc_g'   :0.4816, 'hsc_r'   :0.6234, 'hsc_i'   :0.7741, 'hsc_z'   :0.8912, 'hsc_y'   :0.9780,
               'supcam_b':0.4374, 'supcam_v':0.5448, 'supcam_r':0.6509, 'supcam_i':0.7676, 'supcam_z':0.9195,
               'uds_j'   :1.2556, 'uds_h'   :1.6496, 'uds_k'   :2.2356,
               'video_z' :0.8779, 'video_y' :1.0211, 'video_j' :1.2541, 'video_h' :1.6464, 'video_ks':2.1488,
               'cfht_u'  :0.3811,
               'cfhtls_u':0.3811, 'cfhtls_g':0.4862, 'cfhtls_r':0.6258, 'cfhtls_i':0.7553, 'cfhtls_z':0.8871,
               'irac_1'  :3.5573, 'irac_2'  :4.5049, 'irac_3'  :5.7386, 'irac_4'  :7.9274}

    label = {'hsc_g'   :'HSC-g'   , 'hsc_r'   :'HSC-r'   , 'hsc_i'   :'HSC-i'       , 'hsc_z'   :'HSC-z'     , 'hsc_y'   :'HSC-y'     ,
             'supcam_b':'SupCam-B', 'supcam_v':'SupCam-V', 'supcam_r':'SupCam-R$_c$', 'supcam_i':'SupCam-i\'', 'supcam_z':'SupCam-z\'',
             'uds_j'   :'UDS-J'   , 'uds_h'   :'UDS-H'   , 'uds_k'   :'UDS-K'       ,
             'video_z' :'VIDEO-Z' , 'video_y' :'VIDEO-Y' , 'video_j' :'VIDEO-J'     , 'video_h' :'VIDEO-H'   , 'video_ks':'VIDEO-Ks'  ,
             'cfht_u'  :'MUSUBI-u'  ,
             'cfhtls_u':'CFHTLS-u', 'cfhtls_g':'CFHTLS-g', 'cfhtls_r':'CFHTLS-r'    , 'cfhtls_i':'CFHTLS-i'  , 'cfhtls_z':'CFHTLS-z'  ,
             'irac_1'  :'IRAC-ch1', 'irac_2'  :'IRAC-ch2', 'irac_3'  :'IRAC-ch3'    , 'irac_4'  :'IRAC-ch4'}

    sorted_pivot_l = sorted(pivot_l, key=lambda x: pivot_l[x])
    colors = plt.cm.gist_rainbow_r(np.linspace(0.1,1,len(pivot_l)))

    # fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1,figsize=(15,8),dpi=75)
    fig = plt.figure(figsize=(15,12),dpi=75)
    fig.subplots_adjust(left=0.07,right=0.98,bottom=0.1,top=0.96,wspace=0,hspace=0.16)
    ogs  = gridspec.GridSpec(2,1,height_ratios=[4,1])
    igs0 = gridspec.GridSpecFromSubplotSpec(4,1,subplot_spec=ogs[0],wspace=0,hspace=0)
    igs1 = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=ogs[1])

    ax1 = fig.add_subplot(igs0[0,0])
    ax2 = fig.add_subplot(igs0[1,0])
    ax3 = fig.add_subplot(igs0[2,0])
    ax4 = fig.add_subplot(igs0[3,0])
    ax5 = fig.add_subplot(igs1[0,0])

    i1 = i2 = i3 = i4 = i5 = 0
    for i,(fname,fcolor) in enumerate(zip(sorted_pivot_l,colors)):

        instr,filt = fname.split('_')
        filt_file = "/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/lephare/filters/%s/%s_%s.pb"%(instr,instr,filt)
        fcolor = useful.fcolor_dict[instr][filt]

        filt_waves,filt_sens = np.genfromtxt(filt_file,unpack=True)
        filt_waves = filt_waves * 1e-4
        filt_sens = filt_sens / np.max(filt_sens)

        if instr == 'hsc':
            ax,i=ax1,i1
            i1+=1
            ax.set_xlim(np.array([0.295,2.6]))
        if instr in ['cfht','supcam','uds']:
            ax,i=ax2,i2
            i2+=1
            ax.set_xlim(np.array([0.295,2.6]))
        if instr == 'video':
            ax,i=ax3,i3
            i3+=1
            ax.set_xlim(np.array([0.295,2.6]))
        if instr == 'cfhtls':
            ax,i=ax4,i4
            i4+=1
            ax.set_xlim(np.array([0.295,2.6]))
        if instr == 'irac':
            ax,i=ax5,i5
            i5+=1
            ax.set_xlim(np.array([2.9,10.2]))

        ax.plot(filt_waves, filt_sens, c=fcolor, lw=2, alpha=1)
        ax.fill_between(filt_waves, 0, filt_sens, color=fcolor, lw=1, zorder=1, alpha=0.2)
        ypos = 0.3+0.3*(i%2) if instr in ['supcam','video','cfhtls'] or (instr=='hsc' and filt in ['z','y']) else 0.3
        text = ax.text(pivot_l[fname], ypos, label[fname], color='k', va='center', ha='center', fontsize=24, fontweight=400, alpha=0.5)
        text.set_path_effects([path_effects.Stroke(linewidth=2.5, foreground=fcolor),path_effects.Normal()])
        ax.set_xscale('log')
        ax.set_ylim(0,1.19)
        ax.set_yticks([0,1])

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: "%.1f"%y if y%1!=0 else "%i"%y))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())

        _ = [tick.set_fontsize(20) for tick in ax.xaxis.get_majorticklabels()+ax.xaxis.get_minorticklabels()+\
                                               ax.yaxis.get_majorticklabels()+ax.yaxis.get_minorticklabels()]
    
    ax4.set_xticks([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.5])
    ax5.set_xticks(np.arange(3,11))

    _ = [tick.set_visible(False) for tick in ax1.xaxis.get_majorticklabels()+ax1.xaxis.get_minorticklabels()+\
                                             ax2.xaxis.get_majorticklabels()+ax2.xaxis.get_minorticklabels()+\
                                             ax3.xaxis.get_majorticklabels()+ax3.xaxis.get_minorticklabels()]

    ax5.set_xlabel("Wavelength [$\\mu$m]",fontsize=24)
    fig.text(0.025,0.5,"Transmission",fontsize=24,rotation=90,va="center",ha="center")

    fig.savefig("lephare/filters/transmission_curves.png")

def plot_transmission_curves2():

    fig,ax = plt.subplots(1,1,figsize=(10,6),dpi=75,tight_layout=True)

    trans_dir = os.path.join(cwd,"lephare/filters/")
    # for instr in useful.instr_used_list:
    #     for filt,color in zip(useful.filters[instr],useful.fcolor_dict[instr]):
    instr,filt,color = 'hsc','g','k'
    wave,trans = np.genfromtxt(os.path.join(trans_dir,instr,"%s_%s.pb"%(instr,filt)),unpack=True)
    ax.plot(wave,trans,color=color,lw=1.5,label="%s:%s"%(instr,filt))
    ax.fill_between(wave,0,trans,color=color,alpha=0.2)

    ax.set_xlabel("Wavelength [$\\AA$]")
    ax.set_ylabel("Throughput")

if __name__ == '__main__':
    
    cwd = '/data/highzgal/PUBLICACCESS/SPLASH/PROCESS'
    lephare_dir = '/data/highzgal/mehta/lephare/lephare_dev/filt/SPLASH-SXDS/'
    # mk_transmission_curves()
    plot_transmission_curves2()
    plt.show()
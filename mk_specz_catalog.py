import numpy as np
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def mk_zspec_plot():

    catalog = fitsio.getdata("final_cats/final_catalog_errfix.extra.fits")
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])

    fig = plt.figure(figsize=(10,11.5),dpi=75)
    fig.subplots_adjust(left=0.1,right=0.98,top=0.98,bottom=0.06,hspace=0,wspace=0)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1],sharex=ax1)
    
    for i,zlabel1 in enumerate(['M14_GRISMz','VIPERS_SPECz','UDSz_SPECz','SUBARU_SPECz','XUDS_SPECz','C3R2_SPECz']):
    
        for j,zlabel2 in enumerate(['M14_GRISMz','VIPERS_SPECz','UDSz_SPECz','SUBARU_SPECz','XUDS_SPECz','C3R2_SPECz']):
    
            if i<j:
    
                cond = (catalog[zlabel1]!=-99.) & (catalog[zlabel2]!=-99.)
                if "VIPERS" in zlabel1 or "VIPERS" in zlabel2: cond = cond & ((catalog['VIPERS_FLAG'].astype(int)==4) | (catalog['VIPERS_FLAG'].astype(int)==3))

                diff = (catalog[zlabel2][cond] - catalog[zlabel1][cond]) / (1+catalog[zlabel1][cond])
                ax1.scatter(catalog[zlabel1][cond],catalog[zlabel2][cond],s=15,alpha=1,label='%s vs. %s'%(zlabel1,zlabel2))
                ax2.scatter(catalog[zlabel1][cond],diff,s=15,alpha=1,label='%s vs. %s'%(zlabel1,zlabel2))
                
                crit = np.abs(catalog[zlabel1][cond] - catalog[zlabel2][cond]) / (1+catalog[zlabel1][cond]) > 0.02
                print "%12s vs. %12s: %2i/%4i (%6.2f%%) outliers" % (zlabel1,zlabel2,np.sum(crit.astype(int)),len(crit),100*np.sum(crit.astype(int))/float(len(crit)))
    
    ax1.plot([0,10],[0,10],c='k',ls='--',lw=0.8)
    ax1.fill_between([0,10],[-0.02,10-0.02*(1+10)],[+0.02,10+0.02*(1+10)],color='k',alpha=0.15)

    ax2.axhline(0,c='k',ls='--',lw=0.8)
    ax2.fill_between([0,10],[-0.02,-0.02],[0.02,0.02],color='k',alpha=0.15)

    ax1.set_xlim(0,7)
    ax1.set_ylim(0,7)
    ax1.set_ylabel("$z_{spec,2}$",fontsize=16)
    ax2.set_xlabel("$z_{spec,1}$",fontsize=16)
    ax2.set_ylabel("$z_{spec,2}-z_{spec,2}$ / 1+$z_{spec,1}$",fontsize=16)

    leg = ax1.legend(loc=4,fontsize=12,ncol=1)
    _ = [label.set_visible(False) for label in ax1.get_xticklabels()]

def test():

    catalog = fitsio.getdata("final_cats/final_catalog_errfix.extra.fits")

    zlabels = ['M14_GRISMz','M14_SPECz','VIPERS_SPECz','UDSz_SPECz','SUBARU_SPECz','COMP_SPECz','ALLSPEC_SPECz']

    zcond = np.zeros((len(catalog),len(zlabels)),dtype=int)

    for i,_zlabel in enumerate(zlabels):

        cond = catalog[_zlabel] != -99.
        if "VIPERS" in _zlabel: cond = cond & ((catalog['VIPERS_FLAG'].astype(int)==4) | (catalog['VIPERS_FLAG'].astype(int)==3))
        if "M14" in _zlabel: cond = cond & (catalog['M14_QUALITY'] > 2.0)
        idx = np.where(cond)[0]
        zcond[idx,i] = 1
        print _zlabel, np.sum(zcond[:,i])

    print np.sum(np.any(zcond,axis=1))

    for i in range(len(zlabels)):

        _zcond = np.delete(zcond,i,axis=1)
        print zlabels[i], np.sum(np.logical_or(zcond[:,i].astype(bool),np.any(_zcond,axis=1))) - \
                          np.sum(np.logical_and(zcond[:,i].astype(bool),np.any(_zcond,axis=1)))

    for i in range(len(zlabels)):

        for j in range(i+1,len(zlabels)):

            print "%15s (%5i) vs. %15s (%5i) -- %5i" % (zlabels[i], np.sum(zcond[:,i]), zlabels[j], np.sum(zcond[:,j]), np.sum(np.logical_and(zcond[:,i],zcond[:,j])))

if __name__ == '__main__':
    
    # mk_zspec_plot()

    test()
    plt.show()
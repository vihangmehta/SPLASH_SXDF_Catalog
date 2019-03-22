import os, argparse
import numpy as np
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import useful

def distance_checker(entry,data,crit):
    dist = np.sqrt((data['X_IMAGE']-entry['X_IMAGE'])**2 + (data['Y_IMAGE']-entry['Y_IMAGE'])**2)
    srtd = np.sort(dist)
    assert srtd[0] == 0
    mind = srtd[1]
    return mind > crit/useful.pix_scale

def mk_sex_cat(data_dir,cat_dir,conv_dir=None):

    def call(filt):

        if 'conv' not in conv_dir:

            img_name = os.path.join(data_dir,"mosaic_%s_%s.img.fits" % (instr,filt))
            wht_name = os.path.join(data_dir,"mosaic_%s_%s.wht.fits" % (instr,filt))
            cat_name = os.path.join(cat_dir, "mosaic_%s_%s.ldac"     % (instr,filt))
            zp = useful.zp #[instr][filt]

            call = "sextractor %s -c config/config_psfex.sex " \
                   "-PARAMETERS_NAME config/param_psfex.sex " \
                   "-CATALOG_NAME %s -CATALOG_TYPE FITS_LDAC " \
                   "-WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE %s " \
                   "-MAG_ZEROPOINT %.2f" % (img_name,cat_name,wht_name,zp)

        else:

            img_name = os.path.join(data_dir,"mosaic_%s_%s.img.fits" % (instr,filt))
            wht_name = os.path.join(data_dir,"mosaic_%s_%s.wht.fits" % (instr,filt))
            img_conv_name = os.path.join(conv_dir,"mosaic_conv_%s_%s.img.fits" % (instr,filt))
            wht_conv_name = os.path.join(conv_dir,"mosaic_conv_%s_%s.wht.fits" % (instr,filt))
            cat_name = os.path.join(cat_dir, "mosaic_conv_%s_%s.ldac"     % (instr,filt))
            zp = useful.zp #[instr][filt]

            call = "sextractor %s,%s -c config/config_psfex.sex " \
                   "-PARAMETERS_NAME config/param_psfex.sex " \
                   "-CATALOG_NAME %s -CATALOG_TYPE FITS_LDAC " \
                   "-WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE %s,%s " \
                   "-MAG_ZEROPOINT %.2f" % (img_name,img_conv_name,cat_name,wht_name,wht_conv_name,zp)
        
        if instr == 'irac': call += " -RESCALE_WEIGHTS Y"

        if os.path.isfile(img_name) and os.path.isfile(wht_name):
            return call
        else:
            print "[mk_psfs.py] Warning! Input files not found for %s:%s" % (instr,filt)

    calls = []
    for instr in useful.instr_used_list[:-1]:
        for filt in useful.filters[instr]:
            calls.append(call(filt=filt))

    if len(calls) > 10:
        call_chunks = np.array_split(calls,len(calls)/10)
        for call_chunk in call_chunks:
            useful.multiprocess(call_chunk,cwd=cwd)
    else:
        useful.multiprocess(calls,cwd=cwd)

def mk_star_cat(instr,filt,cat_dir,manual,size_cuts,isolate=False):

    conv = '_conv' if 'conv' in cat_dir else ''
    cat_name = os.path.join(cat_dir,"mosaic%s_%s_%s.ldac" % (conv,instr,filt))
    
    if not os.path.isfile(cat_name):
        print "[mk_psfs.py] Warning! No catalog found for %s:%s" % (instr,filt)
        return
    
    cat_hdu = fitsio.open(cat_name)
    catalog = cat_hdu[2].data
    
    fig,ax = plt.subplots(1,1,figsize=(10,8),dpi=75,tight_layout=True)
    ax.set_title("Selecting stars for %s/%s" % (instr,filt),fontsize=16)
    
    patch = ax.add_patch(Polygon([[0,0],],color='r',lw=2,alpha=0.5,closed=True))
    ax.scatter(catalog['FLUX_RADIUS'],catalog['MAG_AUTO'],c='k',s=5,lw=0,alpha=0.5)
    ax.set_xlabel('Half-light radius (size) [px]')
    ax.set_ylabel('AUTO Magnitude [AB]')
    
    ax.set_ylim(28,12)
    ax.set_xlim(1,15)
    plt.draw()
    
    if manual:
        plt.show(block=False)
        proceed = 'n'
    else:
        proceed = 'y'
        _size_cuts, _mag_cuts = useful.msize_cuts()
        if size_cuts:
            size_cut = size_cuts
        else:
            size_cut = _size_cuts[instr][filt]
        mag_cut  =  _mag_cuts[instr][filt]
        verts = [[size_cut[0],mag_cut[0]],[size_cut[0],mag_cut[1]],[size_cut[1],mag_cut[1]],[size_cut[1],mag_cut[0]],[size_cut[0],mag_cut[0]]]
        patch = ax.add_patch(Polygon(verts,color='r',lw=2,alpha=0.5,closed=True))

    while proceed != 'y':
        
        print "Selecting stars for %s/%s" % (instr,filt)
        size_cut = [float(raw_input("Enter lower limit in size: ")),
                    float(raw_input("Enter upper limit in size: "))]
        mag_cut  = [float(raw_input("Enter lower limit in mag : ")),
                    float(raw_input("Enter upper limit in mag : "))]

        patch.remove()
        verts = [[size_cut[0],mag_cut[0]],[size_cut[0],mag_cut[1]],[size_cut[1],mag_cut[1]],[size_cut[1],mag_cut[0]],[size_cut[0],mag_cut[0]]]
        patch = ax.add_patch(Polygon(verts,color='r',lw=2,alpha=0.5,closed=True))
        plt.draw()
        proceed = raw_input("Continue? [y/n] ")
    
    fig.savefig('%s/plots/mosaic%s_%s_%s.stars.png' % (cat_dir,conv,instr,filt))

    cond = (size_cut[0]<catalog['FLUX_RADIUS']) & (catalog['FLUX_RADIUS']<size_cut[1]) & \
           ( mag_cut[0]<catalog['MAG_AUTO'])    & (catalog['MAG_AUTO']   < mag_cut[1])

    print "[mk_psfs.py] (%s/%s) Selecting %i stars out of %i sources "\
          "using %.2f<size<%.2f and %.2f<mag<%.2f" % (instr,filt,
                    len(cat_hdu[2].data[cond]),len(cat_hdu[2].data),size_cut[0],size_cut[1],mag_cut[0],mag_cut[1])

    if isolate:
        crit = 5 #arcsec
        cond2 = np.array([distance_checker(entry=entry,data=catalog,crit=crit) for entry in catalog[cond]],dtype=bool)
        print "[mk_psfs.py] (%s/%s) Selecting %i stars out of %i stars "\
              'with no source within %s"' % (instr,filt,len(cat_hdu[2].data[cond][cond2]),len(cat_hdu[2].data[cond]),crit)
    else:
        cond2 = np.ones(len(catalog[cond]),dtype=bool)

    np.savetxt('%s/mosaic%s_%s_%s.reg' % (cat_dir,conv,instr,filt),
                np.vstack((cat_hdu[2].data['X_IMAGE'],cat_hdu[2].data['Y_IMAGE'])).T,
                fmt='circle(%10.4f,%10.4f,10) # color=red width=1')

    cat_hdu[2].data = cat_hdu[2].data[cond][cond2]
    cat_hdu.writeto("%s/mosaic%s_%s_%s.stars.ldac" % (cat_dir,conv,instr,filt), overwrite=True)

    np.savetxt('%s/mosaic%s_%s_%s.stars.reg' % (cat_dir,conv,instr,filt),
                np.vstack((cat_hdu[2].data['X_IMAGE'],cat_hdu[2].data['Y_IMAGE'])).T,
                fmt='circle(%10.4f,%10.4f,10) # color=green width=2')

    plt.close(fig)

def mk_star_cat2(cat_dir):

    conv = '_conv' if 'conv' in cat_dir else ''
    stars_x,stars_y = np.zeros(0), np.zeros(0)

    for instr in ['uds','video']:

        for filt in useful.filters[instr]:

            cat_name = os.path.join(cat_dir,"mosaic%s_%s_%s.ldac" % (conv,instr,filt))
            cat_hdu = fitsio.open(cat_name)
            catalog = cat_hdu[2].data

            _size_cuts, _mag_cuts = useful.msize_cuts()
            size_cut = _size_cuts[instr][filt]
            mag_cut  =  _mag_cuts[instr][filt]

            cond = (size_cut[0]<catalog['FLUX_RADIUS']) & (catalog['FLUX_RADIUS']<size_cut[1]) & \
                   ( mag_cut[0]<catalog['MAG_AUTO'])    & (catalog['MAG_AUTO']   < mag_cut[1])

            cat_hdu[2].data = cat_hdu[2].data[cond]
            stars_x = np.append(stars_x,cat_hdu[2].data['X_IMAGE'])
            stars_y = np.append(stars_y,cat_hdu[2].data['Y_IMAGE'])

    # Find objects within 3 px of each other and set them as the same object
    crit_dist = 3. # pixels
    cand_x,cand_y = np.zeros(0),np.zeros(0)
    mask = np.zeros(len(stars_x),dtype=int)
    for i in range(len(mask)):
        if mask[i]!=1:
            x, y = stars_x[i],stars_y[i]
            dist = np.sqrt((stars_x-x)**2 + (stars_y-y)**2)
            cand = (dist <= crit_dist)
            mask[cand] = 1
            x, y = np.mean(stars_x[cand]),np.mean(stars_y[cand])
            cand_x,cand_y = np.append(cand_x,x),np.append(cand_y,y)

    # Find objects within 5" of candidate and if you do, throw the candidate out
    crit_dist = 5. #arcsec
    mask = np.zeros(len(cand_x),dtype=bool)
    for i in range(len(mask)):
        dist = np.sqrt((cand_x-cand_x[i])**2 + (cand_y-cand_y[i])**2)
        srtd = np.sort(dist)
        assert srtd[0] == 0
        mind = srtd[1]
        mask[i] = mind > crit_dist/useful.pix_scale
    cand_x,cand_y = cand_x[mask],cand_y[mask]

    print len(cand_x)

    dtype = []
    for instr in useful.instr_used_list[:-1]:
        for filt in useful.filters[instr]:
            dtype.append(("%s_%s"%(instr,filt),int))
    m_idx = np.recarray(len(cand_x),dtype=dtype)
    for x in m_idx.dtype.names: m_idx[x] = -99.

    gcond = True
    for fname in m_idx.dtype.names:
        cat_name = os.path.join(cat_dir,"mosaic%s_%s.ldac" % (conv,fname))
        cat_hdu = fitsio.open(cat_name)
        catalog = cat_hdu[2].data

        idx = useful.match_x_y(cand_x,cand_y,catalog['X_IMAGE'],catalog['Y_IMAGE'],r=1)
        cond = (idx != len(catalog))
        m_idx[fname][cond] = idx[cond]
        print fname, len(m_idx[fname][m_idx[fname]!=-99.])

        gcond = gcond & (m_idx[fname]!=-99.)
        
        fig,ax = plt.subplots(1,1,figsize=(10,8),dpi=75,tight_layout=True)
        ax.set_title("Selecting stars for %s" % fname,fontsize=16)
        ax.set_xlabel('Half-light radius (size) [px]')
        ax.set_ylabel('AUTO Magnitude [AB]')
        ax.set_ylim(28,12)
        ax.set_xlim(1,15)
        ax.scatter(catalog['FLUX_RADIUS'][idx[cond]],catalog['MAG_AUTO'][idx[cond]],c='k',s=5,lw=0,alpha=0.8)
        plt.show()

    print len(m_idx[gcond])
    
if __name__ == '__main__':
    
    cwd = "/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/"
    data_dir = os.path.join(cwd,'data','orig')

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig",help="",action='store_true')
    parser.add_argument("--conv",help="",action='store_true')

    parser.add_argument("--sex"   ,help="runs SExtractor",action='store_true')
    parser.add_argument("--cut"   ,help="select stars in the SExtractor catalogs",action='store_true')
    parser.add_argument("--manual",help="manually select stars",action='store_true')
    args = parser.parse_args()

    cat_dirs   = np.array([   'orig',    'conv'])
    size_cuts  = np.array([     None, [2.4,3.5]])
    flags      = np.array([args.orig, args.conv],dtype=bool)

    for _cat_dir,_size_cuts in zip(cat_dirs[flags],size_cuts[flags]):

        conv_dir = os.path.join(cwd,'data',_cat_dir)
        cat_dir  = os.path.join(cwd,'psfex_cats',_cat_dir)

        if args.sex:
            mk_sex_cat(data_dir=data_dir,cat_dir=cat_dir,conv_dir=conv_dir)
    
        if args.cut:
            for instr in useful.instr_used_list[:-1]:
                for filt in useful.filters[instr]:
                    mk_star_cat(instr=instr,filt=filt,cat_dir=cat_dir,manual=args.manual,size_cuts=_size_cuts)
    
    # mk_star_cat2(cat_dir=os.path.join(cwd,'psfex_cats','orig'))
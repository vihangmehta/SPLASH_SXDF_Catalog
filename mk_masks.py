import os, sys
import numpy as np
import astropy.io.fits as fitsio
from scipy import ndimage

import useful

def select_only_big_clusters(image):

    unique_val = 256
    clusters, cluster_count = ndimage.label(image == unique_val)
    clusters = clusters.astype(np.uint16)
    ones = np.ones_like(image, dtype=np.int8)
    cluster_sizes = ndimage.sum(ones, labels=clusters, index=np.arange(cluster_count)+1).astype(np.uint32)
    idx = np.arange(cluster_count)[cluster_sizes>1000000] + 1

    image[:,:] = 0
    for i in idx: image[clusters==i] = 256
    return image

def mk_mask(data_dir,instr,filt):

    sys.stdout.write("Creating Mask for %s:%s ... " % (instr,filt),)
    sys.stdout.flush()

    if instr=='hsc':
        
        omsk = fitsio.getdata(os.path.join(data_dir,'mosaic_%s_%s.omsk.fits'%(instr,filt)),memmap=True)
        omsk = omsk.astype(np.uint16)
        msk  = np.zeros(omsk.shape,dtype=np.uint16)
        msk[(omsk&512)==512] = 256       # Bright stars
        msk = select_only_big_clusters(msk)
        msk[(omsk&260)==260] = 256       # Edges
        
        wht,hdr = fitsio.getdata(os.path.join(data_dir,'mosaic_%s_%s.owht.fits'%(instr,filt)),header=True,memmap=True)
        msk[ wht == 0 ] = 256            # Still mask everything with weight of 0
        fitsio.writeto(os.path.join(data_dir,'mosaic_%s_%s.msk.fits'%(instr,filt)),msk,header=hdr,overwrite=True)

    else:
        
        wht,hdr = fitsio.getdata(os.path.join(data_dir,'mosaic_%s_%s.wht.fits'%(instr,filt)),header=True,memmap=True)
        msk = np.zeros(wht.shape,dtype=np.uint16)
        msk[ wht == 0 ] = 256
        fitsio.writeto(os.path.join(data_dir,'mosaic_%s_%s.msk.fits'%(instr,filt)),msk,header=hdr,overwrite=True)

    print "done"

def mk_masked_hsc_weight_maps(data_dir):

    for filt in useful.filters["hsc"]:

        print "Masking weight map for %s:%s ... " % ('hsc',filt)
        msk = fitsio.getdata(os.path.join(data_dir,'mosaic_%s_%s.msk.fits'%('hsc',filt)))
        wht,hdr = fitsio.getdata(os.path.join(data_dir,'mosaic_%s_%s.owht.fits'%('hsc',filt)),header=True)
        wht[msk==256] = 0
        fitsio.writeto(os.path.join(data_dir,'mosaic_%s_%s.wht.fits'%('hsc',filt)),wht,header=hdr,overwrite=True)

# def mk_final_mask(data_dir):

#     gmsk = np.zeros((50000,50000),dtype=np.unit16)

#     for filt in useful.filters["hsc"]:

#         msk,hdr = fitsio.getdata(os.path.join(data_dir,'mosaic_%s_%s.msk.fits'%('hsc',filt)),header=True)
#         gmsk[msk==256] = 1

#     fitsio.writeto(os.path.join(data_dir,'mosaic_final.msk.fits'),gmsk,header=hdr,overwrite=True)

if __name__ == '__main__':
    
    parent_dir = '/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/'
    data_dir = os.path.join(parent_dir,'data/orig/')

    # for instr in useful.instr_used_list[:1]:
    #     for filt in useful.filters[instr]:
    #         mk_mask(data_dir=data_dir,instr=instr,filt=filt)

    mk_masked_hsc_weight_maps(data_dir=data_dir)
    # mk_final_mask(data_dir=data_dir)
import os, time, cv2, argparse
import numpy as np
import astropy.io.fits as fitsio

import useful

def do_fft_convolution(instr,filt,data_dir,ker_dir,conv_dir):

    if instr=='irac':
        print "Not convolving %s:%s ... " %(instr,filt)
        src = os.path.join(cwd,data_dir,'mosaic_%s_%s.img.fits'%(instr,filt))
        dst = os.path.join(cwd,conv_dir,'mosaic_conv_%s_%s.img.fits'%(instr,filt))
        print "Linking %s to %s" % (src,dst)
        useful.force_symlink(src,dst)
        return

    print "Convolving %s:%s " % (instr,filt)

    img, hdr = fitsio.getdata(os.path.join(data_dir,'mosaic_%s_%s.img.fits'%(instr,filt)), header=True)
    ker = fitsio.getdata(os.path.join(ker_dir,'mosaic_%s_%s.stars.homo.fits'%(instr,filt)))
    
    img = np.array(img,dtype=np.float32)
    ker = np.array(ker,dtype=np.float32)

    start = time.time()
    conv_img = cv2.filter2D(img,-1,cv2.flip(ker,-1),anchor=(-1,-1))
    conv_img = conv_img.astype(np.float32)
    end = time.time()

    print "Time for %s:%s" % (instr,filt)
    print "Convolution: %.2fs" % (end-start)

    start = time.time()
    hdr.set('OBJECT','%s %s (convolved)' % (instr,filt))
    fitsio.writeto(os.path.join(conv_dir,'mosaic_conv_%s_%s.img.fits' % (instr,filt)), conv_img, hdr, overwrite=True)
    end = time.time()
    
    print "Write out: %.2fs" % (end-start)

def mk_wht_links(instr,filt,data_dir,conv_dir):

    src = os.path.join(cwd,data_dir,'mosaic_%s_%s.wht.fits'%(instr,filt))
    dst = os.path.join(cwd,conv_dir,'mosaic_conv_%s_%s.wht.fits'%(instr,filt))
    print "Linking %s to %s" % (src,dst)
    useful.force_symlink(src,dst)

if __name__ == '__main__':
    
    cwd = '/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/'
    data_dir = 'data/orig/'

    ker_dir  = os.path.join(cwd,'kernels','conv')
    conv_dir = os.path.join(cwd,'data'   ,'conv')

    for instr in useful.instr_used_list[:1]:
        for filt in useful.filters[instr]:
            do_fft_convolution(instr=instr,filt=filt,data_dir=data_dir,ker_dir=ker_dir,conv_dir=conv_dir)
            mk_wht_links(instr=instr,filt=filt,data_dir=data_dir,conv_dir=conv_dir)

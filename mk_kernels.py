import os, argparse
import numpy as np

try:
    from pyraf import iraf
    from pyraf.iraf import psfmatch
except ImportError:
    raise Exception("No IRAF available.")

import useful

def mk_kernel(instr,filt,instr0,filt0,psf_dir,ker_dir):

    os.chdir(os.path.join(cwd,psf_dir))
    inpsf  = "mosaic_%s_%s.stars.norm_psf.fits" % (instr,filt)
    outpsf = "mosaic_%s_%s.stars.norm_psf.fits" % (instr0,filt0)
    outker = "mosaic_%s_%s.stars.homo.fits"     % (instr,filt)

    if os.path.exists(outker+'.fits'): os.remove(outker)
    psfmatch(inpsf, outpsf, inpsf, outker, convolution='psf', background='none', threshold=0.2)
    os.rename(os.path.join(cwd,psf_dir,outker),os.path.join(cwd,ker_dir,outker))
    os.chdir(cwd)

if __name__ == '__main__':

    cwd = '/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/'

    psf_dir = 'psfex/orig_pixel/'
    ker_dir = 'kernels/conv_pixel/'

    for instr in useful.instr_used_list:
        for filt in useful.filters[instr]:
            mk_kernel(instr=instr,filt=filt,instr0='uds',filt0='k',psf_dir=psf_dir,ker_dir=ker_dir)

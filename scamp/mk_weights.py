import os
import numpy as np
import astropy.io.fits as fitsio

import useful

def mk_irac_whtmap():

    instr = 'irac'

    for filt in useful.filters[instr]:

        print "%s:%s" % (instr,filt)
        unc,hdr = fitsio.getdata("%s/tile_%s_%s_1.unc.fits"%(instr,instr,filt),header=True)
        wht = np.zeros(unc.shape,dtype=np.float32)

        cond_nan = np.isfinite(unc)
        wht[cond_nan] = 1./unc[cond_nan]**2

        hdr["PRODTYPE"] = "WEIGHTS"
        fitsio.writeto("%s/tile_%s_%s_1.wht.fits"%(instr,instr,filt),data=wht,header=hdr)

if __name__ == '__main__':
    
    mk_irac_whtmap()
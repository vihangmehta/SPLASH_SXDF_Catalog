import os
import numpy as np

import useful

cwd = "/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/scamp/"

def mk_sex_cat():

    def call(instr,filt,tile=None):

        if not tile:
            img_det_name = "%s/premosaic_%s.img.fits" % (instr,instr)
            wht_det_name = "%s/premosaic_%s.wht.fits" % (instr,instr)
            img_pht_name = "%s/premosaic_%s_%s.img.fits" % (instr,instr,filt)
            wht_pht_name = "%s/premosaic_%s_%s.wht.fits" % (instr,instr,filt)
            cat_name = "%s/catalog_%s.ldac" % (instr,instr)
            zp = useful.orig_zp[instr][filt]
        else:
            img_det_name = "%s/tile_%s_%i.img.fits" % (instr,instr,tile)
            wht_det_name = "%s/tile_%s_%i.wht.fits" % (instr,instr,tile)
            img_pht_name = "%s/tile_%s_%s_%i.img.fits" % (instr,instr,filt,tile)
            wht_pht_name = "%s/tile_%s_%s_%i.wht.fits" % (instr,instr,filt,tile)
            cat_name = "%s/catalog_%s_%s_%s.ldac" % (instr,instr,filt,tile)
            zp = useful.orig_zp[instr][filt][tile]

        call = "sextractor %s,%s -c config/config_scamp.sex " \
               "-PARAMETERS_NAME config/param_scamp.sex " \
               "-CATALOG_NAME %s -CATALOG_TYPE FITS_LDAC " \
               "-WEIGHT_TYPE MAP_WEIGHT,MAP_WEIGHT -WEIGHT_IMAGE %s,%s " \
               "-MAG_ZEROPOINT %.6f" % (img_det_name,img_pht_name,cat_name,wht_det_name,wht_pht_name,zp)
        
        if instr == 'irac': call += " -RESCALE_WEIGHTS Y"

        if os.path.isfile(img_det_name) and os.path.isfile(wht_det_name) and \
           os.path.isfile(img_pht_name) and os.path.isfile(wht_pht_name):
            return call
        else:
            print "[mk_psfs.py] Warning! Input files not found for %s:%s" % (instr,filt)

    # Rest of the instrs
    calls = []
    instr = ['hsc','cfht','cfhtls','uds','video','irac']
    filts = [  'g',   'u',     'z',  'j',    'j',   '1']
    for instr,filt in zip(instr,filts)[:1]:
        calls.append(call(instr,filt))
    useful.multiprocess(calls,cwd=cwd)

    # SupCam
    # calls  = [call(instr='supcam',filt='b',tile=i) for i in np.arange(5)+1]
    # calls += [call(instr='supcam',filt='v',tile=i) for i in np.arange(5)+1]
    # calls += [call(instr='supcam',filt='r',tile=i) for i in np.arange(5)+1]
    # calls += [call(instr='supcam',filt='i',tile=i) for i in np.arange(5)+1]
    # calls += [call(instr='supcam',filt='z',tile=i) for i in np.arange(5)+1]
    # call_chunks = np.array_split(calls,len(calls)/10)
    # for call_chunk in call_chunks:
    #     useful.multiprocess(call_chunk,cwd=cwd)

if __name__ == '__main__':
    
    mk_sex_cat()
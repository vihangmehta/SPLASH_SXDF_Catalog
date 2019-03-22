import os, argparse
import numpy as np
import astropy.io.fits as fitsio
from multiprocessing import Queue, Process

import useful

filters = {'hsc'   : ['g','r','i','z','y'],
           'supcam': ['b','v','r','i','z'],
           'cfht'  : ['u',],
           'cfhtls': ['u','g','r','i','z'],
      'conv_cfhtls': ['u','g','r','i','z'],
           'uds'   : ['j','h','k'],
           'video' : ['z','y','j','h','ks'],
           'irac'  : ['1','2','3','4']}

cwd = "/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/scamp/"

def calc_fscale(zp0,zp1=23.93):
    """
    -2.5*log(f1) + zp1 = -2.5*log(f0) + zp0
    f1/f0 = 10**((zp1 - zp0) / 2.5)
    """
    fscale = 10**((zp1 - zp0) / 2.5)
    return fscale

def mk_premosaics():
    
    def get_file_lists(instr,filt):

        img_list,wht_list,msk_list = [],[],[]
        for x in sorted(os.listdir(instr)):
            if 'tile_%s_%s'%(instr,filt) in x and '.img' in x:
                img_list.append('%s/%s' % (instr,x))
                if instr=='irac':
                    wht_list.append('%s/%s' % (instr,x.replace('img','unc')))
                else:
                    wht_list.append('%s/%s' % (instr,x.replace('img','wht')))
                msk_list.append('%s/%s' % (instr,x.replace('img','msk')))

        img_list = ','.join(img_list)
        wht_list = ','.join(wht_list)
        msk_list = ','.join(msk_list)

        if instr=='conv_cfhtls':
            img_list = img_list.replace("conv_cfhtls/","cfhtls/")
            wht_list = wht_list.replace("conv_cfhtls/","cfhtls/")
            msk_list = msk_list.replace("conv_cfhtls/","cfhtls/")

        return img_list, wht_list, msk_list

    def call(instr,filt):
        
        img_list, wht_list, msk_list = get_file_lists(instr,filt)
        outimg_name = "%s/premosaic_%s_%s.img.fits" % (instr,instr,filt)
        outwht_name = "%s/premosaic_%s_%s.wht.fits" % (instr,instr,filt)
        outmsk_name = "%s/premosaic_%s_%s.omsk.fits" % (instr,instr,filt)

        if instr=='conv_cfhtls':
            outimg_name = outimg_name.replace("conv_cfhtls/","cfhtls/")
            outwht_name = outwht_name.replace("conv_cfhtls/","cfhtls/")
            outmsk_name = outmsk_name.replace("conv_cfhtls/","cfhtls/")
        
        wht_type = "MAP_RMS" if instr=='irac' else "MAP_WEIGHT"
        if   instr=='hsc':  custom_config = "-CENTER_TYPE MANUAL -CENTER 34.5837,-4.8359 -IMAGE_SIZE 46500,46500 "
        elif instr=='irac': custom_config = "-CENTER_TYPE MANUAL -CENTER 34.501385,-4.998731 -IMAGE_SIZE 12800,12800 "
        elif instr=='video':custom_config = "-CENTER_TYPE MANUAL -CENTER 34.501385,-4.998731 -IMAGE_SIZE 40000,40000 "
        elif instr=='supcam':
            zp = [useful.orig_zp[instr][filt][tile] for tile in np.arange(5)+1]
            fscale = [calc_fscale(_zp,zp1=useful.orig_zp[instr][filt][1]) for _zp in zp]
            fscale = ','.join(map(lambda x: "%.6e" % x, fscale))
            custom_config = "-FSCALASTRO_TYPE FIXED -FSCALE_DEFAULT %s"%fscale
        else: custom_config = ""

        call_img = "swarp %s " \
                   "-c config/config_tiles.swarp " \
                   "-WEIGHT_TYPE %s -WEIGHT_IMAGE %s " \
                   "-COMBINE_TYPE WEIGHTED " \
                   "-IMAGEOUT_NAME %s " \
                   "-WEIGHTOUT_NAME %s %s" % (img_list,wht_type,wht_list,outimg_name,outwht_name,custom_config)

        call_msk = "swarp %s " \
                   "-c config/config_msk.swarp " \
                   "-IMAGEOUT_NAME %s " \
                   "-WEIGHTOUT_NAME ./tmp/tmp.fits " \
                   "-WEIGHT_TYPE NONE " \
                   "-COMBINE Y -COMBINE_TYPE MAX " \
                   "-RESAMPLE Y -RESAMPLING_TYPE NEAREST -FSCALASTRO_TYPE NONE " \
                   "-SUBTRACT_BACK N -BACK_TYPE MANUAL -BACK_DEFAULT 0 %s" % (msk_list,outmsk_name,custom_config)
        
        return call_img,call_msk

    for instr in ['hsc','supcam','video','cfhtls','conv_cfhtls','irac'][:1]:
        for filt in filters[instr]:
            call_img, call_msk = call(instr,filt)
            useful.run(call_img,cwd=cwd,verbose=True)
            # if instr=='hsc':
            #     useful.run(call_msk,cwd=cwd,verbose=True)
            #     os.remove('tmp/tmp.fits')

def mk_instr_mosaics():

    def get_file_lists(instr):

        img_list,wht_list = [],[]
        for x in sorted(os.listdir(instr)):
            if 'premosaic_%s_'%instr in x and '.img' in x:
                img_list.append('%s/%s' % (instr,x))
                wht_list.append('%s/%s' % (instr,x.replace('img','wht')))
        img_list = ','.join(img_list)
        wht_list = ','.join(wht_list)
        return img_list, wht_list

    def get_tile_lists(instr,tile):

        img_list,wht_list = [],[]
        for x in sorted(os.listdir(instr)):
            if 'tile_%s_'%instr in x and '_%i.img'%tile in x and 'tile_%s_%i.img'%(instr,tile) not in x:
                img_list.append('%s/%s' % (instr,x))
                wht_list.append('%s/%s' % (instr,x.replace('img','wht')))
        img_list = ','.join(img_list)
        wht_list = ','.join(wht_list)
        return img_list, wht_list

    def call(instr,tile=None):
        
        if tile:
            img_list, wht_list = get_tile_lists(instr,tile)
            out_img = "%s/tile_%s_%i.img.fits" % (instr,instr,tile)
            out_wht = "%s/tile_%s_%i.wht.fits" % (instr,instr,tile)
        else:
            img_list, wht_list = get_file_lists(instr)
            out_img = "%s/premosaic_%s.img.fits" % (instr,instr)
            out_wht = "%s/premosaic_%s.wht.fits" % (instr,instr)

        call_img = "swarp %s " \
                   "-c config/config_det.swarp " \
                   "-WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE %s " \
                   "-IMAGEOUT_NAME %s " \
                   "-WEIGHTOUT_NAME %s " \
                   "-COMBINE Y -COMBINE_TYPE CHI2 -RESAMPLE N -SUBTRACT_BACK Y " % (img_list,wht_list,out_img,out_wht)
        if instr=='cfht': call_img += "-RESCALE_WEIGHTS Y "
        return call_img

    for instr in ['hsc','irac','uds','video','cfht','cfhtls'][:1]:
        call_img = call(instr=instr)
        useful.run(call_img,cwd=cwd,verbose=True)

    # SupCam mosaics
    # call_img = [call(instr="supcam",tile=i) for i in np.arange(5)+1]
    # _ = [useful.run(call,cwd=cwd,verbose=True) for call in call_img]

def mk_final_mosaics():

    def call(instr,filt,tiles=False):
        
        if tiles:
            inp_files = ",".join(["%s/tile_%s_%s_%i.img.fits" % (instr,instr,filt,tile) for tile in np.arange(5)+1])
            wht_files = inp_files.replace("img","wht")
            zp = [useful.orig_zp[instr][filt][tile] for tile in np.arange(5)+1]
            fscale = [calc_fscale(_zp) for _zp in zp]
            fscale = ','.join(map(lambda x: "%.6e" % x, fscale))
        else:
            inp_files = "%s/premosaic_%s_%s.img.fits" % (instr,instr,filt)
            wht_files = inp_files.replace("img","wht")
            zp = useful.orig_zp[instr][filt]
            fscale = calc_fscale(zp)
            fscale = "%.6e"%fscale

        call_img = "swarp %s " \
                   "-c config/config_msc.swarp " \
                   "-WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE %s " \
                   "-IMAGEOUT_NAME mosaic_%s_%s.img.fits " \
                   "-WEIGHTOUT_NAME mosaic_%s_%s.wht.fits " \
                   "-COMBINE Y -COMBINE_TYPE MEDIAN -RESAMPLE Y -SUBTRACT_BACK N " \
                   "-CENTER_TYPE MANUAL -CENTER 34.50,-5.00 " \
                   "-PIXELSCALE_TYPE MANUAL -PIXEL_SCALE 0.15 -IMAGE_SIZE 50000,50000 " \
                   "-FSCALASTRO_TYPE FIXED -FSCALE_DEFAULT %s" % (inp_files,wht_files,instr,filt,instr,filt,fscale)

        inp_files = "%s/premosaic_%s_%s.omsk.fits" % (instr,instr,filt)
        call_msk = "swarp %s " \
                   "-c config/config_msk.swarp " \
                   "-IMAGEOUT_NAME mosaic_%s_%s.omsk.fits " \
                   "-WEIGHTOUT_NAME ./tmp/tmp.fits " \
                   "-WEIGHT_TYPE NONE " \
                   "-COMBINE Y -COMBINE_TYPE MAX " \
                   "-RESAMPLE Y -RESAMPLING_TYPE NEAREST -FSCALASTRO_TYPE NONE " \
                   "-SUBTRACT_BACK N -BACK_TYPE MANUAL -BACK_DEFAULT 0 " \
                   "-CENTER_TYPE MANUAL -CENTER 34.50,-5.00 " \
                   "-PIXELSCALE_TYPE MANUAL -PIXEL_SCALE 0.15 -IMAGE_SIZE 50000,50000" % (inp_files,instr,filt)
        
        return call_img,call_msk

    for instr in ['hsc','uds','video','cfht','cfhtls','conv_cfhtls','irac'][:1]:
        
        for filt in filters[instr]:
        
            call_img,call_msk = call(instr=instr,filt=filt)
        
            useful.force_symlink(os.path.join(cwd,"%s/catalog_%s.head"%(instr,instr)),
                                 os.path.join(cwd,"%s/premosaic_%s_%s.img.head"%(instr,instr,filt)))
            useful.run(call_img,cwd=cwd,verbose=True)
            os.remove("%s/premosaic_%s_%s.img.head"%(instr,instr,filt))

            # if instr=='hsc':

            #     useful.force_symlink(os.path.join(cwd,"%s/catalog_%s.head"%(instr,instr)),
            #                          os.path.join(cwd,"%s/premosaic_%s_%s.omsk.head"%(instr,instr,filt)))
            #     useful.run(call_msk,cwd=cwd,verbose=True)
            #     os.remove("%s/premosaic_%s_%s.omsk.head"%(instr,instr,filt))
            #     os.remove("tmp/tmp.fits")

    # instr = 'supcam'
    # for filt in filters[instr]:
    #     call_img = call(instr=instr,filt=filt,tiles=True)
    #     _ = [useful.force_symlink(os.path.join(cwd,"%s/catalog_%s_%s_%i.head"%(instr,instr,filt,tile)),
    #                               os.path.join(cwd,"%s/tile_%s_%s_%i.img.head"%(instr,instr,filt,tile))) for tile in np.arange(5)+1]
    #     useful.run(call_img,cwd=cwd,verbose=True)
    #     _ = [os.remove(os.path.join(cwd,"%s/tile_%s_%s_%i.img.head"%(instr,instr,filt,tile))) for tile in np.arange(5)+1]

def mk_irac_rms():

    instr = 'irac'
    for filt in filters[instr]:
        print "Processing weight maps for %s:%s ... " % (instr,filt),
        wht,hdr = fitsio.getdata(os.path.join(cwd,"mosaic_%s_%s.wht.fits"%(instr,filt)),header=True)
        rms = np.zeros(wht.shape,dtype=np.float32)
        cond = (wht!=0)
        rms[cond] = np.sqrt(1./wht[cond])
        fitsio.writeto(os.path.join(cwd,"mosaic_%s_%s.rms.fits"%(instr,filt)),data=rms,header=hdr,overwrite=True)
        print "done."

if __name__ == '__main__':
    
    # mk_premosaics()
    # mk_instr_mosaics()
    # mk_final_mosaics()
    # mk_irac_rms()

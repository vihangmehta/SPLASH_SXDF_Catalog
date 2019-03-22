import os, argparse
import time
import subprocess
import numpy as np
import astropy.io.fits as fitsio

import useful

def mk_psf(cat_dir,psf_dir,basis_type,kernel_dir):

    conv = "_conv" if "conv" in cat_dir else ""
    cat_names = []
    for instr in useful.instr_used_list[:-2]:
        for filt in useful.filters[instr]:
            cat_names.append(os.path.join(cat_dir,"mosaic%s_%s_%s.stars.ldac"%(conv,instr,filt)))

    cat_names = ','.join(cat_names)
    xml_name = 'psfex_%s.xml'%psf_dir.split('/')[1]
    
    chkimg_list = ['chi.fits','proto.fits','samp.fits','resi.fits','snap.fits']
    chkplt_list = ['selfwhm','fwhm','ellipticity','counts','countfrac','chi2','resi']

    if kernel_dir:
        homo_config = "-HOMOBASIS_TYPE GAUSS-LAGUERRE "\
                      "-HOMOPSF_PARAMS 4.67,2.8 " \
                      "-HOMOKERNEL_DIR %s " % kernel_dir
    else:
        homo_config = ""

    call = "psfex %s -c config/config.psfex " \
           "-BASIS_TYPE %s "\
           "-PSF_SIZE 101,101 -PSF_DIR %s %s " \
           "-WRITE_XML Y -XML_NAME %s " \
           "-CHECKIMAGE_NAME %s " \
           "-CHECKPLOT_NAME %s" % (cat_names, basis_type, psf_dir, homo_config, xml_name,
                                   ','.join([os.path.join(psf_dir,i) for i in chkimg_list]),
                                   ','.join([os.path.join(psf_dir,'plots/',i) for i in chkplt_list]))
    useful.run(call,cwd=cwd,verbose=True)

def run(call,cwd):
    
    print "Processing %s ... " % call["fname"]
    start = time.time()
    f = open(os.path.join(cwd,call["log"]), 'w')
    p = subprocess.Popen(call["call"], stdout=f, stderr=f, cwd=cwd, shell=True)
    p.communicate()
    p.wait()
    end = time.time()
    f.write("\n\nTime taken: %.2f seconds"%(end-start))
    return (call, end - start)

def cb_func(x):

    call,time = x
    print "Finished %s in %.2fs." % (call["fname"],time)

def mk_psf2(cat_dir,psf_dir,basis_type,kernel_dir):

    conv = "_conv" if "conv" in cat_dir else ""

    chkimg_list = ['chi.fits','proto.fits','samp.fits','resi.fits','snap.fits']
    chkplt_list = ['selfwhm','fwhm','ellipticity','counts','countfrac','chi2','resi']

    if kernel_dir:
        homo_config = "-HOMOBASIS_TYPE GAUSS-LAGUERRE "\
                      "-HOMOPSF_PARAMS 4.67,2.8 " \
                      "-HOMOKERNEL_DIR %s " % kernel_dir
    else:
        homo_config = ""

    instrs = useful.instr_used_list[:-1] if "conv" in cat_dir else useful.instr_used_list[:-2]
    
    calls = []
    for instr in instrs:
        for filt in useful.filters[instr]:
        
            fname = "%s_%s"%(instr,filt)
            cat_name = os.path.join(cat_dir,"mosaic%s_%s.stars.ldac"%(conv,fname))
            xml_name = os.path.join(psf_dir,'psfex%s_%s.xml'%(conv,fname))
            log_name = os.path.join(psf_dir,'psfex%s_%s.log'%(conv,fname))
        
            call = "psfex %s -c config/config.psfex " \
                   "-BASIS_TYPE %s "\
                   "-PSF_SIZE 101,101 -PSF_DIR %s %s " \
                   "-WRITE_XML Y -XML_NAME %s " \
                   "-CHECKIMAGE_NAME %s " \
                   "-CHECKPLOT_NAME %s" % (cat_name, basis_type, psf_dir, homo_config, xml_name,
                                           ','.join([os.path.join(psf_dir,i) for i in chkimg_list]),
                                           ','.join([os.path.join(psf_dir,'plots/',i) for i in chkplt_list]))

            calls.append({"call":call,"log":log_name,"fname":fname})

    async_run = useful.AsyncFactory(run, cb_func)
    for call in calls: async_run.call(call=call,cwd=cwd)     
    async_run.wait()

def fix_psf(psf_dir):

    psf_names = [os.path.join(psf_dir,x) for x in os.listdir(psf_dir) if ".stars.psf" in x]

    for psf_name in psf_names:
        
        hdu = fitsio.open(psf_name)
        hdr = hdu[1].header
        psf = hdu[1].data[0][0][0]
        
        pos = [(psf.shape[0]/2,psf.shape[1]/2),]
        flux = useful.calc_photometry(psf,pos,radius=3./2./useful.pix_scale)
        psf /= flux
        flux = useful.calc_photometry(psf,pos,radius=3./2./useful.pix_scale)
        
        prihdu = fitsio.PrimaryHDU(data=psf,header=hdr)
        hdul = fitsio.HDUList([prihdu,])
        hdul.writeto(psf_name.replace(".stars.psf",".stars.norm_psf.fits"),overwrite=True)

if __name__ == '__main__':

    cwd = "/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig",help="",action='store_true')
    parser.add_argument("--conv",help="",action='store_true')
    parser.add_argument("--fix-psf",help="",action='store_true')
    args = parser.parse_args()

    if args.orig:
        mk_psf2(cat_dir=os.path.join(cwd,'psfex_cats/orig/'),
               psf_dir='psfex/orig/',
               basis_type='PIXEL',
               kernel_dir='kernels/conv/')
        if args.fix_psf:
            fix_psf(psf_dir='psfex/orig/')

    if args.conv:
        mk_psf2(cat_dir=os.path.join(cwd,'psfex_cats/conv/'),
               psf_dir='psfex/conv/',
               basis_type='PIXEL',
               kernel_dir=None)
        if args.fix_psf:
            fix_psf(psf_dir='psfex/conv/')

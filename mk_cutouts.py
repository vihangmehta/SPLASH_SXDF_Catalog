import sys, os, argparse
import numpy as np
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import useful

def cutout_params():

    x0,x1 = [0,24500],[25500,50000]
    y0,y1 = [0,24500],[25500,50000]
    yy0,xx0 = np.meshgrid(y0,x0)
    yy1,xx1 = np.meshgrid(y1,x1)
    cutout_verts = list(zip(xx0.flatten(),xx1.flatten(),yy0.flatten(),yy1.flatten()))
    return cutout_verts

def mk_cutout(index,img,hdr,fname,data_dir):

    x0,x1,y0,y1 = cutout_verts[index]
    cutout = img[y0:y1,x0:x1]
    newhdr = hdr.copy()
    newhdr['CRPIX1'] -= x0
    newhdr['CRPIX2'] -= y0
    if "det" in fname: outname = "cutout%i_%s" % (index+1,fname)
    else: outname = fname.replace('mosaic','cutout%i'%(index+1))
    fitsio.writeto(os.path.join(data_dir,outname), data=cutout, header=newhdr, overwrite=True)

def mk_cutouts(instr,filt,data_dir):

    if instr is None and filt is None:
        fnames = ["det.img.fits", "det.wht.fits"]
        instr, filt = "Detection", "Image"
    elif 'conv' in data_dir:
        fnames = ["mosaic_conv_%s_%s.img.fits"%(instr,filt),
                  "mosaic_conv_%s_%s.wht.fits"%(instr,filt)]
    else:
        fnames = ["mosaic_%s_%s.img.fits"%(instr,filt),
                  "mosaic_%s_%s.wht.fits"%(instr,filt),
                  "mosaic_%s_%s.msk.fits"%(instr,filt)]

    for fname in fnames:

        sys.stdout.write("Cutouts for %s " % fname)
        sys.stdout.flush()

        if "conv" in data_dir and (".wht" in fname or ".msk" in fname):
            # Just link the weight and mask files for the convolved version
            for index in range(len(cutout_verts)):

                src = os.path.join(os.path.dirname(data_dir),'orig/cutout%i_%s_%s.wht.fits'%(index+1,instr,filt))
                dst = os.path.join(data_dir,'cutout%i_conv_%s_%s.wht.fits'%(index+1,instr,filt))
                useful.force_symlink(src,dst)
                sys.stdout.write("\rCutouts for %s - %i out of %i - " % (fname,index+1,len(cutout_verts)))
                sys.stdout.flush()

        else:
            # Cutout the science images
            img = fitsio.getdata(os.path.join(data_dir,fname))
            hdr = fitsio.getheader(os.path.join(data_dir,fname))

            for index in range(len(cutout_verts)):

                mk_cutout(index,img=img,hdr=hdr,fname=fname,data_dir=data_dir)
                sys.stdout.write("\rCutouts for %s - %i out of %i - " % (fname,index+1,len(cutout_verts)))
                sys.stdout.flush()

        print ("done!")

if __name__ == '__main__':

    cwd = "/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/"
    data_dir = os.path.join(cwd,'data')

    parser = argparse.ArgumentParser()
    parser.add_argument("--det",  help="",action='store_true')
    parser.add_argument("--orig", help="",action='store_true')
    parser.add_argument("--conv", help="",action='store_true')
    args = parser.parse_args()

    cutout_verts = cutout_params()

    if args.det:
        mk_cutouts(instr=None,filt=None,data_dir=data_dir)

    if args.orig:
        _data_dir = 'orig'
        for instr in useful.instr_used_list[:1]:
            for filt in useful.filters[instr]:
                mk_cutouts(instr=instr,filt=filt,data_dir=os.path.join(data_dir,_data_dir))

    if args.conv:
        _data_dir = 'conv'
        for instr in useful.instr_used_list[:1]:
            for filt in useful.filters[instr]:
                mk_cutouts(instr=instr,filt=filt,data_dir=os.path.join(data_dir,_data_dir))

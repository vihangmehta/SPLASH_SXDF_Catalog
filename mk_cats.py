import os, argparse
import numpy as np

import useful

def mk_seg_map():

    call = lambda i: "sextractor data/cutout%i_det.img.fits -c config/config.sex " \
                     "-PARAMETERS_NAME config/param_seg.sex " \
                     "-CATALOG_TYPE FITS_1.0 -CATALOG_NAME data/catalog%i_det.fits " \
                     "-WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE data/cutout%i_det.wht.fits " \
                     "-CHECKIMAGE_TYPE SEGMENTATION -CHECKIMAGE_NAME data/cutout%i_det.seg.fits" % (i+1,i+1,i+1,i+1)

    useful.multiprocess([call(x) for x in range(4)],cwd=cwd)

def mk_sex_cat(data_dir,conv_dir,cat_dir):

    zp = useful.zp #[instr][filt]

    call = lambda instr,filt,i: "sextractor %s/cutout%i_det.img.fits,%s/cutout%i_conv_%s_%s.img.fits " \
                                "-c config/config.sex " \
                                "-PARAMETERS_NAME config/param.sex " \
                                "-CATALOG_NAME %s/catalog%i_conv_%s_%s.fits -CATALOG_TYPE FITS_1.0 " \
                                "-WEIGHT_TYPE MAP_WEIGHT,MAP_WEIGHT " \
                                "-FLAG_IMAGE data/orig/cutout%i_%s_%s.msk.fits " \
                                "-WEIGHT_IMAGE %s/cutout%i_det.wht.fits,%s/cutout%i_conv_%s_%s.wht.fits " \
                                "-CHECKIMAGE_TYPE NONE " \
                                "-MAG_ZEROPOINT %.2f" % (
                                   data_dir,i+1,
                                   conv_dir,i+1,instr,filt,
                                    cat_dir,i+1,instr,filt,
                                            i+1,instr,filt,
                                   data_dir,i+1,
                                   conv_dir,i+1,instr,filt,
                                   zp)

    calls = []
    for instr in useful.instr_used_list[:-1]:
        for filt in useful.filters[instr]:
            calls.extend([call(instr=instr,filt=filt,i=i) for i in range(4)])
    
    calls_split = np.array_split(calls,6)
    for call_chunk in calls_split:
        useful.multiprocess(call_chunk,cwd=cwd)

if __name__ == '__main__':
    
    cwd = "/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/"
    data_dir = os.path.join(cwd, "data")
    orig_dir = os.path.join(data_dir, "orig")
    conv_dir = os.path.join(data_dir, "conv")
    cat_dir  = os.path.join(cwd, "final_cats")

    parser = argparse.ArgumentParser()
    parser.add_argument("--sex",  help="runs SExtractor",action='store_true')
    parser.add_argument("--seg",  help="runs SExtractor to only make the segmentaion map",action='store_true')
    args = parser.parse_args()

    if args.seg:
        mk_seg_map()
    
    if args.sex:
        mk_sex_cat(data_dir=data_dir,conv_dir=conv_dir,cat_dir=cat_dir)

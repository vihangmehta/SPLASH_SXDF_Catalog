import os, argparse

import useful

def mk_det_image(data_dir,output_dir):

    filt = ['hsc_g','hsc_r','hsc_z','hsc_y',
            'uds_j','uds_h','uds_k',
            'video_y','video_j','video_h','video_ks',
            'cfht_u',
            'cfhtls_u']#,'cfhtls_g','cfhtls_r','cfhtls_i']
    
    fnames = ["%s/mosaic_%s.img.fits"%(data_dir,i) for i in filt]
    wnames = ["%s/mosaic_%s.wht.fits"%(data_dir,i) for i in filt]
    wtypes = ['MAP_WEIGHT'] * len(fnames)

    call = "swarp %s -c config/config.swarp " \
           "-IMAGEOUT_NAME %s/det.img.fits " \
           "-WEIGHTOUT_NAME %s/det.wht.fits " \
           "-WEIGHT_TYPE %s -WEIGHT_IMAGE %s -RESCALE_WEIGHTS N " \
           "-COMBINE Y -COMBINE_TYPE CHI2 " \
           "-RESAMPLE N -SUBTRACT_BACK Y " % (','.join(fnames),
                                              output_dir,
                                              output_dir,
                                              ','.join(wtypes),
                                              ','.join(wnames))

    useful.run(call,cwd=cwd,verbose=True)

if __name__ == '__main__':
    
    cwd        = "/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/"
    data_dir   = "/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/data/orig/"
    output_dir = "/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/data/"

    mk_det_image(data_dir=data_dir,output_dir=output_dir)

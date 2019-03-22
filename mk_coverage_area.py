import numpy as np
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt

import useful

cov_area = {"hsc_g"   :2.4664580434,"hsc_r"   :2.4211205954,"hsc_i"   :2.60862148611,"hsc_z"   :2.63250833854,"hsc_y"   :2.66619597222,
            "supcam_b":1.2242506597,"supcam_v":1.2242490468,"supcam_r":1.22424820139,"supcam_i":1.22424957813,"supcam_z":1.22424480903,
            "uds_j"   :0.8571405659,"uds_h"   :0.8555400312,"uds_k"   :0.85652020312,
            "video_z" :1.7603234809,"video_y" :2.5930888281,"video_j" :2.49329090104,"video_h" :2.49368863194,"video_ks":2.49440687847,
            "cfht_u"  :1.6531579461,
            "cfhtls_u":4.3385876093,"cfhtls_g":4.3339804809,"cfhtls_r":4.33165770833,"cfhtls_i":4.33215893229,"cfhtls_z":4.33664773958,
            "irac_1"  :4.2339995555,"irac_2"  :4.2461460277,"irac_3"  :3.32062544444,"irac_4"  :3.32055294444}


def get_coverage_area(instr,filt,new=False):

    if new:

        if instr!='irac':
            img = fitsio.getdata("/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/data/orig/mosaic_%s_%s.msk.fits"%(instr,filt))
            npix = (img.shape[0]*img.shape[1]) - np.sum(img.astype(bool))
            area = npix * (useful.pix_scale/3600.)**2
        else:
            img = fitsio.getdata("/data/highzgal/PUBLICACCESS/SPLASH/DATA/IRAC/SXDS.irac.%s.mosaic.fits"%filt)
            npix = np.sum(np.isfinite(img))
            area = npix * (0.60/3600.)**2

    else:

        fname = "%s_%s"%(instr,filt)
        area = cov_area[fname]

    return area

def get_coverage_areas():

    for instr in useful.instr_used_list:

        for filt in useful.filters[instr]:

            area = get_coverage_area(instr,filt)

            fname = "%s_%s"%(instr,filt)
            print ("%9s %.6f" % (fname,area))

if __name__ == '__main__':

    get_coverage_areas()

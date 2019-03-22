import sys,os
import numpy as np
import astropy.io.fits as fitsio

from mk_cutouts import cutout_params

cutout_verts = cutout_params()

def mk_final_segm():

    catalog = fitsio.getdata(os.path.join(cwd,"final_cats","final_catalog.fits"))
    segm_full = np.zeros((50000,50000))
    
    segm_cut = {}
    for i in range(4):
        segm_cut[i+1] = fitsio.getdata(os.path.join(cwd,"data","cutout%i_det.seg.fits"%(i+1)))

    for entry in catalog:

        sys.stdout.write("\rProcessing obj#%i out of %i (%.3f%%) ... \033[K" % (entry["ID"],len(catalog),entry["ID"]/float(len(catalog)*100.)))
        sys.stdout.flush()

        cond = (segm_cut[entry["cutout_id"]] == entry["cutout_num"])

        x0,x1,y0,y1 = cutout_verts[entry["cutout_id"]-1]
        segm_full[y0:y1,x0:x1][cond] = entry["ID"]

    print "done."

    fitsio.writeto(os.path.join(cwd,"data","det.seg.fits"),segm_full,overwrite=True)

if __name__ == '__main__':
    
    cwd = "/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/"
    mk_final_segm()
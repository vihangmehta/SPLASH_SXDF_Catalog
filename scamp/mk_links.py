import os
import numpy as np
import astropy.io.fits as fitsio

from useful import force_symlink

parent_dir = '/data/highzgal/PUBLICACCESS/SPLASH/DATA/'

def extract_extension(input_path, output_path, n, var_to_wht=False):

    with fitsio.open(input_path, mode='readonly') as hdu:
        primary_header    = hdu[0].header
        extension_data    = hdu[n].data
        extension_header  = hdu[n].header
        extension_header += primary_header

        if var_to_wht:
            extension_data = 1./extension_data

    fitsio.writeto(output_path, extension_data, extension_header, output_verify='fix', overwrite=True)

def mk_links():

    # CFHT
    os.system("mkdir -p cfht/")
    force_symlink("%s/CFHT/HSC-UD-SXDS-U-ver1.nanfix.fits"%parent_dir,"cfht/premosaic_cfht_u.img.fits")
    force_symlink("%s/CFHT/HSC-UD-SXDS-U-ver1.weight.fits"%parent_dir,"cfht/premosaic_cfht_u.wht.fits")
    for i,_file in enumerate([x for x in os.listdir(parent_dir+'CFHT/Tiles/') if 'CFHTLS' not in x and '.weight' not in x and '.fits' in x]):
        force_symlink(    "%s/CFHT/Tiles/%s.fits"       %(parent_dir,_file[:-5]),"cfht/tile_cfht_u_%i.img.fits"%(i+1))
        extract_extension("%s/CFHT/Tiles/%s.weight.fits"%(parent_dir,_file[:-5]),'cfht/tile_cfht_u_%i.wht.fits'%(i+1),1)

    # CFHTLS
    os.system("mkdir -p cfhtls/")
    for filt in ['u','g','r','i','z']:
        for i,_file in enumerate(sorted([x for x in os.listdir(parent_dir+'CFHT/Tiles/') if 'CFHTLS_W_%s'%filt in x and 'weight' not in x])):
            force_symlink("%s/CFHT/Tiles/%s.fits"       %(parent_dir,_file[:-5]),"cfhtls/tile_cfhtls_%s_%s.img.fits"%(filt,i+1))
            force_symlink("%s/CFHT/Tiles/%s_weight.fits"%(parent_dir,_file[:-5]),"cfhtls/tile_cfhtls_%s_%s.wht.fits"%(filt,i+1))

    # SupCam
    os.system("mkdir -p supcam/")
    for filt in ['B','V','R','i','z']:
        for i in range(5):
            force_symlink("%s/SupCam/Tiles/sxds%s%ic_dr1.fits"%(parent_dir,filt,i+1),"supcam/tile_supcam_%s_%i.img.fits"%(filt.lower(),i+1))

    # UDS
    os.system("mkdir -p uds/")
    for filt in ['J','H','K']:
        force_symlink("%s/UDS/UDS-DR11-%s.mef.fits"       %(parent_dir,filt),"uds/premosaic_uds_%s.img.fits"%filt.lower())
        force_symlink("%s/UDS/UDS-DR11-%s.weight.mef.fits"%(parent_dir,filt),"uds/premosaic_uds_%s.wht.fits"%filt.lower())

    # VIDEO
    os.system("mkdir -p video/")
    for filt in ['Z','Y','J','H','Ks']:
        force_symlink("%s/VIDEO/xmm/images/xmm_%s_maxseeing0p90_2016-04-14.fits"     %(parent_dir,filt),"video/premosaic_video_%s.img.fits"%filt.lower())
        force_symlink("%s/VIDEO/xmm/images/xmm_%s_maxseeing0p90_2016-04-14_conf.fits"%(parent_dir,filt),"video/premosaic_video_%s.wht.fits"%filt.lower())

    # IRAC
    os.system("mkdir -p irac/")
    for i in range(4):
        force_symlink("%s/IRAC/SXDS.irac.%i.mosaic.fits"    %(parent_dir,i+1),"irac/premosaic_irac_%i.img.fits"%(i+1))
        force_symlink("%s/IRAC/SXDS.irac.%i.mosaic_std.fits"%(parent_dir,i+1),"irac/premosaic_irac_%i.std.fits"%(i+1))

def mk_hsc_links():

    hsc_list = sorted([x for x in os.listdir(parent_dir+'HSC/Tiles/') if '.fits' in x])
    filt_list = np.unique([x[11] for x in hsc_list])

    os.system("mkdir -p hsc/")
    for filt in filt_list:
        file_list = sorted([x for x in hsc_list if 'HSC-%s'%filt in x])
        for _file in file_list:
            input_file = parent_dir+'HSC/Tiles/'+_file
            print input_file+" -> "+"tile_hsc_%s_%s.*.fits"%(filt.lower(),_file[13:-5].replace(',','.'))
            extract_extension(input_file,'hsc/tile_hsc_%s_%s.img.fits' % (filt.lower(),_file[13:-5].replace(',','.')),1)
            extract_extension(input_file,'hsc/tile_hsc_%s_%s.msk.fits' % (filt.lower(),_file[13:-5].replace(',','.')),2)
            extract_extension(input_file,'hsc/tile_hsc_%s_%s.wht.fits' % (filt.lower(),_file[13:-5].replace(',','.')),3,var_to_wht=True)

if __name__ == '__main__':

    mk_links()
    # mk_hsc_links()

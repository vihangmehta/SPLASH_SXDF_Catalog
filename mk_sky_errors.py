import sys, os, time
import cv2
import numpy as np
import astropy.io.fits as fitsio
import scipy.spatial
import photutils
from astropy.wcs import WCS

import useful
from mk_cutouts import cutout_params

def mk_kernel(pad):

    N = int(np.ceil(pad) * 2) + 1
    ker = np.zeros((N,N))
    col,row = np.indices(ker.shape)
    col,row = col-N/2,row-N/2
    dist    = np.sqrt(col**2+row**2)
    ker[dist<=pad] = 1
    return ker

def setup_segmap(pad=17):

    for icut in range(4):

        print ("Padding SEGMAP Cutout#%i ... " % (icut+1), )

        img,hdr = fitsio.getdata('data/cutout%i_det.seg.fits'%(icut+1),header=True)
        ker = mk_kernel(pad=pad)

        img = np.array(img,dtype=np.int32)
        ker = np.array(ker,dtype=np.int32)

        start = time.time()
        conv_img = cv2.filter2D(img,-1,cv2.flip(ker,-1),anchor=(-1,-1))
        conv_img = conv_img.astype(np.int32)
        end = time.time()

        print ("convolution : %.2fs ... " % (end-start),)

        start = time.time()
        fitsio.writeto('data/cutout%i_det.seg_pad.fits'%(icut+1),data=conv_img,header=hdr,overwrite=True)
        end = time.time()

        print ("write out: %.2fs ... done!" % (end-start))

def setup_mskmap(pad=17):

    for instr in useful.instr_used_list[:-1]:

        for filt in useful.filters[instr]:

            print ("Padding %s-%s MASK ... " % (instr,filt), )

            data_dir = "data/irac/" if instr=="irac" else "data/orig/"

            img,hdr = fitsio.getdata(os.path.join(data_dir,'mosaic_%s_%s.msk.fits'%(instr,filt)),header=True)
            ker = mk_kernel(pad=pad)

            img = np.array(img,dtype=np.int32)
            ker = np.array(ker,dtype=np.int32)

            start = time.time()
            conv_img = cv2.filter2D(img,-1,cv2.flip(ker,-1),anchor=(-1,-1))
            conv_img = conv_img.astype(np.uint16)
            end = time.time()

            print ("convolution : %.2fs ... " % (end-start),)

            start = time.time()
            fitsio.writeto(os.path.join(data_dir,'mosaic_%s_%s.msk_pad.fits'%(instr,filt)),data=conv_img,header=hdr,overwrite=True)
            end = time.time()

            print ("write out: %.2fs ... done!" % (end-start))

def setup_starmask(pad=17):

    img,hdr = fitsio.getdata('data/orig/mosaic_hsc_y.omsk.fits',header=True)
    img = img.astype(int)

    cond = (img&512)==512
    img[ cond] = 1
    img[~cond] = 0

    img = img.astype(np.uint8)
    fitsio.writeto('data/orig/stars.msk.fits',data=img,header=hdr,overwrite=True)

    ker = mk_kernel(pad=pad)

    img = np.array(img,dtype=np.int32)
    ker = np.array(ker,dtype=np.int32)

    start = time.time()
    conv_img = cv2.filter2D(img,-1,cv2.flip(ker,-1),anchor=(-1,-1))
    conv_img = conv_img.astype(np.uint8)
    end = time.time()

    print ("convolution : %.2fs ... " % (end-start),)

    start = time.time()
    fitsio.writeto('data/orig/stars.msk_pad.fits',data=conv_img,header=hdr,overwrite=True)
    end = time.time()

    print ("write out: %.2fs ... done!" % (end-start))

def mk_sky_apers(instr,filt,N,save_dir='errors/'):

    iseed = 0
    seeds = (np.random.rand(10000)*1e8).astype(int)
    np.random.shuffle(seeds)

    cutout_verts_segmap = cutout_params()
    cutout_verts = [(    0, 25000,     0, 25000),
                    (    0, 25000, 25000, 50000),
                    (25000, 50000,     0, 25000),
                    (25000, 50000, 25000, 50000)]

    dtype = [("ID",int),("CUTNUM",int),("IMGX",int),("IMGY",int),("CUTX",int),("CUTY",int),('FLUX_APER',float,(5,)),('AVG_WHT',float,(5,))]

    if N%4!=0: N = 4 * ((N/4)+1)
    split_N = [0,N/4,N/2,3*N/4,N]
    sky_apers = np.recarray(N,dtype=dtype)
    for x in sky_apers.dtype.names: sky_apers[x] = -99.

    sky_apers['ID'] = np.arange(N) + 1
    for icut,(n0,n1) in enumerate(zip(split_N[:-1],split_N[1:])): sky_apers['CUTNUM'][n0:n1] = icut+1

    mskmap = fitsio.getdata('data/orig/mosaic_%s_%s.msk_pad.fits'%(instr,filt))
    strmap = fitsio.getdata('data/orig/stars.msk_pad.fits')

    for icut,(n0,n1) in enumerate(zip(split_N[:-1],split_N[1:])):

        segmap = fitsio.getdata('data/cutout%i_det.seg_pad.fits'%(icut+1))
        cutx = cuty = np.zeros(0,dtype=int)
        x0,x1,y0,y1 = cutout_verts[icut]
        _x0,_x1,_y0,_y1 = cutout_verts_segmap[icut]

        while len(cutx) < len(sky_apers[n0:n1]):

            randn = int(max(1e4,np.random.normal(6e4,2e4)))

            iseed += 1
            np.random.seed(seeds[iseed])
            x = np.random.randint(0,25000,size=randn)

            iseed += 1
            np.random.seed(seeds[iseed])
            y = np.random.randint(0,25000,size=randn)

            cond = (segmap[y+(y0-_y0),x+(x0-_x0)] == 0) & (mskmap[y+y0,x+x0] == 0) & (strmap[y+y0,x+x0] == 0)
            cutx = np.append(cutx,x[cond])
            cuty = np.append(cuty,y[cond])

            # for i in range(len(cutx)):
            #     if cutx[i]!=-99. and cuty[i]!=-99.:
            #         dist = scipy.spatial.distance.cdist(zip(cutx,cuty),[(cutx[i],cuty[i]),]).flatten()
            #         _cond = (dist!=0) & (dist<=2.5)
            #         cutx[_cond],cuty[_cond] = -99,-99

            # cond = (cutx!=-99) & (cuty!=-99)
            # print (len(cutx),len(cutx[~cond]))
            # cutx,cuty = cutx[cond],cuty[cond]

            sys.stdout.write("\rMaking Sky Apertures for %s %s - cutout#%i - %5i/%5i ... "%(instr,filt,icut+1,icut*len(sky_apers[n0:n1])+len(cutx),len(sky_apers)))
            sys.stdout.flush()

        sky_apers['CUTX'][n0:n1] = cutx[:N/4]
        sky_apers['CUTY'][n0:n1] = cuty[:N/4]
        sky_apers['IMGX'][n0:n1] = cutx[:N/4] + x0
        sky_apers['IMGY'][n0:n1] = cuty[:N/4] + y0

    sys.stdout.write("done!\n")
    sys.stdout.flush()

    np.savetxt(os.path.join(save_dir,'sky_apers_%s_%s.reg'%(instr,filt)),
                np.vstack((sky_apers['IMGX'],sky_apers['IMGY'])).T,
                fmt='circle(%.2f,%.2f,10) # width=2 color=green')

    fitsio.writeto(os.path.join(save_dir,'sky_apers_%s_%s.fits'%(instr,filt)),sky_apers,overwrite=True)

def mk_sky_phot(instr,filt,save_dir='errors/'):

    apersizes = useful.apersizes / useful.pix_scale
    aperradii = apersizes / 2.

    img = fitsio.getdata('data/orig/mosaic_%s_%s.img.fits'%(instr,filt))
    wht = fitsio.getdata('data/orig/mosaic_%s_%s.wht.fits'%(instr,filt))

    sky_apers = fitsio.getdata(os.path.join(save_dir,'sky_apers_%s_%s.fits'%(instr,filt)))
    pos = zip(sky_apers['IMGX'],sky_apers['IMGY'])

    for i,radius in enumerate(aperradii):

        sys.stdout.write("\rMeasuring Photometry on sky apertures for %s %s - aperture#%i (%.2f px) ..." % (instr,filt,i+1,radius))
        sys.stdout.flush()

        apertures = photutils.CircularAperture(pos,r=radius)
        sky_apers['FLUX_APER'][:,i] = photutils.aperture_photometry(img, apertures,method='subpixel',subpixels=5)['aperture_sum']
        sky_apers['AVG_WHT'][:,i]   = photutils.aperture_photometry(wht, apertures)['aperture_sum'] / (np.pi*radius**2)

    sys.stdout.write("done!\n")
    sys.stdout.flush()

    fitsio.writeto(os.path.join(save_dir,'sky_apers_%s_%s.fits'%(instr,filt)),sky_apers,overwrite=True)

def get_sky_phot(instr,filt,save_dir='errors/'):

    sky_apers = fitsio.getdata(os.path.join(save_dir,'sky_apers_%s_%s.fits'%(instr,filt)))
    return sky_apers

def mk_irac_sky_errors():

    instr = 'irac'

    apersizes = useful.apersizes / 0.6 # need to use the IRAC pixscale
    aperradii = apersizes / 2.

    sky_apers = fitsio.getdata("errors/sky_apers_cfhtls_u.fits")
    hdr = fitsio.getheader("data/orig/mosaic_cfhtls_u.msk.fits")
    wcs = WCS(hdr)
    sky_ra,sky_dec = wcs.all_pix2world(sky_apers["IMGX"],sky_apers["IMGY"],1)

    for filt in useful.filters[instr]:

        out_id,out_cut,out_tile,out_x,out_y = np.zeros((5,0))
        out_rms = np.zeros(0)

        for cut_id in (np.arange(4)+1):

            for tile_id in (np.arange(49)+1):

                sys.stdout.write("\rProcessing %s:%s-%s ... \033[K" % (filt,cut_id,tile_id))
                sys.stdout.flush()

                rms_name = "irac_phot/out_fits/rms_ch%s_%i_%i.1.fits"%(filt,cut_id,tile_id)
                if os.path.isfile(rms_name):

                    img,hdr = fitsio.getdata(rms_name,header=True)
                    wcs = WCS(hdr)
                    sky_x,sky_y = wcs.all_world2pix(sky_ra,sky_dec,1)

                    cond = (80 < sky_x) & (sky_x < img.shape[1]-80) & \
                           (80 < sky_y) & (sky_y < img.shape[0]-80)

                    pos = zip(sky_x[cond],sky_y[cond])

                    apertures = photutils.CircularAperture(pos,r=aperradii[1])
                    rms       = photutils.aperture_photometry(img, apertures, method='subpixel', subpixels=5)['aperture_sum'] / (np.pi*aperradii[1]**2)
                    # The rms is measured as the average of the pixels within the aperture because
                    # the rms maps are already the computed for a 2" aperture and do not need any additional treatment

                    cond_rms = np.isfinite(rms)

                    out_id   = np.concatenate((out_id  ,sky_apers["ID"][cond][cond_rms]))
                    out_cut  = np.concatenate((out_cut , [cut_id]*np.sum(cond_rms)))
                    out_tile = np.concatenate((out_tile,[tile_id]*np.sum(cond_rms)))
                    out_x    = np.concatenate((out_x   ,sky_apers["IMGX"][cond][cond_rms] / 4.))
                    out_y    = np.concatenate((out_y   ,sky_apers["IMGY"][cond][cond_rms] / 4.))
                    out_rms  = np.concatenate((out_rms ,rms[cond_rms]))

        print()

        sky_irac = np.recarray(len(out_id),
                            dtype=[("ID",int),("cut_id",int),("tile_id",int),("IMGX",float),("IMGY",float),("rms",float)])
        sky_irac["ID"]      = out_id
        sky_irac["cut_id"]  = out_cut
        sky_irac["tile_id"] = out_tile
        sky_irac["IMGX"]    = out_x
        sky_irac["IMGY"]    = out_y
        sky_irac["rms"]     = out_rms

        fitsio.writeto("errors/sky_apers_%s_%s.fits"%(instr,filt),sky_irac,overwrite=True)

if __name__ == '__main__':

    # setup_segmap()
    # setup_mskmap()
    # setup_starmask()

    # for instr in useful.instr_used_list[:-1]:
    #     for filt in useful.filters[instr]:
    #         mk_sky_apers(instr=instr,filt=filt,N=250000)
    #         mk_sky_phot(instr=instr,filt=filt)

    # mk_irac_sky_errors()

    print()

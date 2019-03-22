import os, time, cv2, subprocess
import numpy as np
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import useful

cfhtls_filters = ['u','g','r','i','z']

size_cuts = {'u' : {1:[2.20,3.30],2:[2.70,3.80],3:[2.50,3.40],4:[2.00,2.90],5:[2.50,3.50],6:[2.60,3.50],7:[2.50,3.40],8:[2.20,3.10],9:[2.50,3.20]},
             'g' : {1:[2.20,3.10],2:[2.30,3.10],3:[2.10,3.00],4:[2.30,3.10],5:[2.20,3.20],6:[1.80,2.30],7:[2.30,3.20],8:[2.10,3.10],9:[2.40,3.20]},
             'r' : {1:[1.60,2.40],2:[1.90,2.80],3:[2.00,2.80],4:[2.40,3.20],5:[2.30,3.10],6:[1.90,2.60],7:[2.30,3.10],8:[2.40,3.10],9:[1.90,2.70]},
             'i' : {1:[1.50,2.00],2:[1.40,2.40],3:[2.10,2.80],4:[1.50,2.30],5:[2.00,2.70],6:[1.40,2.10],7:[1.60,2.30],8:[2.00,2.90],9:[1.90,2.80]},
             'z' : {1:[2.00,2.70],2:[1.70,2.40],3:[2.10,2.90],4:[2.20,2.70],5:[1.80,2.60],6:[2.40,3.10],7:[1.70,2.40],8:[2.00,2.70],9:[1.70,2.50]}}

mag_cuts  = {'u' : {1:[17.30,20.00],2:[16.80,20.70],3:[17.00,20.70],4:[17.40,20.80],5:[17.20,20.60],6:[17.00,20.60],7:[17.00,20.50],8:[17.00,20.80],9:[17.00,21.00]},
             'g' : {1:[18.50,21.20],2:[18.40,21.40],3:[18.20,21.00],4:[18.20,21.00],5:[18.20,21.20],6:[18.70,21.60],7:[18.00,21.00],8:[18.30,21.20],9:[18.10,21.30]},
             'r' : {1:[18.40,20.90],2:[17.80,20.60],3:[17.80,20.50],4:[17.70,20.60],5:[17.50,20.70],6:[17.80,20.80],7:[17.50,20.70],8:[17.40,20.60],9:[17.80,20.80]},
             'i' : {1:[18.40,20.80],2:[18.30,20.50],3:[17.80,20.30],4:[18.30,20.60],5:[18.00,20.30],6:[18.70,20.80],7:[18.30,20.80],8:[17.80,20.30],9:[18.00,20.00]},
             'z' : {1:[17.00,19.80],2:[17.40,19.70],3:[16.90,19.70],4:[16.90,19.60],5:[17.00,19.80],6:[16.60,19.50],7:[17.00,19.80],8:[17.00,20.00],9:[17.20,19.80]}}

def mk_ldacs(data_dir):

    def call(filt,tile):

        img_name = os.path.join(data_dir,"tile_cfhtls_%s_%i.img.fits" % (filt,tile))
        wht_name = os.path.join(data_dir,"tile_cfhtls_%s_%i.wht.fits" % (filt,tile))
        cat_name = os.path.join(data_dir,"tile_cfhtls_%s_%i.ldac"     % (filt,tile))
        zp = 30.

        call = "sextractor %s -c config/config_psfex.sex " \
               "-PARAMETERS_NAME config/param_psfex.sex " \
               "-CATALOG_NAME %s -CATALOG_TYPE FITS_LDAC " \
               "-WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE %s " \
               "-MAG_ZEROPOINT %.2f" % (img_name,cat_name,wht_name,zp)

        if os.path.isfile(img_name) and os.path.isfile(wht_name):
            return call
        else:
            print "[mk_psfs.py] Warning! Input files not found for %s:%s" % (instr,filt)

    calls = []
    for filt in cfhtls_filters:
        for i in np.arange(9)+1:
            calls.append(call(filt=filt,tile=i))

    call_chunks = np.array_split(calls,len(calls)/10)
    for call_chunk in call_chunks:
        useful.multiprocess(call_chunk,cwd=cwd)

def mk_star_cat(data_dir,manual):

    for filt in cfhtls_filters:
            
        for i in np.arange(9)+1:

            cat_name = os.path.join(data_dir,"tile_cfhtls_%s_%i.ldac" % (filt,i))    
            cat_hdu = fitsio.open(cat_name)
            catalog = cat_hdu[2].data
            
            fig,ax = plt.subplots(1,1,figsize=(10,8),dpi=75,tight_layout=True)
            ax.set_title("Selecting stars for %s/%i" % (filt,i),fontsize=16)
            
            patch = ax.add_patch(Polygon([[0,0],],color='r',lw=2,alpha=0.5,closed=True))
            ax.scatter(catalog['FLUX_RADIUS'],catalog['MAG_AUTO'],c='k',s=5,lw=0,alpha=0.5)
            ax.set_xlabel('Half-light radius (size) [px]')
            ax.set_ylabel('AUTO Magnitude [AB]')
            
            ax.set_ylim(28,12)
            ax.set_xlim(1,15)
            plt.draw()
            
            if manual:
                plt.show(block=False)
                proceed = 'n'
            else:
                proceed = 'y'
                size_cut = size_cuts[filt][i]
                mag_cut  =  mag_cuts[filt][i]
                verts = [[size_cut[0],mag_cut[0]],[size_cut[0],mag_cut[1]],[size_cut[1],mag_cut[1]],[size_cut[1],mag_cut[0]],[size_cut[0],mag_cut[0]]]
                patch = ax.add_patch(Polygon(verts,color='r',lw=2,alpha=0.5,closed=True))

            while proceed != 'y':
                
                print "Selecting stars for %s/%i" % (filt,i)
                size_cut = [float(raw_input("Enter lower limit in size: ")),
                            float(raw_input("Enter upper limit in size: "))]
                mag_cut  = [float(raw_input("Enter lower limit in mag : ")),
                            float(raw_input("Enter upper limit in mag : "))]

                patch.remove()
                verts = [[size_cut[0],mag_cut[0]],[size_cut[0],mag_cut[1]],[size_cut[1],mag_cut[1]],[size_cut[1],mag_cut[0]],[size_cut[0],mag_cut[0]]]
                patch = ax.add_patch(Polygon(verts,color='r',lw=2,alpha=0.5,closed=True))
                plt.draw()
                proceed = raw_input("Continue? [y/n] ")
            
            fig.savefig('%s/psfs/plots/tile_cfhtls_%s_%i.stars.png' % (data_dir,filt,i))

            cond = (size_cut[0]<catalog['FLUX_RADIUS']) & (catalog['FLUX_RADIUS']<size_cut[1]) & \
                   ( mag_cut[0]<catalog['MAG_AUTO'])    & (catalog['MAG_AUTO']   < mag_cut[1])

            print "(%s/%i) Selecting %i stars out of %i sources "\
                  "using %.2f<size<%.2f and %.2f<mag<%.2f" % (filt,i,
                            len(cat_hdu[2].data[cond]),len(cat_hdu[2].data),size_cut[0],size_cut[1],mag_cut[0],mag_cut[1])

            cond2 = np.ones(len(catalog[cond]),dtype=bool)

            cat_hdu[2].data = cat_hdu[2].data[cond][cond2]
            cat_hdu.writeto("%s/tile_cfhtls_%s_%i.stars.ldac" % (data_dir,filt,i), overwrite=True)

            plt.close(fig)

def run(call,cwd):
    
    print "Processing %s-%i ... " % (call["filt"],call["tile"])
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
    print "Finished %s-%i in %.2fs." % (call["filt"],call["tile"],time)

def mk_psf(data_dir,psf_dir,kernel_dir,basis_type):

    chkimg_list = ['chi.fits','proto.fits','samp.fits','resi.fits','snap.fits']
    chkplt_list = ['selfwhm','fwhm','ellipticity','counts','countfrac','chi2','resi']

    homo_config = "-HOMOBASIS_TYPE GAUSS-LAGUERRE "\
                  "-HOMOPSF_PARAMS 3.76,2.8 " \
                  "-HOMOKERNEL_DIR %s " % kernel_dir

    calls = []
    for filt in cfhtls_filters:
        for i in (np.arange(9)+1):
        
            cat_name = os.path.join(data_dir,"tile_cfhtls_%s_%i.stars.ldac"%(filt,i))
            xml_name = os.path.join(psf_dir,'psfex_cfhtls_%s_%i.xml'%(filt,i))
            log_name = os.path.join(psf_dir,'psfex_cfhtls_%s_%i.log'%(filt,i))
        
            call = "psfex %s -c config/config.psfex " \
                   "-BASIS_TYPE %s "\
                   "-PSF_SIZE 101,101 -PSF_DIR %s %s " \
                   "-WRITE_XML Y -XML_NAME %s " \
                   "-CHECKIMAGE_NAME %s " \
                   "-CHECKPLOT_NAME %s" % (cat_name, basis_type, psf_dir, homo_config, xml_name,
                                           ','.join([os.path.join(psf_dir,_) for _ in chkimg_list]),
                                           ','.join([os.path.join(psf_dir,'plots/',_) for _ in chkplt_list]))

            calls.append({"call":call,"log":log_name,"filt":filt,"tile":i})

    async_run = useful.AsyncFactory(run, cb_func)
    for call in calls: async_run.call(call=call,cwd=cwd)     
    async_run.wait()

def mk_convolve(data_dir,kernel_dir):

    for filt in cfhtls_filters:

        for i in (np.arange(9)+1):

            print "Convolving %s:%i " % (filt,i)

            img, hdr = fitsio.getdata(os.path.join(data_dir,'tile_cfhtls_%s_%i.img.fits'%(filt,i)),header=True)
            ker = fitsio.getdata(os.path.join(kernel_dir,'tile_cfhtls_%s_%i.stars.homo.fits'%(filt,i)))
            
            img = np.array(img,dtype=np.float32)
            ker = np.array(ker,dtype=np.float32)

            start = time.time()
            ker = cv2.flip(ker,-1)
            conv_img = cv2.filter2D(img,-1,ker)
            end = time.time()

            print "Convolution: %.2fs" % (end-start)

            start = time.time()
            hdr.set('OBJECT','%s (convolved)' % hdr['OBJECT'])
            fitsio.writeto(os.path.join(data_dir,'tile_conv_cfhtls_%s_%i.img.fits' % (filt,i)), conv_img, hdr, output_verify="ignore", overwrite=True)
            end = time.time()
            
            print "Write out: %.2fs" % (end-start)

            src = os.path.join(data_dir,'tile_cfhtls_%s_%i.wht.fits'%(filt,i))
            dst = os.path.join(data_dir,'tile_conv_cfhtls_%s_%i.wht.fits'%(filt,i))
            print "Linking %s to %s" % (src,dst)
            useful.force_symlink(src,dst)

if __name__ == '__main__':
    
    cwd = "/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/scamp/"

    # mk_ldacs(data_dir=os.path.join(cwd,'cfhtls'))

    # mk_star_cat(data_dir=os.path.join(cwd,'cfhtls'),manual=False)

    mk_psf(  data_dir=os.path.join(cwd,'cfhtls'),
              psf_dir=os.path.join(cwd,"cfhtls/psfs"),
           kernel_dir=os.path.join(cwd,"cfhtls/psfs/kernels"),
           basis_type='PIXEL')
    
    mk_convolve(  data_dir=os.path.join(cwd,'cfhtls'),
                kernel_dir=os.path.join(cwd,"cfhtls/psfs/kernels"))

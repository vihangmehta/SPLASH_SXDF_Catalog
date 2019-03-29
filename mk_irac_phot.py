import sys
import os
import time
import subprocess
from multiprocessing import Queue, Process
import numpy as np

import astropy.io.fits as fitsio
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import useful
from rebin import rebin,upbin
from mk_cutouts import cutout_params

cutout_num     = np.arange(7)+1 # 11
cutout_size    = 4200. # 5000
cutout_overlap = 650.  # 500

cutout_dim060 = [4200/4,4200/4]
cutout_dim015 = [4200  ,4200  ]

def centers():
    return ((cutout_num-1./2.)*cutout_size) - ((cutout_num-1)*cutout_overlap)

def plot_cutouts():

    cutout_verts = cutout_params()
    fig,ax = plt.subplots(1,1,figsize=(10,9),dpi=75,tight_layout=True)

    with open("irac_phot/irac_cutouts.txt","w") as f:
        f.write("#%9s%15s%15s\n"%("cutout","RA","DEC"))

    with open("irac_phot/irac_cutouts.txt","a") as f:

        for i,color in zip(np.arange(4),['dodgerblue','orange','purple','forestgreen']):

            x0,x1,y0,y1 = cutout_verts[i]

            det_hdr = fitsio.getheader("/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/data/cutout%i_det.img.fits"%(i+1))

            c = centers()
            ycs,xcs = np.meshgrid(c,c)

            wcs = WCS(det_hdr)
            ras,decs = wcs.all_pix2world(xcs.flatten(),ycs.flatten(),1)
            np.savetxt(f,np.vstack((np.zeros(len(ras))+i+1,np.arange(len(ras))+1,ras,decs)).T,fmt="%6i%4i%15.8f%15.8f")

            xcs += x0
            ycs += y0

            for j,(xc,yc) in enumerate(zip(xcs.flatten(),ycs.flatten())):
                ax.add_patch(Rectangle(xy=(xc-cutout_size/2.,yc-cutout_size/2.),width=cutout_size,height=cutout_size,facecolor=color,edgecolor=color,lw=1.5,alpha=0.2))
                ax.text(xc,yc,"%i-%i"%(i+1,j+1),color='k',fontsize=14,fontweight=600,ha='center',va='center')

    ax.set_aspect(1.)
    ax.set_xlim(-500,50500)
    ax.set_ylim(-500,50500)
    fig.savefig("irac_phot/irac_cutouts.png")

def extract_cutout(img,hdr,ra,dec,size):

    wcs = WCS(hdr)
    pos = SkyCoord(ra,dec,frame="fk5",unit="deg")
    ext = Cutout2D(data=img,position=pos,size=size,wcs=wcs,mode='partial',fill_value=0)
    img = ext.data
    wcs = ext.wcs.to_header()
    new_hdr = hdr.copy()
    for x in wcs: new_hdr[x] = wcs[x]
    return img, new_hdr

def rebin_PRF(PRFfile,indir,outdir):

    infile  = os.path.join(indir,PRFfile)
    outfile = os.path.join(outdir,PRFfile.replace(".fits",".rb_060.fits"))
    psf,hdr = fitsio.getdata(infile,header=True)
    # Going from 1.22"/px @ 100x oversample to 0.6"/px ==> 1/(1.22/0.6/10) = 5
    psf = rebin(psf,5,mode='sum')
    fitsio.writeto(outfile,data=psf,header=hdr,output_verify='silentfix+ignore',overwrite=False)
    return PRFfile

def rebin_PRF_cb(x):
    print (x)

def rebin_PRFs(indir,outdir):

    PRFlist = np.sort(os.listdir(indir))
    async_run = useful.AsyncFactory(rebin_PRF, rebin_PRF_cb, nproc=15)
    for PRFfile in PRFlist: async_run.call(PRFfile=PRFfile,indir=indir,outdir=outdir)
    async_run.wait()

def mk_cutouts(i,j,ra,dec,det,det_hdr,seg,seg_hdr,img,img_hdr,rms,rms_hdr,PRF_summary):

    _det,_hdr = extract_cutout(det,det_hdr,ra,dec,cutout_dim015)
    _det = rebin(_det,4,mode='sum')
    _hdr['CRPIX1'] = (_hdr['CRPIX1'] - 0.5)/4. + 0.5
    _hdr['CRPIX2'] = (_hdr['CRPIX2'] - 0.5)/4. + 0.5
    _hdr['CD1_1'] = -0.6/3600.
    _hdr['CD2_2'] = +0.6/3600.
    fitsio.writeto(os.path.join(cwd,"inp_fits/det_%i_%i.img.fits"%(j,i)),data=_det,header=_hdr,overwrite=True)

    _seg,_hdr = extract_cutout(seg,seg_hdr,ra,dec,cutout_dim015)
    _seg = _seg.astype(">i4")
    _seg = rebin(_seg,4,mode='mode-fix')
    _seg = _seg.astype(">i4")
    _hdr['CRPIX1'] = (_hdr['CRPIX1'] - 0.5)/4. + 0.5
    _hdr['CRPIX2'] = (_hdr['CRPIX2'] - 0.5)/4. + 0.5
    _hdr['CD1_1'] = -0.6/3600.
    _hdr['CD2_2'] = +0.6/3600.
    fitsio.writeto(os.path.join(cwd,"inp_fits/det_%i_%i.seg.fits"%(j,i)),data=_seg,header=_hdr,overwrite=True)

    for filt in useful.filters["irac"]:

        # IRAC SCI image
        _img,_hdr = extract_cutout(img[filt],img_hdr[filt],ra,dec,cutout_dim060)
        fitsio.writeto(os.path.join(cwd,"inp_fits/irac_ch%s_%i_%i.img.fits"%(filt,j,i)),data=_img,header=_hdr,overwrite=True)

        # IRAC PRF summary file
        wcs = WCS(_hdr)
        PRFx,PRFy = wcs.all_world2pix(PRF_summary[filt]["ra"],PRF_summary[filt]["dec"],1)
        dx = np.max(np.diff(np.sort(np.unique(PRFx))))
        dy = np.max(np.diff(np.sort(np.unique(PRFx))))
        cond = (-dx/2.<PRFx) & (PRFx<_img.shape[0]+dx/2.) & (-dy/2.<PRFy) & (PRFy<_img.shape[1]+dy/2.)

        name = np.array(["PRFs.rb_060/"+x.replace(".fits",".rb_060.fits") for x in PRF_summary[filt]["name"][cond]],dtype="|S80")
        xc   = PRFx[cond]
        yc   = PRFy[cond]

        _PRF_summary = np.array(zip(name,xc,yc),dtype=[("name","|S80"),("x",float),("y",float)])
        np.savetxt(os.path.join(cwd,"inp_fits/irac_ch%s_%i_%i.PRF.txt"%(filt,j,i)),_PRF_summary,fmt="%s%10.4f%10.4f",header="%s%10s%10s"%("Name","X","Y"))

        # IRAC RMS map
        _rms,_hdr = extract_cutout(rms[filt],rms_hdr[filt],ra,dec,cutout_dim060)
        fitsio.writeto(os.path.join(cwd,"inp_fits/irac_ch%s_%i_%i.rms.fits"%(filt,j,i)),data=_rms,header=_hdr,overwrite=True)

def setup_iraclean():

    def worker(queue,iter_chunk):
        for iters in iter_chunk:
            print ("Working on %i-%i ... " % (iters['j'],iters['i']))
            mk_cutouts(i=iters['i'],j=iters['j'],ra=iters['ra'],dec=iters['dec'],
                       det=det,det_hdr=det_hdr,seg=seg,seg_hdr=seg_hdr,
                       img=img,img_hdr=img_hdr,rms=rms,rms_hdr=rms_hdr,
                       PRF_summary=PRF_summary)
        queue.put(None)

    c = centers()
    ycs,xcs  = np.meshgrid(c,c)

    img,img_hdr,rms,rms_hdr,PRF_summary = {},{},{},{},{}
    for filt in useful.filters["irac"]:
        img[filt],img_hdr[filt] = fitsio.getdata("/data/highzgal/PUBLICACCESS/SPLASH/DATA/IRAC/SXDS.irac.%s.mosaic.fits"%filt,header=True)
        rms[filt],rms_hdr[filt] = fitsio.getdata("/data/highzgal/PUBLICACCESS/SPLASH/DATA/IRAC/SXDS.irac.%s.mosaic_unc.fits"%filt,header=True)
        PRF_summary[filt] = np.genfromtxt(os.path.join(cwd,"PRF_summary/SXDS.irac.%s.mosaic_summary.txt"%filt),
                                                                dtype=[("id",int),("name",np.object),("nPRF",int),("ra",float),("dec",float)])

    for j in range(4):

        det,det_hdr = fitsio.getdata("/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/data/cutout%i_det.img.fits"%(j+1),header=True)
        seg,seg_hdr = fitsio.getdata("/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/data/cutout%i_det.seg.fits"%(j+1),header=True)
        wcs = WCS(det_hdr)
        ras,decs = wcs.all_pix2world(xcs.flatten(),ycs.flatten(),1)

        iters = [{'i':i+1,'j':j+1,'ra':ra,'dec':dec} for i,(ra,dec) in enumerate(zip(ras,decs))]

        # MULTIPROCESS BITS
        iter_chunks = np.array_split(iters,15)

        queue = Queue()
        procs = [Process(target=worker, kwargs={'queue':queue,'iter_chunk':iter_chunk}) for iter_chunk in iter_chunks]
        for proc in procs: proc.start()

        finished = 0
        while finished < len(procs):
            items = queue.get()
            if items is None: finished+=1

        for proc in procs: proc.join()

def multiprocess(calls,cwd):

    def worker(queue,call):
        for _ in call["del_files"]: useful.delete_file(os.path.join(cwd,_))
        run(call["call"],cwd=cwd,log=call["log"])
        queue.put(None)

    queue = Queue()
    procs = [Process(target=worker, args=(queue,call)) for call in calls]
    for proc in procs: proc.start()

    finished = 0
    while finished < len(procs):
        items = queue.get()
        if items is None: finished+=1

    for proc in procs: proc.join()

def run(call,cwd):

    print ("ch%s %i-%i ... " % (call["filt"],call["cutout"],call["tile"]))

    for _ in call["del_files"]: useful.delete_file(os.path.join(cwd,_))

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
    print ("ch%s %i-%i done in %.2f s " % (call["filt"],call["cutout"],call["tile"],time))

def run_iraclean(filt):

    c = centers()
    ycs,xcs  = np.meshgrid(c,c)

    calls = []
    for j in range(4):

        for i in range(len(xcs.flatten())):

            det = "inp_fits/det_%i_%i.img.fits"%(j+1,i+1)
            seg = "inp_fits/det_%i_%i.seg.fits"%(j+1,i+1)
            img = "inp_fits/irac_ch%s_%i_%i.img.fits"%(filt,j+1,i+1)
            unc = "inp_fits/irac_ch%s_%i_%i.rms.fits"%(filt,j+1,i+1)
            psf = "inp_fits/irac_ch%s_%i_%i.PRF.txt" %(filt,j+1,i+1)

            res = "out_fits/res_ch%s_%i_%i.fits"%(filt,j+1,i+1)
            dec = "out_fits/dec_ch%s_%i_%i.fits"%(filt,j+1,i+1)
            rms = "out_fits/rms_ch%s_%i_%i.fits"%(filt,j+1,i+1)

            pht = "out_cats/phot_ch%s_%i_%i.cat"   %(filt,j+1,i+1)
            err = "out_cats/photerr_ch%s_%i_%i.cat"%(filt,j+1,i+1)

            log = "logs/log_ch%s_%i_%i.txt"%(filt,j+1,i+1)

            # if filt in ['1','2']:
            #     call  = "bin/iraclean %s %s %s %s %s %s %s 0.1 0.01 0.1 0.5 10 1 0.3; " % (img,psf,unc,det,seg,res,dec)
            # elif filt in ['3','4']:
            #     call  = "bin/iraclean %s %s %s %s %s %s %s 0.1 0.10 0.1 5.0 10 1 0.3; " % (img,psf,unc,det,seg,res,dec)
            # else:
            #     raise Exception("Invalid filt.")

            call  = "bin/iraclean %s %s %s %s %s %s %s 0.1 0.01 0.1 0.5 10 1 0.3; " % (img,psf,unc,det,seg,res,dec)
            call += "bin/mkphot %s %s %s; " % (seg,dec.replace(".fits",".1.fits"),pht)
            call += "bin/mkerror %s %s %s 1 2.307 %s %s" % (seg,dec.replace(".fits",".1.fits"),res.replace(".fits",".1.fits"),rms.replace(".fits",".1.fits"),err)

            calls.append({"call":call,
                          "del_files":[res.replace(".fits",".1.fits"),dec.replace(".fits",".1.fits"),rms.replace(".fits",".1.fits"),pht,err,log],
                          "log":log,
                          "cutout":j+1,
                          "tile":i+1,
                          "filt":filt})

    async_run = useful.AsyncFactory(run, cb_func)
    for call in calls: async_run.call(call=call,cwd=cwd)
    async_run.wait()

def calc_fscale(zp0,zp1=23.93):
    """
    -2.5*log(f1) + zp1 = -2.5*log(f0) + zp0
    f1/f0 = 10**((zp1 - zp0) / 2.5)
    """
    fscale = 10**((zp1 - zp0) / 2.5)
    return fscale

def convert_fluxes(flux,ferr,filt):

    new_flux = np.zeros(flux.shape) - 99.
    new_ferr = np.zeros(ferr.shape) - 99.

    cond_ferr = np.isfinite(ferr)

    new_flux[cond_ferr] = flux[cond_ferr] * calc_fscale(zp0=useful.orig_zp["irac"][filt])
    new_ferr[cond_ferr] = ferr[cond_ferr] * calc_fscale(zp0=useful.orig_zp["irac"][filt])

    return new_flux, new_ferr

def fix_irac_coverage_flag(catalog,cutout_num,filt,instr='irac'):
    """
    Flag sources with no irace coverage

    First start by flagging everything to have coverage
    Then go cutout-by-cutout and filter-by-filter to check what objects
    lie outside of the coverage area (by checking the mask image)
    For all objects that do not have coverage, turn off the coverage flag (set to 0)
    """
    seg = fitsio.getdata(os.path.join(cwd,os.pardir,"data","cutout%i_det.seg.fits"%cutout_num))
    msk = fitsio.getdata(os.path.join(cwd,os.pardir,"data","orig","cutout%i_%s_%s.msk.fits"%(cutout_num,instr,filt)))

    _seg = seg[msk.astype(bool)]
    _idx = np.unique(_seg)

    cond = np.in1d(catalog['NUMBER'],_idx)
    catalog["COVERAGE_FLAG"][cond] = 0 # No coverage

    return catalog

def mk_cats():

    c = centers()
    ycs,xcs = np.meshgrid(c,c)
    rows,cols = np.indices(xcs.shape)

    for filt in useful.filters["irac"]:

        for j in range(4):

            sex_cat  = fitsio.getdata(os.path.join(cwd,os.pardir,'data','catalog%i_det.fits'%(j+1)))
            irac_cat = np.recarray(len(sex_cat),
                                 dtype=[("NUMBER",int),("X_IMAGE",float),("Y_IMAGE",float),
                                        ("X_WORLD",float),("Y_WORLD",float),
                                        ("FLUX",float),("FLUXERR",float),("NPIX",int),
                                        ("MAG",float),("MAGERR",float),("FLAGS",int),("COVERAGE_FLAG",int)])
            for x in irac_cat.dtype.names: irac_cat[x] = -99.

            irac_cat["NUMBER"]  = sex_cat["NUMBER"]
            irac_cat["X_IMAGE"] = sex_cat["X_IMAGE"]
            irac_cat["Y_IMAGE"] = sex_cat["Y_IMAGE"]
            irac_cat["X_WORLD"] = sex_cat["X_WORLD"]
            irac_cat["Y_WORLD"] = sex_cat["Y_WORLD"]
            irac_cat["FLAGS"]   = -99.
            irac_cat["COVERAGE_FLAG"] = -99.

            for i,(_row,_col,xc,yc) in enumerate(zip(rows.flatten(),cols.flatten(),xcs.flatten(),ycs.flatten())):

                xlim = [xc - cutout_size/2. + cutout_overlap/2., xc + cutout_size/2. - cutout_overlap/2.]
                ylim = [yc - cutout_size/2. + cutout_overlap/2., yc + cutout_size/2. - cutout_overlap/2.]

                if _row==np.min(rows): xlim[0] = xc - cutout_size/2.
                if _row==np.max(rows): xlim[1] = xc + cutout_size/2.
                if _col==np.min(cols): ylim[0] = yc - cutout_size/2.
                if _col==np.max(cols): ylim[1] = yc + cutout_size/2.

                cond = (xlim[0] <= irac_cat["X_IMAGE"]) & (irac_cat["X_IMAGE"] <= xlim[1]) & \
                       (ylim[0] <= irac_cat["Y_IMAGE"]) & (irac_cat["Y_IMAGE"] <= ylim[1])

                try:

                    pht = np.genfromtxt(os.path.join(cwd,"out_cats/phot_ch%s_%i_%i.cat"   %(filt,j+1,i+1)),dtype=[('ID',int),('flux',float)])
                    err = np.genfromtxt(os.path.join(cwd,"out_cats/photerr_ch%s_%i_%i.cat"%(filt,j+1,i+1)),dtype=[('ID',int),('ferr',float),('npix',int)])
                    rms,hdr = fitsio.getdata(os.path.join(cwd,"inp_fits/irac_ch%s_%i_%i.rms.fits"%(filt,j+1,i+1)),header=True)
                    wcs = WCS(hdr)

                    pht["flux"],err["ferr"] = convert_fluxes(pht["flux"],err["ferr"],filt)

                    pht = dict(zip(pht['ID'],pht['flux']))
                    npx = dict(zip(err['ID'],err['npix']))
                    err = dict(zip(err['ID'],err['ferr']))

                    for idx in np.where(cond)[0]:

                        try:

                            irac_cat[idx]["FLUX"]    = pht[irac_cat[idx]["NUMBER"]]
                            irac_cat[idx]["FLUXERR"] = err[irac_cat[idx]["NUMBER"]]
                            irac_cat[idx]["NPIX"]    = npx[irac_cat[idx]["NUMBER"]]

                            if err[irac_cat[idx]["NUMBER"]]!=-99.:
                                irac_cat[idx]["FLAGS"] = 0
                                irac_cat[idx]["COVERAGE_FLAG"] = 1
                            else:
                                irac_cat[idx]["FLAGS"] = 2
                                irac_cat[idx]["COVERAGE_FLAG"] = 0

                        except KeyError:

                            xpx,ypx = wcs.all_world2pix([irac_cat[idx]["X_WORLD"],],[irac_cat[idx]["Y_WORLD"],],1)
                            _xpx,_ypx = int(xpx[0]),int(ypx[0])
                            xpx,ypx = np.clip(_xpx,0,rms.shape[0]-1), np.clip(_ypx,0,rms.shape[1]-1)
                            assert np.sqrt((xpx-_xpx)**2+(ypx-_ypx)**2) < 1.5

                            if np.isfinite(rms[ypx,xpx]):
                                irac_cat[idx]["FLAGS"] = 1
                                irac_cat[idx]["COVERAGE_FLAG"] = 1
                            else:
                                irac_cat[idx]["FLAGS"] = 2
                                irac_cat[idx]["COVERAGE_FLAG"] = 0

                except IOError:

                    print ("[ch.%s %i-%2i] No IRACLEAN phot. catalog found!"%(filt,j+1,i+1))

                    for idx in np.where(cond)[0]:
                        irac_cat[idx]["FLAGS"] = 2
                        irac_cat[idx]["COVERAGE_FLAG"] = 0

                    continue

                cond = (irac_cat["FLUX"] > 0)
                irac_cat["MAG"][cond]    = -2.5*np.log10(irac_cat["FLUX"][cond]) + useful.zp
                irac_cat["MAGERR"][cond] =  2.5/np.log(10) * (irac_cat["FLUXERR"][cond]/irac_cat["FLUX"][cond])

                cond = (irac_cat["FLUX"] == 0) & (irac_cat["FLUXERR"] > 0)
                irac_cat["MAG"][cond]    = -2.5*np.log10(irac_cat["FLUXERR"][cond]) + useful.zp
                irac_cat["MAGERR"][cond] = -1.0

                detect = len(irac_cat[cond][(irac_cat[cond]["FLAGS"]==0)])
                found  = len(irac_cat[cond][(irac_cat[cond]["FLAGS"]!=1)])
                total  = len(irac_cat[cond])

                print ("[ch.%s %i-%2i] %5i detected (%5i found) out of %5i [%6.2f%% (%6.2f%% detected)] sources in IRACLEAN catalog" % ( \
                                            filt,j+1,i+1,detect,found,total,detect/float(total)*100,found/float(total)*100))

            fitsio.writeto(os.path.join(cwd,os.pardir,"final_cats","catalog%i_matched_irac_%s.fits"%(j+1,filt)),irac_cat,overwrite=True)

def objs_missed_in_irac():

    catalog = fitsio.getdata("final_cats/final_catalog.fits")

    for filt in useful.filters["irac"]:

        cond2  = (catalog["SE_FLAGS_irac_%s"%filt] == 2)
        cond1  = (catalog["SE_FLAGS_irac_%s"%filt] == 1)
        cond   = (catalog["COVERAGE_FLAG_irac_%s"%filt] == 0)

        cond00 = (catalog["SE_FLAGS_irac_%s"%filt] == 0) & (catalog["COVERAGE_FLAG_irac_%s"%filt] == 0)
        cond10 = (catalog["SE_FLAGS_irac_%s"%filt] == 1) & (catalog["COVERAGE_FLAG_irac_%s"%filt] == 0)
        cond21 = (catalog["SE_FLAGS_irac_%s"%filt] == 2) & (catalog["COVERAGE_FLAG_irac_%s"%filt] == 1)

        print (len(catalog["ID"][cond2]))
        print (len(catalog["ID"][cond1]))
        print (len(catalog["ID"][cond]))

        print (len(catalog["ID"][cond00]))
        print (len(catalog["ID"][cond10]))
        print (len(catalog["ID"][cond21]))

        with open(os.path.join(cwd,"missed_irac_ch%s.reg"%filt),"w") as f: f.write("fk5\n")
        with open(os.path.join(cwd,"missed_irac_ch%s.reg"%filt),"a") as f:
            for cond,color,size in zip([cond2,cond1,cond],['red','blue','green'],[2.5,2.0,1.5])[:-1]:
                fmt = 'circle(%.10f,%.10f,' + str(size) + '") # width=1 ' + 'color=%s'%color
                np.savetxt(f,np.vstack((catalog["RA"][cond],catalog["DEC"][cond])).T,
                            fmt=fmt,header='',comments='')

def check_new_IRACLEAN_cats():

    binsx = np.arange(0,50,0.025)
    bincx = 0.5*(binsx[1:]+binsx[:-1])
    binsy = np.arange(-10.05,10.05,0.025)
    bincy = 0.5*(binsy[1:]+binsy[:-1])

    fig,axes = plt.subplots(2,2,figsize=(20,10),dpi=75,tight_layout=True)
    axes = axes.flatten()

    instr = "irac"
    for ax,filt in zip(axes,useful.filters[instr]):
        hist = np.zeros((len(bincx),len(bincy)))
        for i in range(4):
            old_cat = fitsio.getdata("final_cats/old_cats_v1.5/catalog{:d}_matched_irac_{:s}.fits".format(i+1,filt))
            new_cat = fitsio.getdata("final_cats/catalog{:d}_matched_irac_{:s}.fits".format(i+1,filt))

            _hist = np.histogram2d(new_cat["MAG"],new_cat["MAG"]-old_cat["MAG"],bins=[binsx,binsy])[0]
            hist += _hist

        hist = np.ma.MaskedArray(hist,mask=hist<5)
        ax.pcolormesh(bincx,bincy,hist.T,cmap=plt.cm.inferno,vmin=0,vmax=np.ma.sum(hist)*8e-5)
        ax.axhline(0,c='k',ls='--',lw=0.8)
        ax.set_xlabel("New mags [AB]",fontsize=18)
        ax.set_ylabel("(New - Old) mags [AB]",fontsize=18)
        ax.set_xlim(17,29)
        ax.set_ylim(-5,5)
        ax.text(0.05,0.95,"{:s}:{:s}".format(instr,filt),fontsize=20,fontweight=600,va='top',ha='left',transform=ax.transAxes)

    fig.savefig("final_cats/plots/chk_new_IRACLEAN.png")

if __name__ == '__main__':

    cwd = "/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/irac_phot/"

    # rebin_PRFs(indir=os.path.join(cwd,"PRFs"),outdir=os.path.join(cwd,"PRFs.rb_060"))

    # setup_iraclean()
    # plot_cutouts()

    # run_iraclean(filt=useful.filters["irac"][0])
    # run_iraclean(filt=useful.filters["irac"][1])
    # run_iraclean(filt=useful.filters["irac"][2])
    # run_iraclean(filt=useful.filters["irac"][3])

    # mk_cats()

    # objs_missed_in_irac()

    check_new_IRACLEAN_cats()
    plt.show()

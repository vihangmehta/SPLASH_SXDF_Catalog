import os,sys,subprocess,time,glob
import numpy as np
import astropy.io.fits as fitsio
from multiprocessing import Queue,Process
from numpy.lib.recfunctions import stack_arrays

import useful

def lephare_setup(phys=False):

    if not phys:
        os.system('$LEPHAREDIR/source/sedtolib -t S -c lephare/photoz/zphot_z01.para')
        os.system('$LEPHAREDIR/source/sedtolib -t Q -c lephare/photoz/zphot_z01.para')
        os.system('$LEPHAREDIR/source/sedtolib -t G -c lephare/photoz/zphot_z01.para')

        os.system('$LEPHAREDIR/source/filter        -c lephare/photoz/zphot_z01.para')

        os.system('$LEPHAREDIR/source/mag_star      -c lephare/photoz/zphot_z01.para')
        os.system('$LEPHAREDIR/source/mag_gal  -t Q -c lephare/photoz/zphot_z01.para -EB_V 0')
        os.system('$LEPHAREDIR/source/mag_gal  -t G -c lephare/photoz/zphot_z01.para')

    else:
        os.system('$LEPHAREDIR/source/sedtolib -t G -c lephare/phys/zphot.para')
        os.system('$LEPHAREDIR/source/mag_gal  -t G -c lephare/phys/zphot.para')

def get_context(catalog):

    magcols = np.array([i for i in useful.fnames])
    context = np.ones((len(magcols),len(catalog)))
    for i,mag in enumerate(magcols):
        context[i,:][catalog["FLUX_"+mag]==-99.] = 0
        context[i,:] *= 2**i
    return np.sum(context,axis=0)

def convert_uJy_to_cgs(flux_uJy):
    """
    1 micro Jy = 1e-29 ergs/s/cm^2/Hz
    """
    cond = (flux_uJy!=-99)
    flux_cgs = np.zeros(len(flux_uJy)) - 99.
    flux_cgs[cond] = flux_uJy[cond] * 1e-29
    return flux_cgs

def mk_photoz_input(cat_dir,run_dir,errfix,aper_num,full=False):

    errfix = '_errfix' if errfix else ''
    catalog = fitsio.getdata(os.path.join(cat_dir,'final_catalog%s.extra.fits'%errfix))

    dtype = [('ID',int),]
    for fname in useful.fnames:
        dtype += [('FLUX_%s'%fname,float),('FLUXERR_%s'%fname,float)]
    dtype += [('CONTEXT',int),('ZSPEC',float)]

    catalog_aper = np.recarray((len(catalog),),dtype=dtype)
    for x in catalog_aper.dtype.names: catalog_aper[x] = -99.

    catalog_aper['ID'] = catalog['ID']

    for fname in useful.fnames:

        if "irac" not in fname:

            # Take the aperture fluxes for all filters except IRAC
            catalog_aper[   "FLUX_%s"%fname] = catalog[   "FLUX_APER_%s"%fname][:,aper_num]
            catalog_aper["FLUXERR_%s"%fname] = catalog["FLUXERR_APER_%s"%fname][:,aper_num]

        else:

            # Take the total fluxes for IRAC
            catalog_aper[   "FLUX_%s"%fname] = catalog[   "FLUX_TOT_%s"%fname]
            catalog_aper["FLUXERR_%s"%fname] = catalog["FLUXERR_TOT_%s"%fname]

            # Put everything that has an OFFSET_FLUX of -99. to -99. (can't trust the OFFSET_FLUX)
            cond = (catalog["OFFSET_FLUX"][:,aper_num]==-99.)
            catalog_aper[   "FLUX_%s"%fname][cond] = -99.
            catalog_aper["FLUXERR_%s"%fname][cond] = -99.

            # Apply OFFSET_FLUX separately for FLUX and FLUXERR to account for the upper lims
            cond = (catalog_aper["FLUX_%s"%fname]>0)
            catalog_aper[   "FLUX_%s"%fname][cond] = catalog_aper[   "FLUX_%s"%fname][cond] / catalog["OFFSET_FLUX"][:,aper_num][cond]
            cond = (catalog_aper["FLUXERR_%s"%fname]>0)
            catalog_aper["FLUXERR_%s"%fname][cond] = catalog_aper["FLUXERR_%s"%fname][cond] / catalog["OFFSET_FLUX"][:,aper_num][cond]

        # Convert uJY to cgs
        catalog_aper[   "FLUX_%s"%fname] = convert_uJy_to_cgs(catalog_aper[   "FLUX_%s"%fname])
        catalog_aper["FLUXERR_%s"%fname] = convert_uJy_to_cgs(catalog_aper["FLUXERR_%s"%fname])

    # Remove IRAC ch.3 and 4
    for fname in ["irac_3","irac_4"]:
        catalog_aper["FLUX_%s"%fname] = -99.
        catalog_aper["FLUXERR_%s"%fname] = -99.

    catalog_aper['CONTEXT'] = get_context(catalog_aper)
    catalog_aper['ZSPEC']   = catalog["ZSPEC"]

    cond_calib = (catalog['USE_ZSPEC_FLAG']==1)

    print ("Full photo-z input catalog [%i\" aperture]: %i sources (out of %s sources)" % (useful.apersizes[aper_num],len(catalog_aper),len(catalog)))
    print ("Calibration  input catalog [%i\" aperture]: %i sources (out of %s sources)" % (useful.apersizes[aper_num],len(catalog_aper[cond_calib]),len(catalog)))

    flxcols = np.array([i for i in catalog_aper.dtype.names if "FLUX_" in i])
    hdr = "%8s" % "ID"
    for x in flxcols: hdr += "%30s" % x.replace("FLUX_","")
    hdr += "%15s%10s"% ("CNTXT","ZSPEC")
    fmt = "%10i" + "".join(["%15.6e%15.6e"]*len(flxcols)) + "%15i%10.4f"

    if full:

        np.savetxt(os.path.join(run_dir,'photoz','catalog%s_r%i.in'%(errfix,useful.apersizes[aper_num])),catalog_aper,fmt=fmt,header=hdr)

        num_procs = 15
        catalog_aper = np.array_split(catalog_aper,num_procs*2)
        for i,_catalog_aper in enumerate(catalog_aper):
            sys.stdout.write("\rWriting out sub-file #%i out %i ... " % (i+1,num_procs*2))
            sys.stdout.flush()
            np.savetxt(os.path.join(run_dir,'photoz','catalog%s_r%i_%02d.in'%(errfix,useful.apersizes[aper_num],i+1)),_catalog_aper,fmt=fmt,header=hdr)
        print ("done.")

    else:

        np.savetxt(os.path.join(run_dir,'calib','catalog%s_r%i.in'%(errfix,useful.apersizes[aper_num])),catalog_aper[cond_calib],fmt=fmt,header=hdr)

def get_errscale():

    opt = 0.02
    nir = 0.05
    irc = 0.1
    zro = 0.0

    text  = " -ERR_SCALE "
    text += "%.2f,%.2f,%.2f,%.2f,%.2f,"%(opt,opt,opt,opt,opt) #HSC
    text += "%.2f,%.2f,%.2f,%.2f,%.2f,"%(opt,opt,opt,opt,opt) #SupCam
    text += "%.2f,%.2f,%.2f,"          %(nir,nir,nir)         #UDS
    text += "%.2f,%.2f,%.2f,%.2f,%.2f,"%(opt,opt,nir,nir,nir) #VIDEO
    text += "%.2f,"                    % opt                  #CFHT
    text += "%.2f,%.2f,%.2f,%.2f,%.2f,"%(opt,opt,opt,opt,opt) #CFHTLS
    text += "%.2f,%.2f,%.2f,%.2f"      %(nir,irc,zro,zro)     #UDS

    return text

def lephare_calib_run(run_dir,errfix=True,aper_num=1):

    mk_photoz_input(cat_dir=cat_dir,run_dir=run_dir,errfix=errfix,aper_num=aper_num)

    errfix = '_errfix' if errfix else ''
    config  = os.path.join(run_dir,'calib','zphot_z01.para')
    paramf  = os.path.join(run_dir,'calib','zphot_out.para')
    cat_in  = os.path.join(run_dir,'calib','catalog%s_r%i.in' %(errfix,useful.apersizes[aper_num]))
    cat_out = os.path.join(run_dir,'calib','catalog%s_r%i.out'%(errfix,useful.apersizes[aper_num]))
    pdz_out = os.path.join(run_dir,'calib','catalog%s_r%i'%(errfix,useful.apersizes[aper_num]))

    call = '$LEPHAREDIR/source/zphota -c %s -CAT_IN %s -CAT_OUT %s -PDZ_OUT %s -PARA_OUT %s -AUTO_ADAPT YES' % (config,cat_in,cat_out,pdz_out,paramf)
    call+= get_errscale()
    useful.run(call,cwd=os.path.join(run_dir,'calib'),verbose=True)

def lephare_photoz_run(run_num,run_dir,errfix=True,aper_num=2):

    errfix = '_errfix' if errfix else ''
    config  = os.path.join(run_dir,'photoz','zphot_z01.para')
    paramf  = os.path.join(run_dir,'photoz','zphot_out.para')

    if errfix and aper_num==1:
        # This was the older v1.5 offsets
        #adapt_zp = "-0.068,-0.042,-0.024,-0.063,-0.047,-0.049,-0.026,-0.127,0.043,0.062,0.052,0.017,-0.013,-0.047,-0.036,0.009,0.026,-0.066,0.031,0.193,0.094,0.125,0.044,0.137,0.100,0.097,0.000,0.000"
        # This is the new v1.6 offsets
        adapt_zp = "-0.069,-0.042,-0.020,-0.059,-0.043,-0.050,-0.027,-0.127,0.046,0.066,0.057,0.023,-0.010,-0.043,-0.031,0.014,0.033,-0.060,0.031,0.193,0.093,0.126,0.047,0.141,0.072,0.067,0.000,0.000"
    # if errfix and aper_num==2:
        # This was the older v1.5 offsets
        # adapt_zp = "-0.058,-0.043,-0.015,-0.052,-0.044,-0.049,-0.029,-0.129,0.031,0.058,0.059,0.029,0.005,-0.035,-0.017,0.035,0.046,-0.047,0.029,0.180,0.092,0.134,0.056,0.117,0.026,0.023,0.000,0.000"

    def slave(queue,i):

        cat_in   = os.path.join(run_dir,'photoz','catalog%s_r%i_%02d.in' %(errfix,useful.apersizes[aper_num],i+1))
        cat_out  = os.path.join(run_dir,'photoz','catalog%s_r%i_%02d.out'%(errfix,useful.apersizes[aper_num],i+1))
        pdz_out  = os.path.join(run_dir,'photoz','catalog%s_r%i_%02d'    %(errfix,useful.apersizes[aper_num],i+1))
        log_file = os.path.join(run_dir,'photoz','catalog%s_r%i_%02d.log'%(errfix,useful.apersizes[aper_num],i+1))

        call = '$LEPHAREDIR/source/zphota -c %s -CAT_IN %s -CAT_OUT %s -PDZ_OUT %s -PARA_OUT %s -APPLY_SYSSHIFT %s' % (config,cat_in,cat_out,pdz_out,paramf,adapt_zp)
        call+= get_errscale()
        print (call)
        # useful.run(call,cwd=os.path.join(run_dir,'photoz'),verbose=False)

        start = time.time()
        f = open(log_file,'w')
        p = subprocess.Popen(call, stdout=f, stderr=f, cwd=os.path.join(run_dir,'photoz'), shell=True)
        p.communicate()
        p.wait()
        end = time.time()
        f.write("\n\nTime taken: %.2f seconds"%(end-start))
        print ("Run #%i done in %.2f seconds"%(i+1,end-start))

        queue.put(1)

    num_procs = 15
    queue = Queue()
    procs = [Process(target=slave,args=(queue,i)) for i in range((run_num-1)*num_procs,run_num*num_procs)]
    for proc in procs: proc.start()

    finished = 0
    while finished < num_procs:
        items = queue.get()
        if items == 1:
            finished += 1

    for proc in procs: proc.join()

def mk_phys_input(cat_dir,run_dir,errfix,zphot_fname):

    errfix = '_errfix' if errfix else ''
    catalog = fitsio.getdata(os.path.join(cat_dir,'final_catalog.extra.fits'))

    photoz_cat = fitsio.getdata(os.path.join(run_dir,'photoz',zphot_fname))
    photoz = photoz_cat["Z_ML"]
    cond = (photoz == -99)
    photoz[cond] = photoz_cat["Z_BEST"][cond]
    assert np.all(catalog["ID"] == photoz_cat["ID"])

    dtype = [('ID',int),]
    for fname in useful.fnames:
        dtype += [('FLUX_%s'%fname,float),('FLUXERR_%s'%fname,float)]
    dtype += [('CONTEXT',int),('ZSPEC',float)]

    catalog_auto = np.recarray((len(catalog),),dtype=dtype)
    for x in catalog_auto.dtype.names: catalog_auto[x] = -99.

    catalog_auto['ID'] = catalog["ID"]

    for fname in useful.fnames:

        if "irac" not in fname:

            catalog_auto[   "FLUX_%s"%fname] = catalog[   "FLUX_AUTO_%s"%fname]
            catalog_auto["FLUXERR_%s"%fname] = catalog["FLUXERR_AUTO_%s"%fname]

        else:

            catalog_auto[   "FLUX_%s"%fname] = catalog[   "FLUX_TOT_%s"%fname]
            catalog_auto["FLUXERR_%s"%fname] = catalog["FLUXERR_TOT_%s"%fname]

        catalog_auto[   "FLUX_%s"%fname] = convert_uJy_to_cgs(catalog_auto[   "FLUX_%s"%fname])
        catalog_auto["FLUXERR_%s"%fname] = convert_uJy_to_cgs(catalog_auto["FLUXERR_%s"%fname])

    # for fname in ["irac_3","irac_4"]:
    #     catalog_auto["FLUX_%s"%fname] = -99.
    #     catalog_auto["FLUXERR_%s"%fname] = -99.

    catalog_auto['CONTEXT'] = get_context(catalog_auto)
    catalog_auto['ZSPEC']   = photoz

    print ("Full phys input catalog : %i sources (out of %s sources)" % (len(catalog_auto),len(catalog)))

    flxcols = np.array([i for i in catalog_auto.dtype.names if "FLUX_" in i])
    hdr = "%8s" % "ID"
    for x in flxcols: hdr += "%30s" % x.replace("FLUX_","")
    hdr += "%15s%10s"% ("CNTXT","ZSPEC")
    fmt = "%10i" + "".join(["%15.6e%15.6e"]*len(flxcols)) + "%15i%10.4f"

    np.savetxt(os.path.join(run_dir,'phys','catalog%s.in'%errfix),catalog_auto,fmt=fmt,header=hdr)

    num_procs = 6
    catalog_auto = np.array_split(catalog_auto,num_procs*4)
    for i,_catalog_auto in enumerate(catalog_auto):
        sys.stdout.write("\rWriting out sub-file #%i out of %i ... " % (i+1,num_procs*4))
        sys.stdout.flush()
        np.savetxt(os.path.join(run_dir,'phys','catalog%s_%02d.in'%(errfix,i+1)),_catalog_auto,fmt=fmt,header=hdr)
    print ("done.")

def lephare_phys_run(run_num,run_dir,errfix=True):

    errfix = '_errfix' if errfix else ''
    config  = os.path.join(run_dir,'phys','zphot.para')
    paramf  = os.path.join(run_dir,'phys','zphot_out.para')
    gen_specfiles = 'NO'

    def slave(queue,i):
        cat_in   = os.path.join(run_dir,'phys','catalog%s_%02d.in' %(errfix,i+1))
        cat_out  = os.path.join(run_dir,'phys','catalog%s_%02d.out'%(errfix,i+1))
        log_file = os.path.join(run_dir,'phys','catalog%s_%02d.log'%(errfix,i+1))

        call = '$LEPHAREDIR/source/zphota -c %s -CAT_IN %s -CAT_OUT %s -SPEC_OUT %s -PARA_OUT %s -ZFIX YES' % (config,cat_in,cat_out,gen_specfiles,paramf)
        call+= get_errscale()
        print (call)
        # useful.run(call,cwd=os.path.join(run_dir,'phys'),verbose=False)

        start = time.time()
        f = open(log_file,'w')
        p = subprocess.Popen(call, stdout=f, stderr=f, cwd=os.path.join(run_dir,'phys'), shell=True)
        p.communicate()
        p.wait()
        end = time.time()
        f.write("\n\nTime taken: %.2f seconds"%(end-start))
        print ("Run #%i done in %.2f seconds"%(i+1,end-start))

        queue.put(1)

    num_procs = 6
    queue = Queue()
    # procs = [Process(target=slave,args=(queue,i)) for i in range(num_procs)]
    procs = [Process(target=slave,args=(queue,i)) for i in range((run_num-1)*num_procs,run_num*num_procs)]
    for proc in procs: proc.start()

    finished = 0
    while finished < num_procs:
        items = queue.get()
        if items == 1:
            finished += 1

    for proc in procs: proc.join()

def mk_combined_zphot():

    catalog_zphot = fitsio.getdata("lephare/photoz/catalog_errfix_r2.out.fits")
    catalog_phys  = fitsio.getdata("lephare/phys/catalog_errfix.out.fits")

    catalog_lephare = np.recarray(len(catalog_zphot),
                            dtype=[('LPH_ID','>i4'),

                                   ('LPH_Z_BEST','>f8'),('LPH_Z_BEST68_LOW','>f8'),('LPH_Z_BEST68_HIGH','>f8'),
                                   ('LPH_Z_ML','>f8'),('LPH_Z_ML68_LOW','>f8'),('LPH_Z_ML68_HIGH','>f8'),
                                   ('LPH_CHI_BEST','>f8'),('LPH_MOD_BEST','>i4'),('LPH_EXTLAW_BEST','>i4'),('LPH_EBV_BEST','>f8'),
                                   ('LPH_PDZ_BEST','>f8'),
                                   # ('LPH_SCALE_BEST','>f8'),('LPH_DIST_MOD_BEST','>f8'),
                                   ('LPH_NBAND_USED','>i4'),
                                   ('LPH_Z_SEC','>f8'),('LPH_CHI_SEC','>f8'),('LPH_MOD_SEC','>i4'),
                                   ('LPH_Z_QSO','>f8'),('LPH_CHI_QSO','>f8'),('LPH_MOD_QSO','>i4'),
                                   ('LPH_CHI_STAR','>f8'),('LPH_MOD_STAR','>i4'),
                                   # ('LPH_CONTEXT','>i4'),('LPH_ZSPEC','>f4'),

                                   ('LPH_CHI_BEST_PHYS','>f8'),
                                   ('LPH_MOD_BEST_PHYS','>i4'),('LPH_EBV_BEST_PHYS','>f8'),('LPH_EXTLAW_BEST_PHYS','>i4'),
                                   # ('LPH_PDZ_BEST_PHYS','>f8'),('LPH_SCALE_BEST_PHYS','>f8'),('LPH_DIST_MOD_BEST_PHYS','>f8'),
                                   ('LPH_NBAND_USED_PHYS','>i4'),('LPH_MAG_ABS','>f8',(28,)),
                                   # ('LPH_K_COR','>f8',(28,)),('LPH_MABS_FILT','>i4',(28,)),
                                   ('LPH_AGE_BEST','>f8'),('LPH_MASS_BEST','>f8'),('LPH_SFR_BEST','>f8'),('LPH_SSFR_BEST','>f8'),
                                   ('LPH_LUM_NUV_BEST','>f8'),('LPH_LUM_R_BEST','>f8'),('LPH_LUM_K_BEST','>f8')])

    for x in catalog_lephare.dtype.names: catalog_lephare[x] = -99.

    catalog_lephare['LPH_ID']                 = catalog_zphot['ID']

    catalog_lephare['LPH_Z_BEST']             = catalog_zphot['Z_BEST']
    catalog_lephare['LPH_Z_BEST68_LOW']       = catalog_zphot['Z_BEST68_LOW']
    catalog_lephare['LPH_Z_BEST68_HIGH']      = catalog_zphot['Z_BEST68_HIGH']
    catalog_lephare['LPH_Z_ML']               = catalog_zphot['Z_ML']
    catalog_lephare['LPH_Z_ML68_LOW']         = catalog_zphot['Z_ML68_LOW']
    catalog_lephare['LPH_Z_ML68_HIGH']        = catalog_zphot['Z_ML68_HIGH']
    catalog_lephare['LPH_CHI_BEST']           = catalog_zphot['CHI_BEST']
    catalog_lephare['LPH_MOD_BEST']           = catalog_zphot['MOD_BEST']
    catalog_lephare['LPH_EXTLAW_BEST']        = catalog_zphot['EXTLAW_BEST']
    catalog_lephare['LPH_EBV_BEST']           = catalog_zphot['EBV_BEST']
    catalog_lephare['LPH_PDZ_BEST']           = catalog_zphot['PDZ_BEST']
    # catalog_lephare['LPH_SCALE_BEST']         = catalog_zphot['SCALE_BEST']
    # catalog_lephare['LPH_DIST_MOD_BEST']      = catalog_zphot['DIST_MOD_BEST']
    catalog_lephare['LPH_NBAND_USED']         = catalog_zphot['NBAND_USED']
    # catalog_lephare['LPH_NBAND_ULIM']         = catalog_zphot['NBAND_ULIM']
    catalog_lephare['LPH_Z_SEC']              = catalog_zphot['Z_SEC']
    catalog_lephare['LPH_CHI_SEC']            = catalog_zphot['CHI_SEC']
    catalog_lephare['LPH_MOD_SEC']            = catalog_zphot['MOD_SEC']
    catalog_lephare['LPH_Z_QSO']              = catalog_zphot['Z_QSO']
    catalog_lephare['LPH_CHI_QSO']            = catalog_zphot['CHI_QSO']
    catalog_lephare['LPH_MOD_QSO']            = catalog_zphot['MOD_QSO']
    catalog_lephare['LPH_CHI_STAR']           = catalog_zphot['CHI_STAR']
    catalog_lephare['LPH_MOD_STAR']           = catalog_zphot['MOD_STAR']
    # catalog_lephare['LPH_CONTEXT']            = catalog_zphot['CONTEXT']
    # catalog_lephare['LPH_ZSPEC']              = catalog_zphot['ZSPEC']

    # catalog_lephare['LPH_Z_BEST_PHYS']        = catalog_phys['Z_BEST']
    catalog_lephare['LPH_CHI_BEST_PHYS']      = catalog_phys['CHI_BEST']
    catalog_lephare['LPH_MOD_BEST_PHYS']      = catalog_phys['MOD_BEST']
    catalog_lephare['LPH_EXTLAW_BEST_PHYS']   = catalog_phys['EXTLAW_BEST']
    catalog_lephare['LPH_EBV_BEST_PHYS']      = catalog_phys['EBV_BEST']
    # catalog_lephare['LPH_PDZ_BEST_PHYS']      = catalog_phys['PDZ_BEST']
    # catalog_lephare['LPH_SCALE_BEST_PHYS']    = catalog_phys['SCALE_BEST']
    # catalog_lephare['LPH_DIST_MOD_BEST_PHYS'] = catalog_phys['DIST_MOD_BEST']
    catalog_lephare['LPH_NBAND_USED_PHYS']    = catalog_phys['NBAND_USED']
    # catalog_lephare['LPH_NBAND_ULIM_PHYS']    = catalog_phys['NBAND_ULIM']
    # catalog_lephare['LPH_K_COR']              = catalog_phys['K_COR']
    catalog_lephare['LPH_MAG_ABS']            = catalog_phys['MAG_ABS']
    # catalog_lephare['LPH_MABS_FILT']          = catalog_phys['MABS_FILT']
    catalog_lephare['LPH_AGE_BEST']           = catalog_phys['AGE_BEST']
    catalog_lephare['LPH_MASS_BEST']          = catalog_phys['MASS_BEST']
    catalog_lephare['LPH_SFR_BEST']           = catalog_phys['SFR_BEST']
    catalog_lephare['LPH_SSFR_BEST']          = catalog_phys['SSFR_BEST']
    catalog_lephare['LPH_LUM_NUV_BEST']       = catalog_phys['LUM_NUV_BEST']
    catalog_lephare['LPH_LUM_R_BEST']         = catalog_phys['LUM_R_BEST']
    catalog_lephare['LPH_LUM_K_BEST']         = catalog_phys['LUM_K_BEST']

    fitsio.writeto('lephare/lephare_combined.fits',catalog_lephare,overwrite=True)

def mk_final_value_added_cat(cat_dir):

    cat = fitsio.getdata(os.path.join(cat_dir,'final_catalog_errfix.fits'))
    cat_ext = fitsio.getdata(os.path.join(cat_dir,'final_catalog_errfix.extra.fits'))
    cat_lephare = fitsio.getdata('lephare/lephare_combined.fits')

    cat_ext_dtype = [x for x in cat_ext.dtype.descr if x not in cat.dtype.descr]

    cat_va = np.recarray(len(cat),dtype=cat.dtype.descr+cat_lephare.dtype.descr)
    cat_va_ext = np.recarray(len(cat),dtype=cat.dtype.descr+cat_lephare.dtype.descr+cat_ext_dtype)

    for x in cat.dtype.names:
        cat_va[x] = cat[x]
        cat_va_ext[x] = cat_ext[x]

    for x in cat_ext_dtype:
        cat_va_ext[x[0]] = cat_ext[x[0]]

    for x in cat_lephare.dtype.names:
        cat_va[x] = cat_lephare[x]
        cat_va_ext[x] = cat_lephare[x]

    fitsio.writeto(os.path.join(cat_dir,'final_catalog_errfix_zphot.fits'),cat_va,overwrite=True)
    fitsio.writeto(os.path.join(cat_dir,'final_catalog_errfix_zphot.extra.fits'),cat_va_ext,overwrite=True)

def mk_fits_output(dirname,iters,aper_num=None):

    if aper_num is not None:
        # flist = sorted(glob.glob("%s/catalog%s_r%i_*.out"%(dirname,"_errfix",useful.apersizes[aper_num])))
        flist = ["%s/catalog%s_r%i_%02d.out"%(dirname,"_errfix",useful.apersizes[aper_num],i+1) for i in range(iters)]
        outname = "%s/catalog%s_r%i.out.fits"%(dirname,"_errfix",useful.apersizes[aper_num])
        read_fn = useful.read_lephare_photoz
    else:
        # flist = sorted(glob.glob("%s/catalog%s_*.out"%(dirname,"_errfix")))
        flist = ["%s/catalog%s_%02d.out"%(dirname,"_errfix",i+1) for i in range(iters)]
        outname = "%s/catalog%s.out.fits"%(dirname,"_errfix")
        read_fn = useful.read_lephare_phys

    output = read_fn(os.path.join(dirname,flist[0]))

    for fname in flist[1:]:

        sys.stdout.write("\rProcessing %s ... \033[K" % fname.split('/')[-1])
        sys.stdout.flush()

        _output = read_fn(fname)
        dtype = [x for x in _output.dtype.names if len(_output[x].shape)<2]

        output = stack_arrays((output,_output[dtype]),
                                usemask=False,asrecarray=True)

        for x in _output.dtype.names:
            if x not in dtype:
                output[x][-len(_output):] = _output[x]
    print()
    fitsio.writeto(outname,output,overwrite=True)

if __name__ == '__main__':

    cwd = '/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/'
    run_dir = os.path.join(cwd,'lephare')
    cat_dir = os.path.join(cwd,'final_cats')

    # lephare_setup(phys=False)
    # lephare_setup(phys=True)

    # mk_photoz_input(cat_dir=cat_dir,run_dir=run_dir,errfix=True,aper_num=1,full=False)

    # lephare_calib_run(run_dir=run_dir,errfix=True,aper_num=1)
    # lephare_calib_run(run_dir=run_dir,errfix=True,aper_num=2)
    # lephare_calib_run(run_dir=run_dir,errfix=False,aper_num=1)
    # lephare_calib_run(run_dir=run_dir,errfix=False,aper_num=2)

    # mk_photoz_input(cat_dir=cat_dir,run_dir=run_dir,errfix=True,aper_num=1,full=True)
    # lephare_photoz_run(run_num=1,run_dir=run_dir,errfix=True,aper_num=1)
    # lephare_photoz_run(run_num=2,run_dir=run_dir,errfix=True,aper_num=1)
    # os.system("cat %s/catalog%s_r%i_*.out > %s/catalog%s_r%i.out" % (\
    #             os.path.join(run_dir,'photoz'),"_errfix",useful.apersizes[1],
    #             os.path.join(run_dir,'photoz'),"_errfix",useful.apersizes[1]))
    # mk_fits_output(dirname=os.path.join(run_dir,'photoz'),iters=30,aper_num=1)

    # mk_phys_input(cat_dir=cat_dir,run_dir=run_dir,errfix=True,zphot_fname='catalog_errfix_r2.out.fits')
    # lephare_phys_run(run_num=1,run_dir=run_dir,errfix=True)
    # lephare_phys_run(run_num=2,run_dir=run_dir,errfix=True)
    # lephare_phys_run(run_num=3,run_dir=run_dir,errfix=True)
    # lephare_phys_run(run_num=4,run_dir=run_dir,errfix=True)
    # os.system("cat %s/catalog%s_*.out > %s/catalog%s.out" % (\
    #             os.path.join(run_dir,'phys'),"_errfix",
    #             os.path.join(run_dir,'phys'),"_errfix"))
    # mk_fits_output(dirname=os.path.join(run_dir,'phys'),iters=24)


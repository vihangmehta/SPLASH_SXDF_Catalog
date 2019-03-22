import os,sys,subprocess,time,glob
import numpy as np
import astropy.io.fits as fitsio
from multiprocessing import Queue,Process
from numpy.lib.recfunctions import stack_arrays

import useful

def get_context(catalog):

    magcols = np.array([i for i in catalog.dtype.names if "FLUX_" in i])
    context = np.ones((len(magcols),len(catalog)))
    for i,mag in enumerate(magcols):
        context[i,:][catalog[mag]==-99.] = 0
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

def calc_opt_nir_offset(catalog):

    term1,term2 = np.zeros(len(catalog)),np.zeros(len(catalog))

    for instr in useful.instr_used_list[:-1]:

        for filt in useful.filters[instr]:

            sys.stdout.write("\rCalculating AUTO-ISO offset for %s:%s ... \033[K" % (instr,filt))
            sys.stdout.flush()

            # Just a simple check for if FLUX > 0
            cond_f = (catalog[   'FLUX_AUTO_%s_%s'%(instr,filt)] > 0.0) & (catalog[   'FLUX_ISO_%s_%s'%(instr,filt)] > 0.0)

            # Check if SN > 3 sigma
            # cond_f = (catalog['FLUX_AUTO_%s_%s'%(instr,filt)] / catalog['FLUXERR_AUTO_%s_%s'%(instr,filt)] > 3.0) & \
            #          (catalog['FLUX_APER_%s_%s'%(instr,filt)][:,aper_num] / catalog['FLUXERR_APER_%s_%s'%(instr,filt)][:,aper_num] > 3.0)

            diff_f,wht_f = np.zeros(len(catalog)),np.zeros(len(catalog))
            diff_f[cond_f] = catalog['FLUX_AUTO_%s_%s'%(instr,filt)][cond_f] / catalog['FLUX_ISO_%s_%s'%(instr,filt)][cond_f]
            wht_f[cond_f] = 1./(catalog['FLUXERR_AUTO_%s_%s'%(instr,filt)][cond_f]**2 + catalog['FLUXERR_ISO_%s_%s'%(instr,filt)][cond_f]**2)
            term1 += wht_f * diff_f
            term2 += wht_f

    cond = (term2>0)
    catalog["OFFSET_FLUX"][:,0][cond] = term1[cond]/term2[cond]

    cond = (catalog["OFFSET_FLUX"][:,0] > 0)
    catalog[ "OFFSET_MAG"][:,0][cond] = -2.5*np.log10(catalog["OFFSET_FLUX"][:,0][cond])

    print "done."
    return catalog

def mk_photoz_input_iso(cat_dir,run_dir,errfix):

    errfix = '_errfix' if errfix else ''
    catalog = fitsio.getdata(os.path.join(cat_dir,'final_catalog%s.extra.fits'%errfix))
    catalog = calc_opt_nir_offset(catalog)

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
            catalog_aper[   "FLUX_%s"%fname] = catalog[   "FLUX_ISO_%s"%fname]
            catalog_aper["FLUXERR_%s"%fname] = catalog["FLUXERR_ISO_%s"%fname]

        else:

            # Take the total fluxes for IRAC
            catalog_aper[   "FLUX_%s"%fname] = catalog[   "FLUX_TOT_%s"%fname]
            catalog_aper["FLUXERR_%s"%fname] = catalog["FLUXERR_TOT_%s"%fname]

            # Put everything that has an OFFSET_FLUX of -99. to -99. (can't trust the OFFSET_FLUX)
            cond = (catalog["OFFSET_FLUX"][:,0]==-99.)
            catalog_aper[   "FLUX_%s"%fname][cond] = -99.
            catalog_aper["FLUXERR_%s"%fname][cond] = -99.

            # Apply OFFSET_FLUX separately for FLUX and FLUXERR to account for the upper lims
            cond = (catalog_aper["FLUX_%s"%fname]!=-99.)
            catalog_aper[   "FLUX_%s"%fname][cond] = catalog_aper[   "FLUX_%s"%fname][cond] / catalog["OFFSET_FLUX"][:,0][cond]
            cond = (catalog_aper["FLUXERR_%s"%fname]!=-99.)
            catalog_aper["FLUXERR_%s"%fname][cond] = catalog_aper["FLUXERR_%s"%fname][cond] / catalog["OFFSET_FLUX"][:,0][cond]

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

    print "Full photo-z input catalog [ISO aperture]: %i sources (out of %s sources)" % (len(catalog_aper),len(catalog))
    print "Calibration  input catalog [ISO aperture]: %i sources (out of %s sources)" % (len(catalog_aper[cond_calib]),len(catalog))

    flxcols = np.array([i for i in catalog_aper.dtype.names if "FLUX_" in i])
    hdr = "%8s" % "ID"
    for x in flxcols: hdr += "%30s" % x.replace("FLUX_","")
    hdr += "%15s%10s"% ("CNTXT","ZSPEC")
    fmt = "%10i" + "".join(["%15.6e%15.6e"]*len(flxcols)) + "%15i%10.4f"

    np.savetxt(os.path.join(run_dir,'calib','catalog%s_iso.in'%errfix),catalog_aper[cond_calib],fmt=fmt,header=hdr)

def mk_photoz_input_no_hsc(cat_dir,run_dir,errfix,aper_num):

    errfix = '_errfix' if errfix else ''
    catalog = fitsio.getdata(os.path.join(cat_dir,'final_catalog%s.extra.fits'%errfix))
    fnames_no_hsc = ["video_y","video_j","video_h","video_ks",
                     "cfhtls_u","cfhtls_g","cfhtls_r","cfhtls_i","cfhtls_z",
                     "irac_1","irac_2","irac_3","irac_4"]

    dtype = [('ID',int),]
    for fname in fnames_no_hsc:
        dtype += [('FLUX_%s'%fname,float),('FLUXERR_%s'%fname,float)]
    dtype += [('CONTEXT',int),('ZSPEC',float)]

    catalog_aper = np.recarray((len(catalog),),dtype=dtype)
    for x in catalog_aper.dtype.names: catalog_aper[x] = -99.

    catalog_aper['ID'] = catalog['ID']

    for fname in fnames_no_hsc:

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
            cond = (catalog_aper["FLUX_%s"%fname]!=-99.)
            catalog_aper[   "FLUX_%s"%fname][cond] = catalog_aper[   "FLUX_%s"%fname][cond] / catalog["OFFSET_FLUX"][:,aper_num][cond]
            cond = (catalog_aper["FLUXERR_%s"%fname]!=-99.)
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

    cond_calib = (catalog['USE_ZSPEC_FLAG']==0) & \
                 (catalog["ZSPEC"]!=-99.) & \
                 (catalog['COVERAGE_FLAG_hsc_g']==0) & \
                 (catalog['COVERAGE_FLAG_hsc_r']==0) & \
                 (catalog['COVERAGE_FLAG_hsc_i']==0) & \
                 (catalog['COVERAGE_FLAG_hsc_z']==0) & \
                 (catalog['COVERAGE_FLAG_hsc_y']==0)

    print "Full photo-z input catalog [%i\" aperture]: %i sources (out of %s sources)" % (useful.apersizes[aper_num],len(catalog_aper),len(catalog))
    print "Calibration  input catalog [%i\" aperture]: %i sources (out of %s sources)" % (useful.apersizes[aper_num],len(catalog_aper[cond_calib]),len(catalog))

    flxcols = np.array([i for i in catalog_aper.dtype.names if "FLUX_" in i])
    hdr = "%8s" % "ID"
    for x in flxcols: hdr += "%30s" % x.replace("FLUX_","")
    hdr += "%15s%10s"% ("CNTXT","ZSPEC")
    fmt = "%10i" + "".join(["%15.6e%15.6e"]*len(flxcols)) + "%15i%10.4f"

    print len(catalog_aper[cond_calib])
    print os.path.join(run_dir,'calib','catalog%s_r%i_no_hsc.in'%(errfix,useful.apersizes[aper_num]))
    np.savetxt(os.path.join(run_dir,'calib','catalog%s_r%i_no_hsc.in'%(errfix,useful.apersizes[aper_num])),catalog_aper[cond_calib],fmt=fmt,header=hdr)

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

def lephare_calib_run_iso(run_dir,errfix=True):

    errfix = '_errfix' if errfix else ''
    config  = os.path.join(run_dir,'calib','zphot_z01.para')
    paramf  = os.path.join(run_dir,'calib','zphot_out.para')
    cat_in  = os.path.join(run_dir,'calib','catalog%s_iso.in' %errfix)
    cat_out = os.path.join(run_dir,'calib','catalog%s_iso.out'%errfix)
    pdz_out = os.path.join(run_dir,'calib','catalog%s_iso'%errfix)

    call = '$LEPHAREDIR/source/zphota -c %s -CAT_IN %s -CAT_OUT %s -PDZ_OUT %s -PARA_OUT %s -AUTO_ADAPT YES' % (config,cat_in,cat_out,pdz_out,paramf)
    call+= get_errscale()
    useful.run(call,cwd=os.path.join(run_dir,'calib'),verbose=True)

def lephare_setup_no_hsc():

    os.system('$LEPHAREDIR/source/filter        -c lephare/calib/zphot_z01_no_hsc.para')
    os.system('$LEPHAREDIR/source/mag_star      -c lephare/calib/zphot_z01_no_hsc.para')
    os.system('$LEPHAREDIR/source/mag_gal  -t Q -c lephare/calib/zphot_z01_no_hsc.para -EB_V 0')
    os.system('$LEPHAREDIR/source/mag_gal  -t G -c lephare/calib/zphot_z01_no_hsc.para')

def lephare_calib_run_no_hsc(run_dir,aper_num=1,errfix=True):

    errfix = '_errfix' if errfix else ''
    config  = os.path.join(run_dir,'calib','zphot_z01_no_hsc.para')
    paramf  = os.path.join(run_dir,'calib','zphot_out.para')
    cat_in  = os.path.join(run_dir,'calib','catalog%s_r%i_no_hsc.in' %(errfix,useful.apersizes[aper_num]))
    cat_out = os.path.join(run_dir,'calib','catalog%s_r%i_no_hsc.out'%(errfix,useful.apersizes[aper_num]))
    pdz_out = os.path.join(run_dir,'calib','catalog%s_r%i_no_hsc'%(errfix,useful.apersizes[aper_num]))

    call = '$LEPHAREDIR/source/zphota -c %s -CAT_IN %s -CAT_OUT %s -PDZ_OUT %s -PARA_OUT %s -AUTO_ADAPT YES' % (config,cat_in,cat_out,pdz_out,paramf)
    call+= get_errscale()
    useful.run(call,cwd=os.path.join(run_dir,'calib'),verbose=True)

if __name__ == '__main__':

    run_dir = '/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/ref_test/lephare/'
    cat_dir = '/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/final_cats/'

    # mk_photoz_input_iso(cat_dir=cat_dir,run_dir=run_dir,errfix=True)
    # lephare_calib_run_iso(run_dir=run_dir,errfix=True)

    # lephare_setup_no_hsc()
    # mk_photoz_input_no_hsc(cat_dir=cat_dir,run_dir=run_dir,aper_num=1,errfix=True)
    # lephare_calib_run_no_hsc(run_dir=run_dir,aper_num=1,errfix=True)

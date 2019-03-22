import sys,os
import numpy as np
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt
import photutils
from collections import OrderedDict

import useful
from mk_cutouts import cutout_params
from add_other_catalogs import *
from gal_extinction import Gal_Extinction
from fix_errors import FixErrors
from mk_upper_lims import CalcUpperLimits
from mk_swarp_errors import errors_swarp

from mk_lephare import mk_combined_zphot, mk_final_value_added_cat
from mk_star_flag import add_sg_classification

def fix_individual_catalogs(cat_dir):

    with open('final_cats/DIM_missed.reg','w') as _: pass
    with open('final_cats/DIM_extra.reg' ,'w') as _: pass

    with open('final_cats/DIM_missed.reg','a') as missed_fhndl:

        with open('final_cats/DIM_extra.reg','a') as extra_fhndl:

            for i in range(4):

                missed_idx, extra_idx = np.zeros(0), np.zeros(0)
                seg_cat = fitsio.getdata('data/catalog%i_det.fits'%(i+1))

                for instr in useful.instr_used_list[:-1]:

                    for filt in useful.filters[instr]:

                        filt_cat = fitsio.getdata(os.path.join(cat_dir,'catalog%i_conv_%s_%s.fits'%(i+1,instr,filt)))

                        idx = useful.match_x_y(seg_cat['X_IMAGE'],seg_cat['Y_IMAGE'],filt_cat['X_IMAGE'],filt_cat['Y_IMAGE'],r=0.5)
                        cond = (idx != len(filt_cat))
                        assert len(np.unique(idx[cond])) == len(idx[cond])

                        print( "Matching FILT_CAT w/ SEG_MAP: %6s-%2s-%i ... %i / %i (%i in FILT_CAT)" % (instr,filt,i+1,len(idx[cond]),len(seg_cat),len(filt_cat)))
                        if len(idx[~cond])>0: missed_idx = np.append(missed_idx,np.where(idx == len(filt_cat))[0])
                        _extra_idx = sorted(set(np.arange(len(filt_cat))).difference(idx[cond]))
                        if len(_extra_idx)>0: extra_idx = np.append(extra_idx,_extra_idx)

                        _filt_cat = np.recarray(len(seg_cat),dtype=filt_cat.dtype)
                        for x in _filt_cat.dtype.names:
                            _filt_cat[x] = -99.
                            _filt_cat[x][cond] = filt_cat[x][idx[cond]]
                        _filt_cat['NUMBER'][cond] = seg_cat['NUMBER'][cond]

                        assert np.all(_filt_cat['X_IMAGE'][cond] == seg_cat['X_IMAGE'][cond])
                        assert np.all(_filt_cat['Y_IMAGE'][cond] == seg_cat['Y_IMAGE'][cond])
                        assert np.all(_filt_cat['A_IMAGE'][cond] == seg_cat['A_IMAGE'][cond])
                        assert np.all(_filt_cat['B_IMAGE'][cond] == seg_cat['B_IMAGE'][cond])

                        fitsio.writeto(os.path.join(cat_dir,'catalog%i_matched_%s_%s.fits'%(i+1,instr,filt)),_filt_cat,overwrite=True)

                missed_idx = list(np.unique(missed_idx).astype(int))
                if len(missed_idx)>0:
                    np.savetxt(missed_fhndl,np.vstack((seg_cat[missed_idx]['X_WORLD'],
                                                       seg_cat[missed_idx]['Y_WORLD'],
                                                       np.ones(len(missed_idx))*(i+1),
                                                       seg_cat[missed_idx]['NUMBER'])).T,
                                fmt='circle(%.8f,%.8f,1.5") #color=red text={%i-%i}',header="fk5",comments="")

                extra_idx  = list(np.unique(extra_idx).astype(int))
                if len(extra_idx)>0:
                    np.savetxt(extra_fhndl,np.vstack((filt_cat[extra_idx]['X_WORLD'],
                                                      filt_cat[extra_idx]['Y_WORLD'],
                                                      np.ones(len(extra_idx))*(i+1),
                                                      filt_cat[extra_idx]['NUMBER'])).T,
                                fmt='circle(%.8f,%.8f,1.5") #color=green text={%i-%i}',header="fk5",comments="")

                print( "Missing: ", missed_idx)
                print( "Extra: ", extra_idx)

def get_dtypes():

    dtype1 = [('ID','>i4'),
              ('RA','>f8'),('DEC','>f8'),
              ('A','>f4'),('B','>f4'),('THETA','>f4'),
              ('X_IMAGE','>f4'),('Y_IMAGE','>f4'),
              ('A_IMAGE','>f4'),('B_IMAGE','>f4'),('THETA_IMAGE','>f4'),
              ('KRON_RADIUS','>f4'),('PETRO_RADIUS','>f4'),('ISOAREAF_IMAGE','>i4'),
              ('ELONGATION','>f4'),('ELLIPTICITY','>f4'),
              ('GAL_EXT_EBV','>f4'),("OFFSET_FLUX",'>f4',(5,)),("OFFSET_MAG",'>f4',(5,)),
              ('ZSPEC','>f4'),('ZSPEC_REF','<U20'),('USE_ZSPEC_FLAG','>i4'),('ZPHOT','>f4'),('STAR_FLAG','>i4')]

    _dtype2 = [('MAG_AUTO', '>f4'),     ('MAGERR_AUTO', '>f4'),
               ('FLUX_AUTO','>f4'),     ('FLUXERR_AUTO','>f4'),
               ('MAG_ISO',  '>f4'),     ('MAGERR_ISO',  '>f4'),
               ('FLUX_ISO', '>f4'),     ('FLUXERR_ISO', '>f4'),
               ('MAG_APER', '>f4',(5,)),('MAGERR_APER', '>f4',(5,)),
               ('FLUX_APER','>f4',(5,)),('FLUXERR_APER','>f4',(5,)),
               ('FLUX_RADIUS','>f4',(3,)),('AVG_WHT','>f4'),
               ('SE_FLAGS','>i4'),('COVERAGE_FLAG','>i4')]

    dtype2 = []
    for instr in useful.instr_used_list[:-1]:
        for filt in useful.filters[instr]:
            for x in [list(_) for _ in _dtype2]:
                x[0] += "_%s_%s" % (instr,filt)
                dtype2.append(x)
    dtype2 = [tuple(_) for _ in dtype2]

    _dtype3 = [("FLUX_TOT",float),("FLUXERR_TOT",float),("MAG_TOT",float),("MAGERR_TOT",float),("SE_FLAGS",int),('COVERAGE_FLAG','>i4')]

    instr = "irac"
    dtype3 = []
    for filt in useful.filters[instr]:
        for x in [list(_) for _ in _dtype3]:
            x[0] += "_%s_%s" % (instr,filt)
            dtype3.append(x)
    dtype3 = [tuple(_) for _ in dtype3]

    dtype4 = [('cutout_num','>i4'),('cutout_id','>i4'),('cutout_x','>f4'),('cutout_y','>f4')]

    return dtype1+dtype2+dtype3+dtype4+add_catalogs_dtype

def remove_overlapping_sources(catalog):

    sys.stdout.write("Removing overlapping sources ... ")
    sys.stdout.flush()

    cond1 = (catalog['cutout_num']==1) & ((catalog['X_IMAGE'] >= 25000) | (catalog['Y_IMAGE'] >= 25000))
    cond2 = (catalog['cutout_num']==2) & ((catalog['X_IMAGE'] >= 25000) | (catalog['Y_IMAGE'] <  25000))
    cond3 = (catalog['cutout_num']==3) & ((catalog['X_IMAGE'] <  25000) | (catalog['Y_IMAGE'] >= 25000))
    cond4 = (catalog['cutout_num']==4) & ((catalog['X_IMAGE'] <  25000) | (catalog['Y_IMAGE'] <  25000))

    cond = cond1 | cond2 | cond3 | cond4
    print( "%i removed." % len(catalog[cond]))
    return catalog[~cond]

def remove_missing_SEx_sources(catalog):

    sys.stdout.write("Removing SEx sources missing from DIM ... ")
    sys.stdout.flush()

    cond = {}
    gcond = np.zeros(len(catalog),dtype=bool)

    for x in catalog.dtype.names:
        if "COVERAGE_FLAG_" in x and "irac" not in x:
            cond[x] = (catalog[x]==-99)
            gcond = gcond | cond[x]

    for x in cond: assert not np.any(gcond ^ cond[x])
    print( "%i removed." % len(catalog[gcond]))
    return catalog[~gcond]

def fix_sextractor_99s(catalog):

    sys.stdout.write("Fixing SExtractor 99s ... ")
    sys.stdout.flush()

    for x in catalog.dtype.names:
        if 'MAG_' in x or 'MAGERR_' in x:
            cond = (catalog[x] == 99.)
            catalog[x][cond] = -99.
    print( "done.")
    return catalog

def fix_negative_isoarea(catalog):

    for i in range(4):

        sys.stdout.write("\rFixing ISOAREA<=0 for %s:%s-%i ... \033[K" % (instr,filt,i+1))
        sys.stdout.flush()

        seg_map = fitsio.getdata('data/cutout%i_det.seg.fits'%(i+1))
        cond = (catalog['ISOAREAF_IMAGE']<=0)
        for entry in catalog[cond]:
            entry['ISOAREAF_IMAGE'] = len(seg_map[seg_map==entry['cutout_id']])

    print( "done")
    return catalog

def fix_sources_with_no_coverage_in_filter(catalog):

    sys.stdout.write("Fixing sources with no coverage ... ")
    sys.stdout.flush()

    for x in catalog.dtype.names:

        if 'COVERAGE_FLAG_' in x:

            instr,filt = x.split('_')[-2:]

            if instr!='irac':

                cond = (catalog[x] == 0)

                catalog[   'MAG_AUTO_%s_%s'%(instr,filt)][cond] = -99.
                catalog['MAGERR_AUTO_%s_%s'%(instr,filt)][cond] = -99.
                catalog[   'MAG_ISO_%s_%s' %(instr,filt)][cond] = -99.
                catalog['MAGERR_ISO_%s_%s' %(instr,filt)][cond] = -99.
                catalog[   'MAG_APER_%s_%s'%(instr,filt)][cond] = -99.
                catalog['MAGERR_APER_%s_%s'%(instr,filt)][cond] = -99.

                catalog[   'FLUX_AUTO_%s_%s'%(instr,filt)][cond] = -99.
                catalog['FLUXERR_AUTO_%s_%s'%(instr,filt)][cond] = -99.
                catalog[   'FLUX_ISO_%s_%s' %(instr,filt)][cond] = -99.
                catalog['FLUXERR_ISO_%s_%s' %(instr,filt)][cond] = -99.
                catalog[   'FLUX_APER_%s_%s'%(instr,filt)][cond] = -99.
                catalog['FLUXERR_APER_%s_%s'%(instr,filt)][cond] = -99.

    print( "done.")
    return catalog

def remove_sources_with_no_coverage_in_optnir(catalog):
    """
    Removes saturated stars outside the HSC coverage (mostly when only detected in CFHT-LS)
    Mainly prevents OFFSET_FLUX being set to -99.
    """
    sys.stdout.write("Removing sources with no coverage in Opt/NIR filters ... ")
    sys.stdout.flush()

    cond = np.zeros(len(catalog),dtype=bool)

    for fname in useful.fnames:
        if "irac" not in fname:
            cond = cond | (catalog["COVERAGE_FLAG_%s"%fname]==1)

    print( "%i removed." % len(catalog[~cond]))
    return catalog[cond]

def fix_galactic_extinction(catalog):

    gal_ext = Gal_Extinction()
    catalog['GAL_EXT_EBV'] = gal_ext.calc_EBV(catalog['RA'],catalog['DEC'])
    Av = gal_ext.calc_Av(ebv=catalog['GAL_EXT_EBV'])

    sys.stdout.write("Fixing fluxes/magnitudes for galactic extinction ... ")
    sys.stdout.flush()

    for fname in useful.fnames:

        instr,filt = fname.split('_')

        if instr=='irac':

            catalog['MAG_TOT_%s'%fname], catalog['FLUX_TOT_%s'%fname] = \
            gal_ext.remove_gal_ext(catalog['MAG_TOT_%s'%fname],
                                   catalog['FLUX_TOT_%s'%fname],
                                   instr=instr,filt=filt,Av=Av)

        else:

            catalog['MAG_AUTO_%s'%fname], catalog['FLUX_AUTO_%s'%fname] = \
                gal_ext.remove_gal_ext(catalog['MAG_AUTO_%s'%fname],
                                       catalog['FLUX_AUTO_%s'%fname],
                                       instr=instr,filt=filt,Av=Av)

            catalog['MAG_ISO_%s'%fname], catalog['FLUX_ISO_%s'%fname] = \
                gal_ext.remove_gal_ext(catalog['MAG_ISO_%s'%fname],
                                       catalog['FLUX_ISO_%s'%fname],
                                       instr=instr,filt=filt,Av=Av)

            for i in range(5):
                catalog['MAG_APER_%s'%fname][:,i], catalog['FLUX_APER_%s'%fname][:,i] = \
                    gal_ext.remove_gal_ext(catalog['MAG_APER_%s'%fname][:,i],
                                           catalog['FLUX_APER_%s'%fname][:,i],
                                           instr=instr,filt=filt,Av=Av)

    print( "done.")
    return catalog

def fix_zero_fluxerrors(catalog):

    sys.stdout.write("Fixing FLUXERR==0 objects ... ")
    sys.stdout.flush()

    for fname in useful.fnames:

        if "irac" not in fname:

            cond_auto = (catalog["FLUX_AUTO_%s"%fname]==0) & (catalog["FLUXERR_AUTO_%s"%fname]==0)
            catalog[    "MAG_AUTO_%s"%fname][cond_auto] = -99.
            catalog[ "MAGERR_AUTO_%s"%fname][cond_auto] = -99.
            catalog[   "FLUX_AUTO_%s"%fname][cond_auto] = -99.
            catalog["FLUXERR_AUTO_%s"%fname][cond_auto] = -99.

            cond_iso = (catalog["FLUX_ISO_%s"%fname]==0) & (catalog["FLUXERR_ISO_%s"%fname]==0)
            catalog[    "MAG_ISO_%s"%fname][cond_iso] = -99.
            catalog[ "MAGERR_ISO_%s"%fname][cond_iso] = -99.
            catalog[   "FLUX_ISO_%s"%fname][cond_iso] = -99.
            catalog["FLUXERR_ISO_%s"%fname][cond_iso] = -99.

            for i in range(5):

                cond_aper = (catalog["FLUX_APER_%s"%fname][:,i]==0) & (catalog["FLUXERR_APER_%s"%fname][:,i]==0)
                catalog[    "MAG_APER_%s"%fname][:,i][cond_aper] = -99.
                catalog[ "MAGERR_APER_%s"%fname][:,i][cond_aper] = -99.
                catalog[   "FLUX_APER_%s"%fname][:,i][cond_aper] = -99.
                catalog["FLUXERR_APER_%s"%fname][:,i][cond_aper] = -99.

        else:

            cond_tot = (catalog["FLUX_TOT_%s"%fname]==0) & (catalog["FLUXERR_TOT_%s"%fname]==0)
            catalog[    "MAG_TOT_%s"%fname][cond_tot] = -99.
            catalog[ "MAGERR_TOT_%s"%fname][cond_tot] = -99.
            catalog[   "FLUX_TOT_%s"%fname][cond_tot] = -99.
            catalog["FLUXERR_TOT_%s"%fname][cond_tot] = -99.

    print( "done.")
    return catalog

def fix_swarp_errors(catalog):

    for instr in useful.instr_used_list[:-1]:

        for filt in useful.filters[instr]:

            fname = "%s_%s" % (instr,filt)
            fix_factor = 1./errors_swarp[instr][filt]

            cond_fauto = (catalog['FLUXERR_AUTO_%s'%fname] != -99.)
            catalog['FLUXERR_AUTO_%s'%fname][cond_fauto] *= fix_factor

            cond_mauto = (catalog[ 'MAGERR_AUTO_%s'%fname] != -99.)
            catalog[ 'MAGERR_AUTO_%s'%fname][cond_mauto] *= fix_factor

            cond_fiso = (catalog['FLUXERR_ISO_%s'%fname] != -99.)
            catalog['FLUXERR_ISO_%s'%fname][cond_fiso] *= fix_factor

            cond_miso = (catalog[ 'MAGERR_ISO_%s'%fname] != -99.)
            catalog[ 'MAGERR_ISO_%s'%fname][cond_miso] *= fix_factor

            for i in range(5):

                cond_faper = (catalog['FLUXERR_APER_%s'%fname][:,i] != -99.)
                catalog['FLUXERR_APER_%s'%fname][:,i][cond_faper] *= fix_factor

                cond_maper = (catalog[ 'MAGERR_APER_%s'%fname][:,i] != -99.)
                catalog[ 'MAGERR_APER_%s'%fname][:,i][cond_maper] *= fix_factor

    return catalog

def calc_opt_nir_offset(catalog):

    fnames = []
    for instr in useful.instr_used_list[:-1]:
        for filt in useful.filters[instr]:
            fnames.append("_".join([instr,filt]))

    f_ratio,f_error = np.ma.zeros((2,len(catalog),len(fnames),5),dtype=float)
    f_ratio.mask,f_error.mask = np.zeros((2,len(catalog),len(fnames),5),dtype=bool)

    for aper_num in range(len(useful.apersizes)):

        for i,fname in enumerate(fnames):

            instr,filt = fname.split("_")
            sys.stdout.write("\rCalculating AUTO-APER offset for %s:%s-aper#%i ... \033[K" % (instr,filt,aper_num+1))
            sys.stdout.flush()

            # Just a simple check for if FLUX > 0
            cond_f = (catalog['FLUX_AUTO_%s'%(fname)] > 0.0) & (catalog['FLUX_APER_%s'%(fname)][:,aper_num] > 0.0)

            # Check if SN > 3 sigma
            # cond_f = (catalog['FLUX_AUTO_%s_%s'%(instr,filt)] / catalog['FLUXERR_AUTO_%s_%s'%(instr,filt)] > 3.0) & \
            #          (catalog['FLUX_APER_%s_%s'%(instr,filt)][:,aper_num] / catalog['FLUXERR_APER_%s_%s'%(instr,filt)][:,aper_num] > 3.0)

            f_ratio[cond_f,i,aper_num] = catalog[   'FLUX_AUTO_%s_%s'%(instr,filt)][cond_f]    / catalog[   'FLUX_APER_%s_%s'%(instr,filt)][cond_f,aper_num]
            f_error[cond_f,i,aper_num] = catalog['FLUXERR_AUTO_%s_%s'%(instr,filt)][cond_f]**2 + catalog['FLUXERR_APER_%s_%s'%(instr,filt)][cond_f,aper_num]**2
            f_ratio.mask[:,i,aper_num] = ~cond_f
            f_error.mask[:,i,aper_num] = ~cond_f

        ##########################################
        ### ADDED to fix the weird offset problem
        ##########################################
        # f_ratio_median = np.ma.median(f_ratio[:,:,aper_num],axis=-1)
        f_ratio_16ile,f_ratio_84ile = np.nanpercentile(f_ratio[:,:,aper_num].filled(np.NaN),[16,84],axis=-1)

        for i,fname in enumerate(fnames):
            cond = (f_ratio_16ile > f_ratio[:,i,aper_num]) | (f_ratio[:,i,aper_num] > f_ratio_84ile)
            f_ratio.mask[cond,i,aper_num] = True
            f_error.mask[cond,i,aper_num] = True
        ##########################################

    catalog["OFFSET_FLUX"] = np.ma.average(f_ratio,weights=1/f_error,axis=1).filled(-99.)
    catalog["OFFSET_FLUX"] = np.clip(catalog["OFFSET_FLUX"],1,np.inf)
    catalog[ "OFFSET_MAG"] = -2.5*np.log10(catalog["OFFSET_FLUX"])

    print("done!")
    return catalog

def calc_opt_nir_offset_OLD(catalog):

    for aper_num in range(len(useful.apersizes)):

        term1,term2 = np.zeros(len(catalog)),np.zeros(len(catalog))

        for instr in useful.instr_used_list[:-1]:

            for filt in useful.filters[instr]:

                sys.stdout.write("\rCalculating AUTO-APER offset for %s:%s-aper#%i ... \033[K" % (instr,filt,aper_num+1))
                sys.stdout.flush()

                # Just a simple check for if FLUX > 0
                cond_f = (catalog[   'FLUX_AUTO_%s_%s'%(instr,filt)] > 0.0) & (catalog[   'FLUX_APER_%s_%s'%(instr,filt)][:,aper_num] > 0.0)

                # Check if SN > 3 sigma
                # cond_f = (catalog['FLUX_AUTO_%s_%s'%(instr,filt)] / catalog['FLUXERR_AUTO_%s_%s'%(instr,filt)] > 3.0) & \
                #          (catalog['FLUX_APER_%s_%s'%(instr,filt)][:,aper_num] / catalog['FLUXERR_APER_%s_%s'%(instr,filt)][:,aper_num] > 3.0)

                diff_f,wht_f = np.zeros(len(catalog)),np.zeros(len(catalog))
                diff_f[cond_f] = catalog['FLUX_AUTO_%s_%s'%(instr,filt)][cond_f] / catalog['FLUX_APER_%s_%s'%(instr,filt)][:,aper_num][cond_f]
                wht_f[cond_f] = 1./(catalog['FLUXERR_AUTO_%s_%s'%(instr,filt)][cond_f]**2 + catalog['FLUXERR_APER_%s_%s'%(instr,filt)][:,aper_num][cond_f]**2)
                term1 += wht_f * diff_f
                term2 += wht_f

        cond = (term2>0)
        catalog["OFFSET_FLUX"][:,aper_num][cond] = term1[cond]/term2[cond]

        cond = (catalog["OFFSET_FLUX"][:,aper_num] > 0)
        catalog[ "OFFSET_MAG"][:,aper_num][cond] = -2.5*np.log10(catalog["OFFSET_FLUX"][:,aper_num][cond])

    print( "done.")
    return catalog

def add_avg_whts(catalog,new=False):

    if new:

        radius = 3. / 2. / useful.pix_scale # Compute the average weights for just 3" aperture

        dtype = [('ID','>i4'),('X_IMAGE','>f4'),('Y_IMAGE','>f4')]
        for instr in useful.instr_used_list[:-1]:
            for filt in useful.filters[instr]:
                dtype.extend([('AVG_WHT_%s_%s'%(instr,filt),'>f4'),])

        cat_whts = np.recarray(len(catalog),dtype=dtype)
        for x in cat_whts.dtype.names: cat_whts[x] = -99.

        cat_whts['ID'] = catalog['ID']
        cat_whts['X_IMAGE'] = catalog['X_IMAGE']
        cat_whts['Y_IMAGE'] = catalog['Y_IMAGE']
        pos = zip(cat_whts['X_IMAGE'],cat_whts['Y_IMAGE'])

        for instr in useful.instr_used_list[:-1]:
            for filt in useful.filters[instr]:

                sys.stdout.write("\rCalculating new average weights for %s:%s ... \033[K"%(instr,filt))
                sys.stdout.flush()

                wht  = fitsio.getdata(os.path.join(cwd,'data','orig','mosaic_%s_%s.wht.fits'%(instr,filt)))
                aperture = photutils.CircularAperture(pos,r=radius)
                avg_wht  = photutils.aperture_photometry(wht, aperture)['aperture_sum'] / (np.pi*radius**2)
                cat_whts["AVG_WHT_%s_%s"%(instr,filt)] = avg_wht

        fitsio.writeto("final_cats/catalog_avg_whts.fits",cat_whts,overwrite=True)
        print( "done.")

    cat_whts = fitsio.getdata("final_cats/catalog_avg_whts.fits")
    print( "Adding average weight information to catalog ... ",)

    assert np.all((cat_whts["X_IMAGE"]==catalog["X_IMAGE"]) & (cat_whts["Y_IMAGE"]==catalog["Y_IMAGE"]))
    for instr in useful.instr_used_list[:-1]:
        for filt in useful.filters[instr]:
            catalog['AVG_WHT_%s_%s'%(instr,filt)] = cat_whts['AVG_WHT_%s_%s'%(instr,filt)]
            cond = (catalog["COVERAGE_FLAG_%s_%s"%(instr,filt)]==0)
            catalog['AVG_WHT_%s_%s'%(instr,filt)][cond] = 0

    print( "done.")
    return catalog

def mk_final_cat(cat_dir,new_avg_whts=False):

    j = [0,] + [len(fitsio.getdata(os.path.join(cwd,'data','catalog%i_det.fits' % (i+1)))) for i in range(4)]
    j = np.cumsum(j)

    catalog = np.recarray((j[-1],),dtype=get_dtypes())
    for x in catalog.dtype.names: catalog[x] = -99.
    catalog["ZSPEC_REF"] = "--"

    for i in range(4):

        x0,x1,y0,y1 = cutout_verts[i]

        cat_name = os.path.join(cwd,'data','catalog%i_det.fits' % (i+1))
        sex_cat = fitsio.getdata(cat_name)

        catalog[j[i]:j[i+1]]['cutout_num'] = i+1
        catalog[j[i]:j[i+1]]['cutout_id']  = sex_cat['NUMBER']
        catalog[j[i]:j[i+1]]['cutout_x']   = sex_cat['X_IMAGE']
        catalog[j[i]:j[i+1]]['cutout_y']   = sex_cat['Y_IMAGE']

        catalog[j[i]:j[i+1]]['X_IMAGE']        = catalog[j[i]:j[i+1]]['cutout_x']+x0
        catalog[j[i]:j[i+1]]['Y_IMAGE']        = catalog[j[i]:j[i+1]]['cutout_y']+y0
        catalog[j[i]:j[i+1]]['A_IMAGE']        = sex_cat['A_IMAGE']
        catalog[j[i]:j[i+1]]['B_IMAGE']        = sex_cat['B_IMAGE']
        catalog[j[i]:j[i+1]]['THETA_IMAGE']    = sex_cat['THETA_IMAGE']
        catalog[j[i]:j[i+1]]['ELONGATION']     = sex_cat['ELONGATION']
        catalog[j[i]:j[i+1]]['ELLIPTICITY']    = sex_cat['ELLIPTICITY']
        catalog[j[i]:j[i+1]]['KRON_RADIUS']    = sex_cat['KRON_RADIUS']
        catalog[j[i]:j[i+1]]['PETRO_RADIUS']   = sex_cat['PETRO_RADIUS']
        catalog[j[i]:j[i+1]]['ISOAREAF_IMAGE'] = sex_cat['ISOAREAF_IMAGE']

        catalog[j[i]:j[i+1]]['RA']    = sex_cat['X_WORLD']
        catalog[j[i]:j[i+1]]['DEC']   = sex_cat['Y_WORLD']
        catalog[j[i]:j[i+1]]['A']     = sex_cat['A_WORLD']
        catalog[j[i]:j[i+1]]['B']     = sex_cat['B_WORLD']
        catalog[j[i]:j[i+1]]['THETA'] = sex_cat['THETA_WORLD']

        for instr in useful.instr_used_list:

            for filt in useful.filters[instr]:

                sys.stdout.write("\rProcessing cutout#%i - %s:%s ... \033[K" % (i+1,instr,filt))
                sys.stdout.flush()
                cat_name = 'catalog%i_matched_%s_%s.fits' % (i+1,instr,filt)
                sex_cat = fitsio.getdata(os.path.join(cat_dir,cat_name))

                if instr=='irac':

                    catalog[j[i]:j[i+1]]['MAG_TOT_%s_%s'      %(instr,filt)] = sex_cat['MAG']
                    catalog[j[i]:j[i+1]]['MAGERR_TOT_%s_%s'   %(instr,filt)] = sex_cat['MAGERR']
                    catalog[j[i]:j[i+1]]['FLUX_TOT_%s_%s'     %(instr,filt)] = sex_cat['FLUX']
                    catalog[j[i]:j[i+1]]['FLUXERR_TOT_%s_%s'  %(instr,filt)] = sex_cat['FLUXERR']
                    catalog[j[i]:j[i+1]]['SE_FLAGS_%s_%s'     %(instr,filt)] = sex_cat['FLAGS']
                    catalog[j[i]:j[i+1]]['COVERAGE_FLAG_%s_%s'%(instr,filt)] = sex_cat['COVERAGE_FLAG']

                else:

                    catalog[j[i]:j[i+1]]['MAG_AUTO_%s_%s'     %(instr,filt)] = sex_cat['MAG_AUTO']
                    catalog[j[i]:j[i+1]]['MAG_ISO_%s_%s'      %(instr,filt)] = sex_cat['MAG_ISO']
                    catalog[j[i]:j[i+1]]['MAG_APER_%s_%s'     %(instr,filt)] = sex_cat['MAG_APER']
                    catalog[j[i]:j[i+1]]['MAGERR_AUTO_%s_%s'  %(instr,filt)] = sex_cat['MAGERR_AUTO']
                    catalog[j[i]:j[i+1]]['MAGERR_ISO_%s_%s'   %(instr,filt)] = sex_cat['MAGERR_ISO']
                    catalog[j[i]:j[i+1]]['MAGERR_APER_%s_%s'  %(instr,filt)] = sex_cat['MAGERR_APER']

                    catalog[j[i]:j[i+1]]['FLUX_AUTO_%s_%s'    %(instr,filt)] = sex_cat['FLUX_AUTO']
                    catalog[j[i]:j[i+1]]['FLUX_ISO_%s_%s'     %(instr,filt)] = sex_cat['FLUX_ISO']
                    catalog[j[i]:j[i+1]]['FLUX_APER_%s_%s'    %(instr,filt)] = sex_cat['FLUX_APER']
                    catalog[j[i]:j[i+1]]['FLUXERR_AUTO_%s_%s' %(instr,filt)] = sex_cat['FLUXERR_AUTO']
                    catalog[j[i]:j[i+1]]['FLUXERR_ISO_%s_%s'  %(instr,filt)] = sex_cat['FLUXERR_ISO']
                    catalog[j[i]:j[i+1]]['FLUXERR_APER_%s_%s' %(instr,filt)] = sex_cat['FLUXERR_APER']

                    catalog[j[i]:j[i+1]]['FLUX_RADIUS_%s_%s'  %(instr,filt)] = sex_cat['FLUX_RADIUS']
                    # catalog[j[i]:j[i+1]]['FWHM_IMAGE_%s_%s'   %(instr,filt)] = sex_cat['FWHM_IMAGE']
                    # catalog[j[i]:j[i+1]]['ISOAREA_IMAGE_%s_%s'%(instr,filt)] = sex_cat['ISOAREA_IMAGE']
                    catalog[j[i]:j[i+1]]['SE_FLAGS_%s_%s'     %(instr,filt)] = sex_cat['FLAGS']

                    assert np.all([_ in [-99,32768,33024] for _ in np.unique(sex_cat['IMAFLAGS_ISO'])])
                    sex_cat['IMAFLAGS_ISO'][sex_cat['IMAFLAGS_ISO'] == 32768] = 1   # has coverage
                    sex_cat['IMAFLAGS_ISO'][sex_cat['IMAFLAGS_ISO'] == 33024] = 0   # no coverage
                    catalog[j[i]:j[i+1]]['COVERAGE_FLAG_%s_%s'%(instr,filt)] = sex_cat['IMAFLAGS_ISO']

    print( "done!")

    catalog['KRON_RADIUS'][ catalog['KRON_RADIUS']  == 0] = 3.5
    catalog['PETRO_RADIUS'][catalog['PETRO_RADIUS'] == 0] = 3.5

    catalog = remove_overlapping_sources(catalog)
    catalog = remove_missing_SEx_sources(catalog)
    catalog = fix_sextractor_99s(catalog)
    catalog = fix_sources_with_no_coverage_in_filter(catalog)
    catalog = remove_sources_with_no_coverage_in_optnir(catalog)
    catalog = fix_zero_fluxerrors(catalog)
    catalog = fix_swarp_errors(catalog)
    catalog = fix_galactic_extinction(catalog)

    catalog = calc_opt_nir_offset(catalog)
    catalog = match_radio(catalog)
    catalog = match_xray(catalog)

    catalog['ID'] = np.arange(len(catalog)) + 1
    catalog = add_avg_whts(catalog,new=new_avg_whts)

    print( "Final Catalog: %i sources" % len(catalog))

    regions = np.recarray(len(catalog),dtype=[('x',float),('y',float),('r',float),('c','<U8')])
    regions['x'] = catalog['X_IMAGE']
    regions['y'] = catalog['Y_IMAGE']
    regions['r'] = 8+(catalog['cutout_num'])
    regions['c'][catalog['cutout_num']==1] = 'green'
    regions['c'][catalog['cutout_num']==2] = 'red'
    regions['c'][catalog['cutout_num']==3] = 'cyan'
    regions['c'][catalog['cutout_num']==4] = 'magenta'
    np.savetxt(os.path.join(cat_dir,'final_catalog.reg'),regions,fmt='circle(%10.2f,%10.2f,%i) # width=2 color=%s')

    fitsio.writeto(os.path.join(cat_dir,'final_catalog.fits'),catalog,overwrite=True)

def add_specz(catalog):

    #'M15_GRISMz' -- removed Morris+15
    for i,zlabel in enumerate(['XUDS_SPECz','C3R2_SPECz','VIPERS_SPECz','SUBARU_SPECz','UDSz_SPECz']):

        cond = (catalog[zlabel]!=-99.) & (catalog[zlabel]>0)

        if "VIPERS" in zlabel:
            cond = cond & ((catalog['VIPERS_FLAG'].astype(int)==4) | (catalog['VIPERS_FLAG'].astype(int)==3))
        if "C3R2" in zlabel:
            cond = cond & ((catalog['C3R2_FLAG'].astype(int)==4) | (catalog['C3R2_FLAG'].astype(int)==3))
        # if "M15" in zlabel:
        #     cond = cond & (catalog['M15_QUALITY'] > 2.)

        catalog['ZSPEC'][cond] = catalog[zlabel][cond]
        catalog['ZSPEC_REF'][cond] = zlabel.split("_")[0]

        if "SUBARU" in zlabel:
            catalog['ZSPEC_REF'][cond] = np.array([useful.SUBARU_key[x] for x in catalog["SUBARU_REF"][cond]])

        if "XUDS" in zlabel:
            catalog['ZSPEC_REF'][cond] = np.array([useful.XUDS_key[x] for x in catalog["XUDS_REF"][cond]])

    cond = (catalog['ZSPEC']!=-99.)
    cond_use = cond & \
               (catalog['COVERAGE_FLAG_hsc_g']==1) & \
               (catalog['COVERAGE_FLAG_hsc_r']==1) & \
               (catalog['COVERAGE_FLAG_hsc_i']==1) & \
               (catalog['COVERAGE_FLAG_hsc_z']==1) & \
               (catalog['COVERAGE_FLAG_hsc_y']==1)

    catalog["USE_ZSPEC_FLAG"][ cond_use] = 1
    catalog["USE_ZSPEC_FLAG"][~cond_use] = 0

    print( "Spec-z found for %i out of %i objects (%i usable)" % (np.sum(cond),len(catalog),np.sum(cond_use)))

    return catalog

def mk_final_cat_ex(cat_dir):

    catalog = fitsio.getdata(os.path.join(cat_dir,'final_catalog.fits'))

    print( "Matching other catalogs ...")

    catalog_ex = np.recarray((len(catalog),),dtype=get_dtypes()+oth_catalogs_dtype)
    for x in catalog_ex.dtype.names:
        try:
            catalog_ex[x] = catalog[x]
        except KeyError:
            catalog_ex[x] = -99

    for match_fn in [match_ps1,
                     match_uds,
                     match_uds_s2cls,
                     match_furusawa08,
                     match_video,
                     match_udsz,
                     match_xuds_compile,
                     match_subaru_compile,
                     match_morris14,
                     match_c3r2,
                     match_vipers,
                     match_3dhst,
                     match_ned]:

        catalog_ex = match_fn(catalog_ex)

    catalog_ex = add_specz(catalog_ex)
    catalog["ZSPEC"]          = catalog_ex["ZSPEC"]
    catalog["ZSPEC_REF"]      = catalog_ex["ZSPEC_REF"]
    catalog["USE_ZSPEC_FLAG"] = catalog_ex["USE_ZSPEC_FLAG"]

    fitsio.writeto(os.path.join(cat_dir,'final_catalog.fits'),catalog,overwrite=True)
    fitsio.writeto(os.path.join(cat_dir,'final_catalog.extra.fits'),catalog_ex,overwrite=True)

def fix_errors(catalog):

    fix_err = FixErrors(catalog)

    for fname in fix_err.fnames:

        instr,filt = fname.split('_')
        sys.stdout.write("\rFix errors for %s ... \033[K"%fname)
        sys.stdout.flush()

        cond_fauto = (catalog['FLUXERR_AUTO_%s'%fname] != -99.)
        catalog['FLUXERR_AUTO_%s'%fname][cond_fauto] *= fix_err(aper=fix_err.npix_auto[cond_fauto],
                                                                wht=catalog["AVG_WHT_%s"%fname][cond_fauto],
                                                                instr=instr,filt=filt)
        cond_mauto = (catalog[ 'MAGERR_AUTO_%s'%fname] != -99.)
        catalog[ 'MAGERR_AUTO_%s'%fname][cond_mauto] *= fix_err(aper=fix_err.npix_auto[cond_mauto],
                                                                wht=catalog["AVG_WHT_%s"%fname][cond_mauto],
                                                                instr=instr,filt=filt)

        cond_fiso  = (catalog['FLUXERR_ISO_%s'%fname] != -99.)
        catalog[ 'FLUXERR_ISO_%s'%fname][cond_fiso]  *= fix_err(aper=fix_err.npix_iso[cond_fiso],
                                                                wht=catalog["AVG_WHT_%s"%fname][cond_fiso],
                                                                instr=instr,filt=filt)
        cond_miso  = (catalog[ 'MAGERR_ISO_%s'%fname] != -99.)
        catalog[  'MAGERR_ISO_%s'%fname][cond_miso]  *= fix_err(aper=fix_err.npix_iso[cond_miso],
                                                                wht=catalog["AVG_WHT_%s"%fname][cond_miso],
                                                                instr=instr,filt=filt)

        for i in range(5):

            cond_faper = (catalog['FLUXERR_APER_%s'%fname][:,i] != -99.)
            catalog['FLUXERR_APER_%s'%fname][:,i][cond_faper] *= fix_err(aper=fix_err.npix_aper[i],
                                                                         wht=catalog["AVG_WHT_%s"%fname][cond_faper],
                                                                         instr=instr,filt=filt,fixed_aper=True)
            cond_maper = (catalog[ 'MAGERR_APER_%s'%fname][:,i] != -99.)
            catalog[ 'MAGERR_APER_%s'%fname][:,i][cond_maper] *= fix_err(aper=fix_err.npix_aper[i],
                                                                         wht=catalog["AVG_WHT_%s"%fname][cond_maper],
                                                                         instr=instr,filt=filt,fixed_aper=True)

    print()
    return catalog

def get_upper_lims(catalog):

    up_lim = CalcUpperLimits()

    for fname in up_lim.fnames:

        instr,filt = fname.split('_')
        sys.stdout.write("\rGet upper limits for %s ... \033[K"%fname)
        sys.stdout.flush()

        for aper_num in range(5):

            cond_flag = (catalog["COVERAGE_FLAG_%s"%fname]==1)

            mlim_aper = np.zeros(len(catalog),dtype=float)
            mlim_aper[cond_flag] = up_lim(aper=up_lim.apersizes[aper_num],
                                          wht=catalog["AVG_WHT_%s"%fname][cond_flag],
                                          instr=instr,filt=filt)
            # mlim_aper[cond_flag] = mlim_aper[cond_flag] - 2.5*np.log10(3.)

            cond_mag  = cond_flag & \
                        ((catalog["MAG_APER_%s"%fname][:,aper_num] >  mlim_aper) | \
                         (catalog["MAG_APER_%s"%fname][:,aper_num] == -99.))

            catalog[   "MAG_APER_%s"%fname][:,aper_num][cond_mag] = mlim_aper[cond_mag]
            catalog["MAGERR_APER_%s"%fname][:,aper_num][cond_mag] = -1.0

    print()
    return catalog

def fix_catalog_errors(cat_dir='final_cats/'):

    catalog    = fitsio.getdata(os.path.join(cat_dir,"final_catalog.fits"))
    catalog_ex = fitsio.getdata(os.path.join(cat_dir,"final_catalog.extra.fits"))

    catalog = fix_errors(catalog)           # Fix errors
    catalog = calc_opt_nir_offset(catalog)  # Recalculate the Opt-NIR offset
    catalog = get_upper_lims(catalog)       # Fix the upper limits

    for x in catalog.dtype.names: catalog_ex[x] = catalog[x]

    fitsio.writeto(os.path.join(cat_dir,'final_catalog_errfix.fits'),catalog,overwrite=True)
    fitsio.writeto(os.path.join(cat_dir,'final_catalog_errfix.extra.fits'),catalog_ex,overwrite=True)

def obscure_prop_zspec(catalog):

    reflabels = [x for x in np.unique(catalog["ZSPEC_REF"]) if "_in_prep" in x]

    cond = np.zeros(len(catalog),dtype=bool)
    for ref in reflabels:
        _cond = np.array([ref in x for x in catalog["ZSPEC_REF"]],dtype=bool)
        print( "Obscured %i \"%s\" spec-z's" % (np.sum(_cond),ref))
        cond = cond & _cond

    catalog[cond]["ZSPEC"] = -1.0
    return catalog

def trim_columns(catalog):

    remove = [x for x in catalog.dtype.names if "AVG_WHT" in x] + \
             ["cutout_num", "cutout_id", "cutout_x", "cutout_y","USE_ZSPEC_FLAG"]
    keep   = [x for x in catalog.dtype.names if x not in remove]
    catalog = useful.view_fields(catalog,keep)
    return catalog

def trim_lephare_columns(catalog):

    dtype_cat = [x for x in catalog.dtype.descr if ("LPH_" not in x[0]) and (""!=x[0])]

    lph_cols = [("Z_MED"       ,"LPH_Z_ML"),
                ("Z_MED_L68"   ,"LPH_Z_ML68_LOW"),
                ("Z_MED_U68"   ,"LPH_Z_ML68_HIGH"),
                ("Z_BEST"      ,"LPH_Z_BEST"),
                ("Z_BEST_L68"  ,"LPH_Z_BEST68_LOW"),
                ("Z_BEST_U68"  ,"LPH_Z_BEST68_HIGH"),
                ("CHI_BEST"    ,"LPH_CHI_BEST"),
                ("PDZ_BEST"    ,"LPH_PDZ_BEST"),
                ("Z_SEC"       ,"LPH_Z_SEC"),
                ("CHI_SEC"     ,"LPH_CHI_SEC"),
                ("Z_QSO"       ,"LPH_Z_QSO"),
                ("CHI_QSO"     ,"LPH_CHI_QSO"),
                ("CHI_STAR"    ,"LPH_CHI_STAR"),
                ("NBAND_Z"     ,"LPH_NBAND_USED"),
                ("EBV_BEST"    ,"LPH_EBV_BEST_PHYS"),
                ("EXTLAW_BEST" ,"LPH_EXTLAW_BEST_PHYS"),
                ("MASS_BEST"   ,"LPH_MASS_BEST"),
                ("AGE_BEST"    ,"LPH_AGE_BEST"),
                ("SFR_BEST"    ,"LPH_SFR_BEST"),
                ("SSFR_BEST"   ,"LPH_SSFR_BEST"),
                ("CHI_PHYS"    ,"LPH_CHI_BEST_PHYS"),
                ("LUM_NUV_BEST","LPH_LUM_NUV_BEST"),
                ("LUM_R_BEST"  ,"LPH_LUM_R_BEST"),
                ("LUM_K_BEST"  ,"LPH_LUM_K_BEST"),
                ("MAG_ABS"     ,"LPH_MAG_ABS"),
                ("NBAND_PHYS"  ,"LPH_NBAND_USED_PHYS")]
    lph_cols = OrderedDict(lph_cols)

    dtype_lph = []
    for x in lph_cols:
        fmt = [y[1:] for y in catalog.dtype.descr if lph_cols[x]==y[0]][0]
        if   len(fmt)==1: dtype_lph = dtype_lph + [(x,fmt[0])]
        elif len(fmt)==2: dtype_lph = dtype_lph + [(x,fmt[0],fmt[1])]
        else: raise Expection("Invalid dtype.")

    new_catalog = np.recarray(len(catalog),dtype=dtype_cat+dtype_lph)

    for x in dtype_cat: new_catalog[x[0]] = catalog[x[0]]
    for x in dtype_lph: new_catalog[x[0]] = catalog[lph_cols[x[0]]]

    return new_catalog

def replace_cfht_with_musubi(catalog):

    dtype = []
    for x in catalog.dtype.descr:
        name, fmt = x[0],x[1:]
        name = name.replace("cfht_u","musubi_u")
        if   len(fmt)==1: dtype = dtype + [(name,fmt[0])]
        elif len(fmt)==2: dtype = dtype + [(name,fmt[0],fmt[1])]
        else: raise Expection("Invalid dtype.")

    _catalog = np.recarray(len(catalog),dtype=dtype)
    for x in catalog.dtype.names:
        _catalog[x.replace("cfht_u","musubi_u")] = catalog[x]

    return _catalog

def mk_publish_cat(version,cat_dir='final_cats/'):

    fname   = "final_catalog_errfix_zphot.fits"
    outname = "sxds_catalog_%s.fits"%version

    catalog = fitsio.getdata(os.path.join(cat_dir,fname))

    catalog["ZPHOT"] = catalog["LPH_Z_ML"]
    cond = (catalog["LPH_Z_ML"]==-99.)
    catalog["ZPHOT"][cond] = catalog["LPH_Z_BEST"][cond]

    catalog = add_sg_classification(catalog)
    catalog = obscure_prop_zspec(catalog)
    catalog = trim_columns(catalog)
    catalog = trim_lephare_columns(catalog)
    catalog = replace_cfht_with_musubi(catalog)

    fitsio.writeto(os.path.join(cat_dir,outname),catalog,overwrite=True)

if __name__ == '__main__':

    cwd = '/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/'
    cutout_verts = cutout_params()

    cat_dir = os.path.join(cwd,'final_cats')
    # fix_individual_catalogs(cat_dir=cat_dir)

    # mk_final_cat(cat_dir=cat_dir,new_avg_whts=False)
    # mk_final_cat_ex(cat_dir=cat_dir)

    # fix_catalog_errors(cat_dir=cat_dir)

    # AFTER LePhare
    mk_combined_zphot()
    mk_final_value_added_cat(cat_dir=cat_dir)

    # Finally
    mk_publish_cat(version="v1.6",cat_dir=cat_dir)

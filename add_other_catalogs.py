import astropy.io.fits as fitsio
import numpy as np

from useful import match_ra_dec

add_catalogs_dtype = [('1_4GHz_ID','>i4'),('1_4GHz_RA','>f8'),('1_4GHz_DEC','>f8'),
                      ('1_4GHz_FLUX','>f4'),('1_4GHz_FLUXERR','>f4'),
                      ('1_4GHz_Rel','>f4'),('1_4GHz_n_Rel','<U2'),

                      ('Xray_ID','<U8'),
                      ('Xray_RA','>f8'),('Xray_DEC','>f8'),
                      ('Xray_zUSE','>f4'),('Xray_HR2','>f4'),
                      ('Xray_logNH','>f4'),('Xray_logLHX','>f4')]

oth_catalogs_dtype = [('PS1_ID','>i4'),('PS1_RA','>f8'),('PS1_RAERR','>f8'),('PS1_DEC','>f8'),('PS1_DECERR','>f8'),
                      ('PS1_MAG_g','>f4'),('PS1_MAGERR_g','>f4'),
                      ('PS1_MAG_r','>f4'),('PS1_MAGERR_r','>f4'),
                      ('PS1_MAG_i','>f4'),('PS1_MAGERR_i','>f4'),
                      ('PS1_MAG_z','>f4'),('PS1_MAGERR_z','>f4'),
                      ('PS1_MAG_y','>f4'),('PS1_MAGERR_y','>f4'),

                      ('UDS_ID','>i4'),('UDS_RA','>f8'),('UDS_DEC','>f8'),
                      ('UDS_z','>f4'),('UDS_zerr','>f4'),('UDS_ztype','>i4'),
                      ('UDS_MAG_NUV','>f4'),('UDS_MAGERR_NUV','>f4'),
                      ('UDS_MAG_u','>f4'),('UDS_MAGERR_u','>f4'),
                      ('UDS_MAG_b','>f4'),('UDS_MAGERR_b','>f4'),
                      ('UDS_MAG_v','>f4'),('UDS_MAGERR_v','>f4'),
                      ('UDS_MAG_r','>f4'),('UDS_MAGERR_r','>f4'),
                      ('UDS_MAG_i','>f4'),('UDS_MAGERR_i','>f4'),
                      ('UDS_MAG_z','>f4'),('UDS_MAGERR_z','>f4'),
                      ('UDS_MAG_j','>f4'),('UDS_MAGERR_j','>f4'),
                      ('UDS_MAG_h','>f4'),('UDS_MAGERR_h','>f4'),
                      ('UDS_MAG_k','>f4'),('UDS_MAGERR_k','>f4'),
                      ('UDS_MAG_IRAC1','>f4'),('UDS_MAGERR_IRAC1','>f4'),
                      ('UDS_MAG_IRAC2','>f4'),('UDS_MAGERR_IRAC2','>f4'),
                      ('UDS_STAR_FLAG','>i4'),
                      ('UDS_AGE','>f4'),('UDS_Mstar','>f4'),('UDS_SFR','>f4'),('UDS_SSFR','>f4'),('UDS_LNUV','>f4'),

                      ('UDS_S2CLS_ID','>i4'),('UDS_S2CLS_RA','>f8'),('UDS_S2CLS_DEC','>f8'),
                      ('UDS_S2CLS_z','>f4'),('UDS_S2CLS_zerr','>f4',(2,)),('UDS_S2CLS_zchi2','>f4'),
                      ('UDS_S2CLS_MAG_APER_u','>f4',(2,)),('UDS_S2CLS_MAGERR_APER_u','>f4',(2,)),
                      ('UDS_S2CLS_MAG_APER_b','>f4',(2,)),('UDS_S2CLS_MAGERR_APER_b','>f4',(2,)),
                      ('UDS_S2CLS_MAG_APER_v','>f4',(2,)),('UDS_S2CLS_MAGERR_APER_v','>f4',(2,)),
                      ('UDS_S2CLS_MAG_APER_r','>f4',(2,)),('UDS_S2CLS_MAGERR_APER_r','>f4',(2,)),
                      ('UDS_S2CLS_MAG_APER_i','>f4',(2,)),('UDS_S2CLS_MAGERR_APER_i','>f4',(2,)),
                      ('UDS_S2CLS_MAG_APER_z','>f4',(2,)),('UDS_S2CLS_MAGERR_APER_z','>f4',(2,)),
                      ('UDS_S2CLS_MAG_APER_j','>f4',(2,)),('UDS_S2CLS_MAGERR_APER_j','>f4',(2,)),
                      ('UDS_S2CLS_MAG_APER_h','>f4',(2,)),('UDS_S2CLS_MAGERR_APER_h','>f4',(2,)),
                      ('UDS_S2CLS_MAG_APER_k','>f4',(2,)),('UDS_S2CLS_MAGERR_APER_k','>f4',(2,)),
                      ('UDS_S2CLS_MAG_APER_IRAC1','>f4',(2,)),('UDS_S2CLS_MAGERR_APER_IRAC1','>f4',(2,)),
                      ('UDS_S2CLS_MAG_APER_IRAC2','>f4',(2,)),('UDS_S2CLS_MAGERR_APER_IRAC2','>f4',(2,)),
                      ('UDS_S2CLS_STAR_FLAG','>i4'),('UDS_S2CLS_Mstar','>f4'),

                      ('F08_ID','>i4'),('F08_RA','>f8'),('F08_DEC','>f8'),('F08_KRON_RADIUS','>f4'),
                      ('F08_MAG_b','>f4'),('F08_MAGERR_b','>f4'),
                      ('F08_MAG_v','>f4'),('F08_MAGERR_v','>f4'),
                      ('F08_MAG_r','>f4'),('F08_MAGERR_r','>f4'),
                      ('F08_MAG_i','>f4'),('F08_MAGERR_i','>f4'),
                      ('F08_MAG_z','>f4'),('F08_MAGERR_z','>f4'),
                      ('F08_MAG_APER_b','>f4',(2,)),('F08_MAGERR_APER_b','>f4',(2,)),
                      ('F08_MAG_APER_v','>f4',(2,)),('F08_MAGERR_APER_v','>f4',(2,)),
                      ('F08_MAG_APER_r','>f4',(2,)),('F08_MAGERR_APER_r','>f4',(2,)),
                      ('F08_MAG_APER_i','>f4',(2,)),('F08_MAGERR_APER_i','>f4',(2,)),
                      ('F08_MAG_APER_z','>f4',(2,)),('F08_MAGERR_APER_z','>f4',(2,)),

                      ('VIDEO_ID','>i4'),('VIDEO_RA','>f8'),('VIDEO_DEC','>f8'),
                      ('VIDEO_MAG_z','>f4'),('VIDEO_MAGERR_z','>f4'),('VIDEO_ERRFIX_MAGERR_z','>f4'),
                      ('VIDEO_MAG_y','>f4'),('VIDEO_MAGERR_y','>f4'),('VIDEO_ERRFIX_MAGERR_y','>f4'),
                      ('VIDEO_MAG_j','>f4'),('VIDEO_MAGERR_j','>f4'),('VIDEO_ERRFIX_MAGERR_j','>f4'),
                      ('VIDEO_MAG_h','>f4'),('VIDEO_MAGERR_h','>f4'),('VIDEO_ERRFIX_MAGERR_h','>f4'),
                      ('VIDEO_MAG_ks','>f4'),('VIDEO_MAGERR_ks','>f4'),('VIDEO_ERRFIX_MAGERR_ks','>f4'),
                      ('VIDEO_MAG_APER_z','>f4',(5,)),('VIDEO_MAGERR_APER_z','>f4',(5,)),('VIDEO_ERRFIX_MAGERR_APER_z','>f4',(5,)),
                      ('VIDEO_MAG_APER_y','>f4',(5,)),('VIDEO_MAGERR_APER_y','>f4',(5,)),('VIDEO_ERRFIX_MAGERR_APER_y','>f4',(5,)),
                      ('VIDEO_MAG_APER_j','>f4',(5,)),('VIDEO_MAGERR_APER_j','>f4',(5,)),('VIDEO_ERRFIX_MAGERR_APER_j','>f4',(5,)),
                      ('VIDEO_MAG_APER_h','>f4',(5,)),('VIDEO_MAGERR_APER_h','>f4',(5,)),('VIDEO_ERRFIX_MAGERR_APER_h','>f4',(5,)),
                      ('VIDEO_MAG_APER_ks','>f4',(5,)),('VIDEO_MAGERR_APER_ks','>f4',(5,)),('VIDEO_ERRFIX_MAGERR_APER_ks','>f4',(5,)),

                      ('UDSz_ID','>i4'),('UDSz_RA','>f8'),('UDSz_DEC','>f8'),('UDSz_SPECz','>f4'),('UDSz_FLAG','<U3'),
                      ('XUDS_ID','<U20'),('XUDS_RA','>f8'),('XUDS_DEC','>f8'),('XUDS_SPECz','>f4'),('XUDS_REF','<U4'),
                      ('M15_ID','>i4'),('M15_RA','>f8'),('M15_DEC','>f8'),('M15_GRISMz','>f4'),('M15_SPECz','>f4'),('M15_QUALITY','>f4'),
                      ('VIPERS_ID','>i4'),('VIPERS_RA','>f8'),('VIPERS_DEC','>f8'),('VIPERS_SPECz','>f4'),('VIPERS_FLAG','>f4'),
                      ('C3R2_ID','>i4'),('C3R2_RA','>f8'),('C3R2_DEC','>f8'),('C3R2_EBV','>f4'),('C3R2_INSTR','<U15'),('C3R2_SPECz','>f4'),('C3R2_FLAG','>f4'),
                      ('SUBARU_ID','<U46'),('SUBARU_RA','>f8'),('SUBARU_DEC','>f8'),('SUBARU_SPECz','>f4'),('SUBARU_REF','<U18'),
                      ('3DHST_ID','>i4'),('3DHST_RA','>f4'),('3DHST_DEC','>f4'),('3DHST_GRISMz','>f4'),
                      ('NED_ID','>i4'),('NED_RA','>f8'),('NED_DEC','>f8'),('NED_z','>f4'),('NED_MAG','<U5')]

def match_radio(catalog):

    radio_catalog = fitsio.getdata('/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/radio/SXDS_radio_100uJy_catalog_Simpsons06.fits')

    m1,m2,d12 = match_ra_dec(catalog['RA'],catalog['DEC'],radio_catalog['RAo'],radio_catalog['DEo'])
    cond = (m2 != len(radio_catalog))
    m1, m2 = m1[cond], m2[cond]
    print ('Radio Catalog: Matched %i out of %i sources (%i available)' % (len(m1), len(catalog), len(radio_catalog)))

    catalog['1_4GHz_ID'][m1]      = radio_catalog['Seq'][m2]
    catalog['1_4GHz_RA'][m1]      = radio_catalog['RAJ2000'][m2]
    catalog['1_4GHz_DEC'][m1]     = radio_catalog['DEJ2000'][m2]
    catalog['1_4GHz_FLUX'][m1]    = radio_catalog['S1_4GHz'][m2]
    catalog['1_4GHz_FLUXERR'][m1] = radio_catalog['e_S1_4GHz'][m2]
    catalog['1_4GHz_Rel'][m1]     = radio_catalog['Rel'][m2]
    catalog['1_4GHz_n_Rel'][m1]   = radio_catalog['n_Rel'][m2]

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/radio_Simpsons06.reg",
                    np.vstack((radio_catalog["RAo"],radio_catalog["DEo"])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=red",header='fk5',comments='')

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/radio_Simpsons06.matched.reg",
                    np.vstack((catalog["RA"][m1],catalog["DEC"][m1])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=green",header='fk5',comments='')

    return catalog

def match_xray(catalog):

    xray_catalog = fitsio.getdata('/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/xray/SXDS_Xray_catalog_Akiyama15.fits')

    m1,m2,d12 = match_ra_dec(catalog['RA'],catalog['DEC'],xray_catalog['ORA'],xray_catalog['ODE'])
    cond = (m2 != len(xray_catalog))
    m1, m2 = m1[cond], m2[cond]
    print ('X-ray Catalog: Matched %i out of %i sources (%i available)' % (len(m1), len(catalog), len(xray_catalog)))

    catalog['Xray_ID'][m1]     = xray_catalog['ID'][m2]
    catalog['Xray_RA'][m1]     = xray_catalog['XRA'][m2]
    catalog['Xray_DEC'][m1]    = xray_catalog['XDE'][m2]
    catalog['Xray_zUSE'][m1]   = xray_catalog['zUSE'][m2]
    catalog['Xray_HR2'][m1]    = xray_catalog['HR2'][m2]
    catalog['Xray_logNH'][m1]  = xray_catalog['logNH'][m2]
    catalog['Xray_logLHX'][m1] = xray_catalog['logLHX'][m2]

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/xray_Akiyama15.reg",
                    np.vstack((xray_catalog["ORA"],xray_catalog["ODE"])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=red",header='fk5',comments='')

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/xray_Akiyama15.matched.reg",
                    np.vstack((catalog["RA"][m1],catalog["DEC"][m1])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=green",header='fk5',comments='')

    return catalog

def match_ps1(catalog):

    ps1_catalog = fitsio.getdata('/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/UDSwide_hasinger.fits',1)

    m1,m2,d12 = match_ra_dec(catalog['RA'],catalog['DEC'],ps1_catalog['o_ra'],ps1_catalog['o_dec'])
    cond = (m2 != len(ps1_catalog))
    m1, m2 = m1[cond], m2[cond]
    print ('PS1 Catalog: Matched %i out of %i sources (%i available)' % (len(m1), len(catalog), len(ps1_catalog)))

    catalog['PS1_ID'][m1]       = ps1_catalog['o_objID'][m2]
    catalog['PS1_RA'][m1]       = ps1_catalog['o_ra'][m2]
    catalog['PS1_RAERR'][m1]    = ps1_catalog['o_raErr'][m2]
    catalog['PS1_DEC'][m1]      = ps1_catalog['o_dec'][m2]
    catalog['PS1_DECERR'][m1]   = ps1_catalog['o_decErr'][m2]
    catalog['PS1_MAG_g'][m1]    = ps1_catalog['o_gStackKronMag'][m2]
    catalog['PS1_MAGERR_g'][m1] = ps1_catalog['o_gStackKronMagErr'][m2]
    catalog['PS1_MAG_r'][m1]    = ps1_catalog['o_rStackKronMag'][m2]
    catalog['PS1_MAGERR_r'][m1] = ps1_catalog['o_rStackKronMagErr'][m2]
    catalog['PS1_MAG_i'][m1]    = ps1_catalog['o_iStackKronMag'][m2]
    catalog['PS1_MAGERR_i'][m1] = ps1_catalog['o_iStackKronMagErr'][m2]
    catalog['PS1_MAG_z'][m1]    = ps1_catalog['o_zStackKronMag'][m2]
    catalog['PS1_MAGERR_z'][m1] = ps1_catalog['o_zStackKronMagErr'][m2]
    catalog['PS1_MAG_y'][m1]    = ps1_catalog['o_yStackKronMag'][m2]
    catalog['PS1_MAGERR_y'][m1] = ps1_catalog['o_yStackKronMagErr'][m2]

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/PS1.reg",
                    np.vstack((ps1_catalog["o_ra"],ps1_catalog["o_dec"])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=red",header='fk5',comments='')

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/PS1.matched.reg",
                    np.vstack((catalog["RA"][m1],catalog["DEC"][m1])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=green",header='fk5',comments='')

    return catalog

def match_uds(catalog):

    uds_catalog = fitsio.getdata('/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/UDS-DR10-multi-140630_Kref_cold+hot_ext.cat_Ksel.2_Mgas.fits',1)

    m1,m2,d12 = match_ra_dec(catalog['RA'],catalog['DEC'],uds_catalog['RA'],uds_catalog['DEC'])
    cond = (m2 != len(uds_catalog))
    m1, m2 = m1[cond], m2[cond]
    print ('UDS Catalog: Matched %i out of %i sources (%i available)' % (len(m1), len(catalog), len(uds_catalog)))

    catalog['UDS_ID'][m1]           = uds_catalog['ID'][m2]
    catalog['UDS_RA'][m1]           = uds_catalog['RA'][m2]
    catalog['UDS_DEC'][m1]          = uds_catalog['DEC'][m2]
    catalog['UDS_z'][m1]            = uds_catalog['redshift'][m2]
    catalog['UDS_zerr'][m1]         = uds_catalog['redshift_err'][m2]
    catalog['UDS_ztype'][m1]        = uds_catalog['z_type'][m2]
    catalog['UDS_MAG_NUV'][m1]      = uds_catalog['NUV_tot'][m2]
    catalog['UDS_MAGERR_NUV'][m1]   = uds_catalog['NUV_tot_err'][m2]
    catalog['UDS_MAG_u'][m1]        = uds_catalog['u_tot'][m2]
    catalog['UDS_MAGERR_u'][m1]     = uds_catalog['u_tot_err'][m2]
    catalog['UDS_MAG_b'][m1]        = uds_catalog['B_tot'][m2]
    catalog['UDS_MAGERR_b'][m1]     = uds_catalog['B_tot_err'][m2]
    catalog['UDS_MAG_v'][m1]        = uds_catalog['V_tot'][m2]
    catalog['UDS_MAGERR_v'][m1]     = uds_catalog['V_tot_err'][m2]
    catalog['UDS_MAG_r'][m1]        = uds_catalog['R_tot'][m2]
    catalog['UDS_MAGERR_r'][m1]     = uds_catalog['R_tot_err'][m2]
    catalog['UDS_MAG_i'][m1]        = uds_catalog['i_tot'][m2]
    catalog['UDS_MAGERR_i'][m1]     = uds_catalog['i_tot_err'][m2]
    catalog['UDS_MAG_z'][m1]        = uds_catalog['z_tot'][m2]
    catalog['UDS_MAGERR_z'][m1]     = uds_catalog['z_tot_err'][m2]
    catalog['UDS_MAG_j'][m1]        = uds_catalog['J_tot'][m2]
    catalog['UDS_MAGERR_j'][m1]     = uds_catalog['J_tot_err'][m2]
    catalog['UDS_MAG_h'][m1]        = uds_catalog['H_tot'][m2]
    catalog['UDS_MAGERR_h'][m1]     = uds_catalog['H_tot_err'][m2]
    catalog['UDS_MAG_k'][m1]        = uds_catalog['K_tot'][m2]
    catalog['UDS_MAGERR_k'][m1]     = uds_catalog['K_tot_err'][m2]
    catalog['UDS_MAG_IRAC1'][m1]    = uds_catalog['IRAC1_tot'][m2]
    catalog['UDS_MAGERR_IRAC1'][m1] = uds_catalog['IRAC1_tot_err'][m2]
    catalog['UDS_MAG_IRAC2'][m1]    = uds_catalog['IRAC2_tot'][m2]
    catalog['UDS_MAGERR_IRAC2'][m1] = uds_catalog['IRAC2_tot_err'][m2]
    catalog['UDS_STAR_FLAG'][m1]    = uds_catalog['star_flag'][m2]

    catalog['UDS_AGE'][m1]          = uds_catalog['age'][m2]
    catalog['UDS_Mstar'][m1]        = uds_catalog['Mstar'][m2]
    catalog['UDS_SFR'][m1]          = uds_catalog['SFR'][m2]
    catalog['UDS_SSFR'][m1]         = uds_catalog['SSFR'][m2]
    catalog['UDS_LNUV'][m1]         = uds_catalog['LUM_NUV'][m2]

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/UDS_DR10.reg",
                    np.vstack((uds_catalog["RA"],uds_catalog["DEC"])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=red",header='fk5',comments='')

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/UDS_DR10.matched.reg",
                    np.vstack((catalog["RA"][m1],catalog["DEC"][m1])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=green",header='fk5',comments='')

    return catalog

def match_uds_s2cls(catalog):

    uds_s2cls_catalog = fitsio.getdata('/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/UDS_DR8_forS2CLS_v4.dat.fits',1)

    m1,m2,d12 = match_ra_dec(catalog['RA'],catalog['DEC'],uds_s2cls_catalog['RA'],uds_s2cls_catalog['DEC'])
    cond = (m2 != len(uds_s2cls_catalog))
    m1, m2 = m1[cond], m2[cond]
    print ('UDS S2CLS Catalog: Matched %i out of %i sources (%i available)' % (len(m1), len(catalog), len(uds_s2cls_catalog)))

    catalog['UDS_S2CLS_ID'][m1]                = uds_s2cls_catalog['ID'][m2]
    catalog['UDS_S2CLS_RA'][m1]                = uds_s2cls_catalog['RA'][m2]
    catalog['UDS_S2CLS_DEC'][m1]               = uds_s2cls_catalog['DEC'][m2]
    catalog['UDS_S2CLS_z'][m1]                 = uds_s2cls_catalog['z_phot_maxL'][m2]
    catalog['UDS_S2CLS_zchi2'][m1]             = uds_s2cls_catalog['chisq_at_maxL'][m2]
    catalog['UDS_S2CLS_zerr'][m1]              = np.vstack((uds_s2cls_catalog['lower_68'][m2],uds_s2cls_catalog['lower_68'][m2])).T
    catalog['UDS_S2CLS_MAG_APER_u'][m1]        = np.vstack((uds_s2cls_catalog['MAG_APER_u_2.0'][m2],uds_s2cls_catalog['MAG_APER_u_3.0'][m2])).T
    catalog['UDS_S2CLS_MAGERR_APER_u'][m1]     = np.vstack((uds_s2cls_catalog['MAGERR_APER_u_2.0'][m2],uds_s2cls_catalog['MAGERR_APER_u_3.0'][m2])).T
    catalog['UDS_S2CLS_MAG_APER_b'][m1]        = np.vstack((uds_s2cls_catalog['MAG_APER_B_2.0'][m2],uds_s2cls_catalog['MAG_APER_B_3.0'][m2])).T
    catalog['UDS_S2CLS_MAGERR_APER_b'][m1]     = np.vstack((uds_s2cls_catalog['MAGERR_APER_B_2.0'][m2],uds_s2cls_catalog['MAGERR_APER_B_3.0'][m2])).T
    catalog['UDS_S2CLS_MAG_APER_v'][m1]        = np.vstack((uds_s2cls_catalog['MAG_APER_V_2.0'][m2],uds_s2cls_catalog['MAG_APER_V_3.0'][m2])).T
    catalog['UDS_S2CLS_MAGERR_APER_v'][m1]     = np.vstack((uds_s2cls_catalog['MAGERR_APER_V_2.0'][m2],uds_s2cls_catalog['MAGERR_APER_V_3.0'][m2])).T
    catalog['UDS_S2CLS_MAG_APER_r'][m1]        = np.vstack((uds_s2cls_catalog['MAG_APER_R_2.0'][m2],uds_s2cls_catalog['MAG_APER_R_3.0'][m2])).T
    catalog['UDS_S2CLS_MAGERR_APER_r'][m1]     = np.vstack((uds_s2cls_catalog['MAGERR_APER_R_2.0'][m2],uds_s2cls_catalog['MAGERR_APER_R_3.0'][m2])).T
    catalog['UDS_S2CLS_MAG_APER_i'][m1]        = np.vstack((uds_s2cls_catalog['MAG_APER_i_2.0'][m2],uds_s2cls_catalog['MAG_APER_i_3.0'][m2])).T
    catalog['UDS_S2CLS_MAGERR_APER_i'][m1]     = np.vstack((uds_s2cls_catalog['MAGERR_APER_i_2.0'][m2],uds_s2cls_catalog['MAGERR_APER_i_3.0'][m2])).T
    catalog['UDS_S2CLS_MAG_APER_z'][m1]        = np.vstack((uds_s2cls_catalog['MAG_APER_z_2.0'][m2],uds_s2cls_catalog['MAG_APER_z_3.0'][m2])).T
    catalog['UDS_S2CLS_MAGERR_APER_z'][m1]     = np.vstack((uds_s2cls_catalog['MAGERR_APER_z_2.0'][m2],uds_s2cls_catalog['MAGERR_APER_z_3.0'][m2])).T
    catalog['UDS_S2CLS_MAG_APER_j'][m1]        = np.vstack((uds_s2cls_catalog['MAG_APER_J_2.0'][m2],uds_s2cls_catalog['MAG_APER_J_3.0'][m2])).T
    catalog['UDS_S2CLS_MAGERR_APER_j'][m1]     = np.vstack((uds_s2cls_catalog['MAGERR_APER_J_2.0'][m2],uds_s2cls_catalog['MAGERR_APER_J_3.0'][m2])).T
    catalog['UDS_S2CLS_MAG_APER_h'][m1]        = np.vstack((uds_s2cls_catalog['MAG_APER_H_2.0'][m2],uds_s2cls_catalog['MAG_APER_H_3.0'][m2])).T
    catalog['UDS_S2CLS_MAGERR_APER_h'][m1]     = np.vstack((uds_s2cls_catalog['MAGERR_APER_H_2.0'][m2],uds_s2cls_catalog['MAGERR_APER_H_3.0'][m2])).T
    catalog['UDS_S2CLS_MAG_APER_k'][m1]        = np.vstack((uds_s2cls_catalog['MAG_APER_K_2.0'][m2],uds_s2cls_catalog['MAG_APER_K_3.0'][m2])).T
    catalog['UDS_S2CLS_MAGERR_APER_k'][m1]     = np.vstack((uds_s2cls_catalog['MAGERR_APER_K_2.0'][m2],uds_s2cls_catalog['MAGERR_APER_K_3.0'][m2])).T
    catalog['UDS_S2CLS_MAG_APER_IRAC1'][m1]    = np.vstack((uds_s2cls_catalog['MAG_APER_IRAC1_2.0'][m2],uds_s2cls_catalog['MAG_APER_IRAC1_3.0'][m2])).T
    catalog['UDS_S2CLS_MAGERR_APER_IRAC1'][m1] = np.vstack((uds_s2cls_catalog['MAGERR_APER_IRAC1_2.0'][m2],uds_s2cls_catalog['MAGERR_APER_IRAC1_3.0'][m2])).T
    catalog['UDS_S2CLS_MAG_APER_IRAC2'][m1]    = np.vstack((uds_s2cls_catalog['MAG_APER_IRAC2_2.0'][m2],uds_s2cls_catalog['MAG_APER_IRAC2_3.0'][m2])).T
    catalog['UDS_S2CLS_MAGERR_APER_IRAC2'][m1] = np.vstack((uds_s2cls_catalog['MAGERR_APER_IRAC2_2.0'][m2],uds_s2cls_catalog['MAGERR_APER_IRAC2_3.0'][m2])).T
    catalog['UDS_S2CLS_STAR_FLAG'][m1]         = uds_s2cls_catalog['Stars_2'][m2]

    catalog['UDS_S2CLS_Mstar'][m1]             = uds_s2cls_catalog['Bestfit_Mass'][m2]

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/UDS_S2CLS.reg",
                    np.vstack((uds_s2cls_catalog["RA"],uds_s2cls_catalog["DEC"])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=red",header='fk5',comments='')

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/UDS_S2CLS.matched.reg",
                    np.vstack((catalog["RA"][m1],catalog["DEC"][m1])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=green",header='fk5',comments='')

    return catalog

def match_furusawa08(catalog):

    f08_catalog = fitsio.getdata('/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/furusawa08.fits')

    m1,m2,d12 = match_ra_dec(catalog['RA'],catalog['DEC'],f08_catalog['RAJ2000'],f08_catalog['DEJ2000'])
    cond = (m2 != len(f08_catalog))
    m1, m2 = m1[cond], m2[cond]
    print ('Furusawa+08 Catalog: Matched %i out of %i sources (%i available)' % (len(m1), len(catalog), len(f08_catalog)))

    for x in ['BmagA','e_BmagA','VmagA','e_VmagA','RmagA','e_RmagA','imagA','e_imagA','zmagA','e_zmagA',
              'Bmag2','e_Bmag2','Vmag2','e_Vmag2','Rmag2','e_Rmag2','imag2','e_imag2','zmag2','e_zmag2',
              'Bmag3','e_Bmag3','Vmag3','e_Vmag3','Rmag3','e_Rmag3','imag3','e_imag3','zmag3','e_zmag3']:

        cond = np.isnan(f08_catalog[x])
        f08_catalog[x][cond] = -99.

    catalog['F08_ID'][m1]                = f08_catalog['Seq'][m2]
    catalog['F08_RA'][m1]                = f08_catalog['RAJ2000'][m2]
    catalog['F08_DEC'][m1]               = f08_catalog['DEJ2000'][m2]
    catalog['F08_KRON_RADIUS'][m1]       = f08_catalog['rK'][m2]
    catalog['F08_MAG_b'][m1]             = f08_catalog['BmagA'][m2]
    catalog['F08_MAGERR_b'][m1]          = f08_catalog['e_BmagA'][m2]
    catalog['F08_MAG_v'][m1]             = f08_catalog['VmagA'][m2]
    catalog['F08_MAGERR_v'][m1]          = f08_catalog['e_VmagA'][m2]
    catalog['F08_MAG_r'][m1]             = f08_catalog['RmagA'][m2]
    catalog['F08_MAGERR_r'][m1]          = f08_catalog['e_RmagA'][m2]
    catalog['F08_MAG_i'][m1]             = f08_catalog['imagA'][m2]
    catalog['F08_MAGERR_i'][m1]          = f08_catalog['e_imagA'][m2]
    catalog['F08_MAG_z'][m1]             = f08_catalog['zmagA'][m2]
    catalog['F08_MAGERR_z'][m1]          = f08_catalog['e_zmagA'][m2]
    catalog['F08_MAG_APER_b'][m1]        = np.vstack((f08_catalog['Bmag2'][m2],f08_catalog['Bmag3'][m2])).T
    catalog['F08_MAGERR_APER_b'][m1]     = np.vstack((f08_catalog['e_Bmag2'][m2],f08_catalog['e_Bmag3'][m2])).T
    catalog['F08_MAG_APER_v'][m1]        = np.vstack((f08_catalog['Vmag2'][m2],f08_catalog['Vmag3'][m2])).T
    catalog['F08_MAGERR_APER_v'][m1]     = np.vstack((f08_catalog['e_Vmag2'][m2],f08_catalog['e_Vmag3'][m2])).T
    catalog['F08_MAG_APER_r'][m1]        = np.vstack((f08_catalog['Rmag2'][m2],f08_catalog['Rmag3'][m2])).T
    catalog['F08_MAGERR_APER_r'][m1]     = np.vstack((f08_catalog['e_Rmag2'][m2],f08_catalog['e_Rmag3'][m2])).T
    catalog['F08_MAG_APER_i'][m1]        = np.vstack((f08_catalog['imag2'][m2],f08_catalog['imag3'][m2])).T
    catalog['F08_MAGERR_APER_i'][m1]     = np.vstack((f08_catalog['e_imag2'][m2],f08_catalog['e_imag3'][m2])).T
    catalog['F08_MAG_APER_z'][m1]        = np.vstack((f08_catalog['zmag2'][m2],f08_catalog['zmag3'][m2])).T
    catalog['F08_MAGERR_APER_z'][m1]     = np.vstack((f08_catalog['e_zmag2'][m2],f08_catalog['e_zmag3'][m2])).T

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/furusawa08.reg",
                    np.vstack((f08_catalog["RAJ2000"],f08_catalog["DEJ2000"])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=red",header='fk5',comments='')

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/furusawa08.matched.reg",
                    np.vstack((catalog["RA"][m1],catalog["DEC"][m1])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=green",header='fk5',comments='')

    return catalog

def match_video(catalog):

    video_catalog = fitsio.getdata('/data/highzgal/PUBLICACCESS/SPLASH/DATA/VIDEO/xmm/cats/VIDEO-xmm_2016-04-14_fullcat.fits')

    for x in video_catalog.dtype.names:
        if "AUTO" in x or "APER" in x:
            cond = (video_catalog[x] == 99.)
            video_catalog[x][cond] = -99.

    m1,m2,d12 = match_ra_dec(catalog['RA'],catalog['DEC'],video_catalog['ALPHA_J2000'],video_catalog['DELTA_J2000'])
    cond = (m2 != len(video_catalog))
    m1, m2 = m1[cond], m2[cond]
    print ('VIDEO Catalog: Matched %i out of %i sources (%i available)' % (len(m1), len(catalog), len(video_catalog)))

    catalog['VIDEO_ID'][m1]           = video_catalog['ID'][m2]
    catalog['VIDEO_RA'][m1]           = video_catalog['ALPHA_J2000'][m2]
    catalog['VIDEO_DEC'][m1]          = video_catalog['DELTA_J2000'][m2]

    catalog['VIDEO_MAG_z'][m1]        = video_catalog['Z_MAG_AUTO'][m2]
    catalog['VIDEO_MAGERR_z'][m1]     = video_catalog['Z_MAGERR_AUTO'][m2]
    catalog['VIDEO_MAG_y'][m1]        = video_catalog['Y_MAG_AUTO'][m2]
    catalog['VIDEO_MAGERR_y'][m1]     = video_catalog['Y_MAGERR_AUTO'][m2]
    catalog['VIDEO_MAG_j'][m1]        = video_catalog['J_MAG_AUTO'][m2]
    catalog['VIDEO_MAGERR_j'][m1]     = video_catalog['J_MAGERR_AUTO'][m2]
    catalog['VIDEO_MAG_h'][m1]        = video_catalog['H_MAG_AUTO'][m2]
    catalog['VIDEO_MAGERR_h'][m1]     = video_catalog['H_MAGERR_AUTO'][m2]
    catalog['VIDEO_MAG_ks'][m1]       = video_catalog['K_MAG_AUTO'][m2]
    catalog['VIDEO_MAGERR_ks'][m1]    = video_catalog['K_MAGERR_AUTO'][m2]

    for i in range(5):

        catalog['VIDEO_MAG_APER_z'][m1,i]    = video_catalog['Z_MAG_APER_%i'%(i+1)][m2]
        catalog['VIDEO_MAGERR_APER_z'][m1,i] = video_catalog['Z_MAGERR_APER_%i'%(i+1)][m2]
        catalog['VIDEO_MAG_APER_y'][m1,i]    = video_catalog['Y_MAG_APER_%i'%(i+1)][m2]
        catalog['VIDEO_MAGERR_APER_y'][m1,i] = video_catalog['Y_MAGERR_APER_%i'%(i+1)][m2]
        catalog['VIDEO_MAG_APER_j'][m1,i]    = video_catalog['J_MAG_APER_%i'%(i+1)][m2]
        catalog['VIDEO_MAGERR_APER_j'][m1,i] = video_catalog['J_MAGERR_APER_%i'%(i+1)][m2]
        catalog['VIDEO_MAG_APER_h'][m1,i]    = video_catalog['H_MAG_APER_%i'%(i+1)][m2]
        catalog['VIDEO_MAGERR_APER_h'][m1,i] = video_catalog['H_MAGERR_APER_%i'%(i+1)][m2]
        catalog['VIDEO_MAG_APER_ks'][m1,i]    = video_catalog['K_MAG_APER_%i'%(i+1)][m2]
        catalog['VIDEO_MAGERR_APER_ks'][m1,i] = video_catalog['K_MAGERR_APER_%i'%(i+1)][m2]

    video_catalog = fitsio.getdata('/data/highzgal/PUBLICACCESS/SPLASH/DATA/VIDEO/xmm/cats/VIDEO-xmm_2016-04-14_fullcat_errfix.fits')

    for x in video_catalog.dtype.names:
        if "AUTO" in x or "APER" in x:
            cond = (video_catalog[x] == 99.)
            video_catalog[x][cond] = -99.

    catalog['VIDEO_ERRFIX_MAGERR_z'][m1] = video_catalog['Z_MAGERR_AUTO'][m2]
    catalog['VIDEO_ERRFIX_MAGERR_y'][m1] = video_catalog['Y_MAGERR_AUTO'][m2]
    catalog['VIDEO_ERRFIX_MAGERR_j'][m1] = video_catalog['J_MAGERR_AUTO'][m2]
    catalog['VIDEO_ERRFIX_MAGERR_h'][m1] = video_catalog['H_MAGERR_AUTO'][m2]
    catalog['VIDEO_ERRFIX_MAGERR_ks'][m1] = video_catalog['K_MAGERR_AUTO'][m2]

    for i in range(5):

            catalog['VIDEO_ERRFIX_MAGERR_APER_z'][m1,i] = video_catalog['Z_MAGERR_APER_%i'%(i+1)][m2]
            catalog['VIDEO_ERRFIX_MAGERR_APER_y'][m1,i] = video_catalog['Y_MAGERR_APER_%i'%(i+1)][m2]
            catalog['VIDEO_ERRFIX_MAGERR_APER_j'][m1,i] = video_catalog['J_MAGERR_APER_%i'%(i+1)][m2]
            catalog['VIDEO_ERRFIX_MAGERR_APER_h'][m1,i] = video_catalog['H_MAGERR_APER_%i'%(i+1)][m2]
            catalog['VIDEO_ERRFIX_MAGERR_APER_ks'][m1,i] = video_catalog['K_MAGERR_APER_%i'%(i+1)][m2]

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/video.reg",
                    np.vstack((video_catalog["ALPHA_J2000"],video_catalog["DELTA_J2000"])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=red",header='fk5',comments='')

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/video.matched.reg",
                    np.vstack((catalog["RA"][m1],catalog["DEC"][m1])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=green",header='fk5',comments='')

    return catalog

def match_c3r2(catalog):

    ipac = fitsio.getdata('/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/allspec_capak.fits',1)
    ipac = ipac[np.isfinite(ipac["ra"]) & np.isfinite(ipac["dec"])]

    m1,m2,d12 = match_ra_dec(catalog['RA'],catalog['DEC'],ipac['ra'],ipac['dec'])
    cond = (m2 != len(ipac))
    m1, m2 = m1[cond], m2[cond]
    print ('IPAC Spec-z Catalog: Matched %i out of %i sources (%i available)' % (len(m1), len(catalog), len(ipac))    )

    catalog['C3R2_ID'][m1]    = ipac['id'][m2]
    catalog['C3R2_RA'][m1]    = ipac['ra'][m2]
    catalog['C3R2_DEC'][m1]   = ipac['dec'][m2]
    catalog['C3R2_EBV'][m1]   = ipac['e(b-v)'][m2]
    catalog['C3R2_INSTR'][m1] = ipac['Instr'][m2]
    catalog['C3R2_SPECz'][m1] = ipac['specz'][m2]
    catalog['C3R2_FLAG'][m1]  = ipac['flags'][m2]

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/ipac_specz.reg",
                    np.vstack((ipac["ra"],ipac["dec"])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=red",header='fk5',comments='')

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/ipac_specz.matched.reg",
                    np.vstack((catalog["RA"][m1],catalog["DEC"][m1])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=green",header='fk5',comments='')

    return catalog

def match_udsz(catalog):

    udsz_catalog = fitsio.getdata('/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/UDSz-secure-March2014.fits',1)

    m1,m2,d12 = match_ra_dec(catalog['RA'],catalog['DEC'],udsz_catalog['RA'],udsz_catalog['DEC'])
    cond = (m2 != len(udsz_catalog))
    m1, m2 = m1[cond], m2[cond]
    print ('UDSz Catalog: Matched %i out of %i sources (%i available)' % (len(m1), len(catalog), len(udsz_catalog)))

    catalog['UDSz_ID'][m1]    = udsz_catalog['DR1ID'][m2]
    catalog['UDSz_RA'][m1]    = udsz_catalog['RA'][m2]
    catalog['UDSz_DEC'][m1]   = udsz_catalog['DEC'][m2]
    catalog['UDSz_SPECz'][m1] = udsz_catalog['redshift'][m2]
    catalog['UDSz_FLAG'][m1]  = udsz_catalog['Flag'][m2]

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/UDSz.reg",
                    np.vstack((udsz_catalog["RA"],udsz_catalog["DEC"])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=red",header='fk5',comments='')

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/UDSz.matched.reg",
                    np.vstack((catalog["RA"][m1],catalog["DEC"][m1])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=green",header='fk5',comments='')

    return catalog

def match_xuds_compile(catalog):

    specz_compile = np.genfromtxt('/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/A15+SIP+UDSz+S15+S06+G07+F10+O08+VB07+Y05.cat',
                                    dtype=[('ID',np.object),('RA',float),('DEC',float),('SPECz',float),('Ref',np.object),('Comment',np.object)])

    remove = (specz_compile['Ref']=='SIP') & (specz_compile['SPECz']<=0.0)
    specz_compile = specz_compile[~remove]

    remove = (specz_compile['SPECz'] == 9.999)
    specz_compile = specz_compile[~remove]

    m1,m2,d12 = match_ra_dec(catalog['RA'],catalog['DEC'],specz_compile['RA'],specz_compile['DEC'])
    cond = (m2 != len(specz_compile))
    m1, m2 = m1[cond], m2[cond]
    print ('Compiled spec-z Catalog: Matched %i out of %i sources (%i available)' % (len(m1), len(catalog), len(specz_compile)))

    catalog['XUDS_ID'][m1]    = specz_compile['ID'][m2]
    catalog['XUDS_RA'][m1]    = specz_compile['RA'][m2]
    catalog['XUDS_DEC'][m1]   = specz_compile['DEC'][m2]
    catalog['XUDS_SPECz'][m1] = specz_compile['SPECz'][m2]
    catalog['XUDS_REF'][m1]   = specz_compile['Ref'][m2]

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/specz_compile.reg",
                    np.vstack((specz_compile["RA"],specz_compile["DEC"])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=red",header='fk5',comments='')

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/specz_compile.matched.reg",
                    np.vstack((catalog["RA"][m1],catalog["DEC"][m1])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=green",header='fk5',comments='')

    return catalog

def match_subaru_compile(catalog):

    specz_compile_NB = np.genfromtxt('/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/sxds_specz_all_v2.dat',
                                    dtype=[('ID',np.object),('RA',float),('DEC',float),('SPECz',float),('Ref',np.object)])

    m1,m2,d12 = match_ra_dec(catalog['RA'],catalog['DEC'],specz_compile_NB['RA'],specz_compile_NB['DEC'])
    cond = (m2 != len(specz_compile_NB))
    m1, m2 = m1[cond], m2[cond]
    print ('Compiled NB spec-z Catalog: Matched %i out of %i sources (%i available)' % (len(m1), len(catalog), len(specz_compile_NB)))

    catalog['SUBARU_ID'][m1]    = specz_compile_NB['ID'][m2]
    catalog['SUBARU_RA'][m1]    = specz_compile_NB['RA'][m2]
    catalog['SUBARU_DEC'][m1]   = specz_compile_NB['DEC'][m2]
    catalog['SUBARU_SPECz'][m1] = specz_compile_NB['SPECz'][m2]
    catalog['SUBARU_REF'][m1]   = specz_compile_NB['Ref'][m2]

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/sxds_specz_all_v2.reg",
                    np.vstack((specz_compile_NB["RA"],specz_compile_NB["DEC"])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=red",header='fk5',comments='')

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/sxds_specz_all_v2.matched.reg",
                    np.vstack((catalog["RA"][m1],catalog["DEC"][m1])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=green",header='fk5',comments='')

    return catalog

def match_morris14(catalog):

    morris14 = np.genfromtxt('/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/Morris14_grismz.UDS.070715.txt',
                                    dtype=[('ID',int),('RA',float),('DEC',float),('GRISMz',float),('SPECz',float),('QUALITY',float)])

    m1,m2,d12 = match_ra_dec(catalog['RA'],catalog['DEC'],morris14['RA'],morris14['DEC'])
    cond = (m2 != len(morris14))
    m1, m2 = m1[cond], m2[cond]
    print ('Morris14 Catalog: Matched %i out of %i sources (%i available)' % (len(m1), len(catalog), len(morris14)))

    catalog['M15_ID'][m1]      = morris14['ID'][m2]
    catalog['M15_RA'][m1]      = morris14['RA'][m2]
    catalog['M15_DEC'][m1]     = morris14['DEC'][m2]
    catalog['M15_GRISMz'][m1]  = morris14['GRISMz'][m2]
    catalog['M15_SPECz'][m1]   = morris14['SPECz'][m2]
    catalog['M15_QUALITY'][m1] = morris14['QUALITY'][m2]

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/morris14.reg",
                    np.vstack((morris14["RA"],morris14["DEC"])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=red",header='fk5',comments='')

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/morris14.matched.reg",
                    np.vstack((catalog["RA"][m1],catalog["DEC"][m1])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=green",header='fk5',comments='')

    return catalog

def match_vipers(catalog):

    vipers = fitsio.getdata('/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/VIPERS_W1_SPECTRO_PDR2.fits')

    m1,m2,d12 = match_ra_dec(catalog['RA'],catalog['DEC'],vipers['alpha'],vipers['delta'])
    cond = (m2 != len(vipers))
    m1, m2 = m1[cond], m2[cond]
    print ('VIPERS Catalog: Matched %i out of %i sources (%i available)' % (len(m1), len(catalog), len(vipers)))

    catalog['VIPERS_ID'][m1]    = vipers['num'][m2]
    catalog['VIPERS_RA'][m1]    = vipers['alpha'][m2]
    catalog['VIPERS_DEC'][m1]   = vipers['delta'][m2]
    catalog['VIPERS_SPECz'][m1] = vipers['zspec'][m2]
    catalog['VIPERS_FLAG'][m1]  = vipers['zflg'][m2]

    cond = (catalog['VIPERS_SPECz'] == 0.)
    catalog['VIPERS_SPECz'][cond] = -99.

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/vipers.reg",
                    np.vstack((vipers["alpha"],vipers["delta"])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=red",header='fk5',comments='')

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/vipers.matched.reg",
                    np.vstack((catalog["RA"][m1],catalog["DEC"][m1])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=green",header='fk5',comments='')

    return catalog

def match_3dhst(catalog):

    catalog_3dhst = fitsio.getdata('/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/3dhst.v4.1.5.master.fits')

    m1,m2,d12 = match_ra_dec(catalog['RA'],catalog['DEC'],catalog_3dhst['ra'],catalog_3dhst['dec'])
    cond = (m2 != len(catalog_3dhst))
    m1, m2 = m1[cond], m2[cond]
    print ('3DHST Catalog: Matched %i out of %i sources (%i available)' % (len(m1), len(catalog), len(catalog_3dhst)))

    catalog['3DHST_ID'][m1]      = catalog_3dhst['phot_id'][m2]
    catalog['3DHST_RA'][m1]      = catalog_3dhst['ra'][m2]
    catalog['3DHST_DEC'][m1]     = catalog_3dhst['dec'][m2]
    catalog['3DHST_GRISMz'][m1]  = catalog_3dhst['z_peak_grism'][m2]

    cond = (catalog['3DHST_GRISMz'] == -1.)
    catalog['3DHST_GRISMz'][cond] = -99.

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/3DHST.reg",
                    np.vstack((catalog_3dhst["ra"],catalog_3dhst["dec"])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=red",header='fk5',comments='')

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/3DHST.matched.reg",
                    np.vstack((catalog["RA"][m1],catalog["DEC"][m1])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=green",header='fk5',comments='')

    return catalog

def match_ned(catalog):

    ned_catalog = np.genfromtxt('/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/NED_search.txt', delimiter='|',
                                    dtype=[('ID',int),('Object',np.object),('RA',float),('DEC',float),('Type',np.object),
                                           ('Velocity',int),('z',float),('Flag','<U8'),('Magnitude','<U5'),('Distance',float),
                                           ('Ref',int),('Notes',int),('Phot_pts',int),('Positions',int),('z_pts',int),('Dia_pts',int),('Assc',int)])

    m1,m2,d12 = match_ra_dec(catalog['RA'],catalog['DEC'],ned_catalog['RA'],ned_catalog['DEC'])
    cond = (m2 != len(ned_catalog))
    m1, m2 = m1[cond], m2[cond]
    print ('NED Search: Matched %i out of %i sources (%i available)' % (len(m1), len(catalog), len(ned_catalog)))

    catalog['NED_ID'][m1]   = ned_catalog['ID'][m2]
    catalog['NED_RA'][m1]   = ned_catalog['RA'][m2]
    catalog['NED_DEC'][m1]  = ned_catalog['DEC'][m2]
    catalog['NED_z'][m1]    = ned_catalog['z'][m2]
    catalog['NED_MAG'][m1]  = ned_catalog['Magnitude'][m2]

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/ned_search.reg",
                    np.vstack((ned_catalog["RA"],ned_catalog["DEC"])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=red",header='fk5',comments='')

    np.savetxt("/data/highzgal/PUBLICACCESS/SPLASH/CATALOGS/region_files/ned_search.matched.reg",
                    np.vstack((catalog["RA"][m1],catalog["DEC"][m1])).T,
                    fmt="circle(%.10f,%.10f,2\")#color=green",header='fk5',comments='')

    return catalog

if __name__ == '__main__':

    print ("No main() defined.")

from __future__ import print_function
import os, errno, subprocess, time
import numpy as np
import photutils
import esutil
import scipy.spatial
import matplotlib.pyplot as plt
from multiprocessing import Queue, Process, Pool

pix_scale = 0.15

filters = {'hsc'   : ['g','r','i','z','y'],
           'supcam': ['b','v','r','i','z'],
           'cfht'  : ['u',],
           'cfhtls': ['u','g','r','i','z'],
           'uds'   : ['j','h','k'],
           'video' : ['z','y','j','h','ks'],
           'irac'  : ['1','2','3','4']}
instrs = filters.keys()

instr_used_list = ['hsc','supcam','uds','video','cfht','cfhtls','irac']
apersizes = np.array([1,2,3,4,5])

fcolor_dict = {'hsc'   : dict(zip(filters['hsc'   ],plt.cm.Blues_r(  np.linspace(0.1,0.7,len(filters['hsc'   ]))))),
               'supcam': dict(zip(filters['supcam'],plt.cm.Greens_r( np.linspace(0.1,0.7,len(filters['supcam']))))),
               'cfht'  : dict(zip(filters['cfht'  ],['violet',])),
               'cfhtls': dict(zip(filters['cfhtls'],plt.cm.Purples_r(np.linspace(0.1,0.7,len(filters['cfhtls']))))),
               'uds'   : dict(zip(filters['uds'   ],plt.cm.YlOrBr_r( np.linspace(0.1,0.7,len(filters['uds'   ]))))),
               'video' : dict(zip(filters['video' ],plt.cm.Reds_r(   np.linspace(0.2,0.7,len(filters['video' ]))))),
               'irac'  : dict(zip(filters['irac'  ],plt.cm.Greys_r(  np.linspace(0.1,0.6,len(filters['irac'  ])))))}

# pivot_l = {'hsc'    : {'g':0.4816,'r':0.6234,'i':0.7741,'z':0.8912,'y':0.9780},
#            'supcam' : {'b':0.4374,'v':0.5448,'r':0.6509,'i':0.7676,'z':0.9195},
#            'cfht'   : {'u':0.3746},
#            'cfhtls' : {'u':0.3811,'g':0.4862,'r':0.6258,'i':0.7553,'z':0.8871},
#            'uds'    : {'j':1.2556,'h':1.6496,'k':2.2356},
#            'video'  : {'z':0.8779,'y':1.0211,'j':1.2541,'h':1.6464,'ks':2.1488},
#            'irac'   : {'1':3.5573,'2':4.5049,'3':5.7386,'4':7.9274}}

pivot_l = {'hsc_g'   :0.4816, 'hsc_r'   :0.6234, 'hsc_i'   :0.7741, 'hsc_z'   :0.8912, 'hsc_y'   :0.9780,
           'supcam_b':0.4374, 'supcam_v':0.5448, 'supcam_r':0.6509, 'supcam_i':0.7676, 'supcam_z':0.9195,
           'uds_j'   :1.2556, 'uds_h'   :1.6496, 'uds_k'   :2.2356,
           'video_z' :0.8779, 'video_y' :1.0211, 'video_j' :1.2541, 'video_h' :1.6464, 'video_ks':2.1488,
           'cfht_u'  :0.3746,
           'cfhtls_u':0.3811, 'cfhtls_g':0.4862, 'cfhtls_r':0.6258, 'cfhtls_i':0.7553, 'cfhtls_z':0.8871,
           'irac_1'  :3.5573, 'irac_2'  :4.5049, 'irac_3'  :5.7386, 'irac_4'  :7.9274}
sorted_pivot_l = sorted(pivot_l, key=lambda x: pivot_l[x])

fnames,fcolors = [],[]
for instr in instr_used_list:
    for filt in filters[instr]:
        fnames.append("%s_%s" % (instr,filt))
        fcolors.append(fcolor_dict[instr][filt])

zp = 23.93
orig_zp = {'hsc'    : {'g':27.0,'r':27.0,'i':27.0,'z':27.0,'y':27.0},
           #'supcam' : {'b':34.723,'v':33.639,'r':34.315,'i':34.055,'z':33.076},
           'supcam' : {"b":{1:34.723, 2:34.701, 3:34.706, 4:34.698, 5:34.716},
                       "v":{1:33.639, 2:33.648, 3:33.643, 4:33.639, 5:33.649},
                       "r":{1:34.315, 2:34.276, 3:34.219, 4:34.259, 5:34.247},
                       "i":{1:34.055, 2:34.042, 3:34.046, 4:33.986, 5:34.087},
                       "z":{1:33.076, 2:32.278, 3:32.258, 4:32.743, 5:32.776}},
           'cfht'   : {'u':23.9},
           'cfhtls' : {'u':30.0,'g':30.0,'r':30.0,'i':30.0,'z':30.0},
      'conv_cfhtls' : {'u':30.0,'g':30.0,'r':30.0,'i':30.0,'z':30.0},
           'uds'    : {'j':30.0+0.938,'h':30.0+1.379,'k':30.0+1.900},
           'video'  : {'z':30.0,'y':30.0,'j':30.0,'h':30.0,'ks':30.0},
           'irac'   : {'1':21.5814,'2':21.5814,'3':21.5814,'4':21.5814}}

gain = {'hsc'    : {'g':00000,'r':00000,'i':00000,'z':00000,'y':00000},
        'supcam' : {'b':20700,'v':19140,'r':14880,'i':38820,'z':13020},             # from F08 table
        'cfht'   : {'u':2310.211},                                                  # from premosaic headers
        'cfhtls' : {'u':00000,'g':00000,'r':00000,'i':00000,'z':00000},
        'uds'    : {'j':541.274,'h':382.842,'k':640.502},                           # from premosaic headers
        'video'  : {'z':3795.15,'y':3257.70,'j':2787.94,'h':4898.85,'ks':3795.15},  # from premosaic headers
        'irac'   : {'1':00000,'2':00000,'3':00000,'4':00000}}

SUBARU_key = {'IMACS'       :"Higuchi_in_prep",
              'CurtisLake2012': "CurtisLake+12",
              'Matsuoka2016':"Matsuoka+16",
              'Momcheva2016':"Momcheva+16",
              'Ono2017'     :"Ono+17",
              'Ouchi2008'   :"Ouchi+08",
              'Paris2017'   :"Paris+17",
              'Shibuya2017' :"Shibuya+17",
              'Shibuya17b'  :"Shibuya+17",
              'Wang2016'    :"Wang+16",
              'Saito2008'   :"Saito+08",
              'in_prep'     :"Harikane_in_prep",
              'shinogi_tab2':"Higuchi_in_prep"}

XUDS_key = {'A15' :"Akiyama+15",
            'F10' :"Finoguenov+10",
            'O08' :"Ouchi+08",
            'G07' :"Geach+07",
            'S06' :"Simpson+06",
            'S15' :"Santini+15",
            'SIP' :"XUDS_comp",
            'VB07':"vanBreukelen+07",
            'Y05' :"Yamada+05",
            'UDSz':"UDSz_"}

def run(call,cwd,verbose=True):

    print("Running command:<{:s}> in directory:<{:s}> ... ".format(call,cwd))
    start = time.time()
    if verbose:
        p = subprocess.Popen(call, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd, shell=True)
        for line in iter(p.stdout.readline,b""):
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
    else:
        devnull = open(os.devnull, 'w')
        p = subprocess.Popen(call, stdout=devnull, stderr=devnull, cwd=cwd, shell=True)
    p.communicate()
    p.wait()
    return time.time() - start

class AsyncFactory:

    def __init__(self, func, cb_func, nproc=None):
        self.func = func
        self.cb_func = cb_func
        if nproc: self.pool = Pool(processes=nproc)
        else: self.pool = Pool()

    def call(self, *args, **kwargs):
        self.pool.apply_async(self.func, args, kwargs, self.cb_func)

    def wait(self):
        self.pool.close()
        self.pool.join()

def multiprocess(calls,cwd,verbose=False):

    def worker(queue,call):
        run(call,cwd=cwd,verbose=verbose)
        queue.put(None)

    queue = Queue()
    procs = [Process(target=worker, args=(queue,call)) for call in calls]
    for proc in procs: proc.start()

    finished = 0
    while finished < len(procs):
        items = queue.get()
        if items is None: finished+=1

    for proc in procs: proc.join()

def force_symlink(src,dst):
    try:
        os.symlink(src, dst)
    except (OSError, e):
        if e.errno == errno.EEXIST:
            os.remove(dst)
            os.symlink(src, dst)

def delete_file(path):
    if os.path.isfile(path):
        os.remove(path)

def calc_photometry(img,pos,radius):

    aperture   = photutils.CircularAperture(pos,r=radius)
    photometry = photutils.aperture_photometry(img, aperture)
    if len(photometry['aperture_sum'])==1:
        flux   = photometry['aperture_sum'][0]
    else:
        flux   = np.array(photometry['aperture_sum'])
    return flux

def calc_sn(mag,dmag):
    """
    Mag  = -2.5*log(S)
    dMag = -2.5*log(S) + 2.5*log(S+N)
         =  2.5*log((S+N)/S)
         =  2.5*log(1+N/S)
    """
    cond_99 = (dmag==-99.)
    cond_00 = (dmag==0)
    cond_nan = (dmag>10)
    sn = dmag * 0
    sn[cond_99] = -99.
    sn[cond_00] = 9999.
    sn[cond_nan]= -99.
    sn[~cond_00&~cond_99&~cond_nan] = 1./(10**(dmag[~cond_00&~cond_99&~cond_nan]/2.5) - 1)
    sn[np.abs(mag)==99.] = -99.
    return sn

def get_psf_fluxes(psf,radii):

    dim = psf.shape
    pos = [(dim[0]/2,dim[1]/2),]
    fluxes = np.array([calc_photometry(psf,pos,r) for r in radii])
    return fluxes

def get_binsize(data):
    # Freedman-Diaconis Rule
    return 2*(np.percentile(data,75)-np.percentile(data,25)) / len(data)**(1./3.)

def msize_cuts():

    size_cuts = {'hsc'    : {'g':[2.3,3.3],'r':[2.1,3.3],'i':[2.1,2.8],'z':[2.0,3.0],'y':[2.0,3.0]},
                 'supcam' : {'b':[2.8,3.8],'v':[2.8,3.6],'r':[2.8,3.6],'i':[2.7,3.7],'z':[2.9,3.5]},
                 'cfht'   : {'u':[3.2,3.9]},
                 'cfhtls' : {'u':[2.4,4.4],'g':[2.3,3.8],'r':[2.2,3.8],'i':[1.7,3.3],'z':[2.1,3.6]},
                 'uds'    : {'j':[2.8,3.5],'h':[2.8,3.6],'k':[2.6,3.4]},
                 'video'  : {'z':[3.2,3.9],'y':[3.2,4.0],'j':[2.8,3.6],'h':[2.5,3.4],'ks':[2.5,3.6]},
                 'irac'   : {'1':[0.0,99.],'2':[0.0,99.],'3':[0.0,99.],'4':[0.0,99.]}}

    mag_cuts = {'hsc'    : {'g':[19.0,21.5],'r':[19.4,21.0],'i':[19.0,20.5],'z':[18.5,20.0],'y':[18.0,20.0]},
                'supcam' : {'b':[20.3,22.5],'v':[20.3,22.3],'r':[20.3,22.0],'i':[19.8,21.5],'z':[19.5,21.8]},
                'cfht'   : {'u':[16.8,21.6]},
                'cfhtls' : {'u':[17.3,20.7],'g':[18.2,21.4],'r':[17.8,20.5],'i':[18.0,20.0],'z':[16.8,19.4]},
                'uds'    : {'j':[16.0,20.4],'h':[15.5,19.5],'k':[15.0,19.0]},
                'video'  : {'z':[15.6,20.2],'y':[15.0,19.5],'j':[15.5,19.5],'h':[16.0,19.0],'ks':[15.5,18.5]},
                'irac'   : {'1':[00.0,99.0],'2':[00.0,99.0],'3':[00.0,99.0],'4':[00.0,99.0]}}

    return size_cuts, mag_cuts

def calc_m_from_f(f):
    return -2.5*np.log10(f)

def get_PSF_list(psf_dir,normed=False):

    if normed:
        psf_list = np.sort(np.array([x for x in os.listdir(psf_dir) if 'mosaic_' in x and '.stars.norm_psf.fits' in x]))
    else:
        psf_list = np.sort(np.array([x for x in os.listdir(psf_dir) if 'mosaic_' in x and '.stars.psf' in x]))

    sort_list = []
    for instr in instr_used_list:
        for filt in filters[instr]:
            sort_list.append("_%s_%s" % (instr,filt))

    idx = []
    for x in sort_list:
        _idx = np.where([x in i for i in psf_list])[0]
        if len(_idx)>0: idx.append(_idx[0])

    return psf_list[idx]

def psfex_moffat_pars(basis_type):

    if   'orig' in basis_type:

        # moffat_pars = {'hsc'    : {'g':[0.62,2.59],'r':[0.54,2.04],'i':[0.57,2.70],'z':[0.43,1.57],'y':[0.52,2.36]},
        #                'supcam' : {'b':[0.80,3.11],'v':[0.79,3.59],'r':[0.80,3.34],'i':[0.78,3.08],'z':[0.79,3.36]},
        #                'cfht'   : {'u':[0.82,2.50]},
        #                'cfhtls' : {'u':[0.76,3.06],'g':[9.99,9.99],'r':[9.99,9.99],'i':[9.99,9.99],'z':[9.99,9.99]},
        #                'uds'    : {'j':[0.75,2.90],'h':[0.76,2.90],'k':[0.70,2.92]},
        #                'video'  : {'z':[0.71,2.21],'y':[0.74,2.35],'j':[0.74,2.98],'h':[0.74,3.83],'ks':[0.78,3.80]},
        #                'irac'   : {'1':[1.54,2.32],'2':[1.47,2.03],'3':[1.65,1.72],'4':[1.85,1.92]}}

        moffat_pars = {'hsc'    : {'g':[0.63,2.57],'r':[0.55,2.07],'i':[0.58,2.78],'z':[0.45,1.64],'y':[0.55,2.43]},
                       'supcam' : {'b':[0.80,3.19],'v':[0.80,3.65],'r':[0.80,3.36],'i':[0.80,3.18],'z':[0.80,3.34]},
                       'cfht'   : {'u':[0.89,2.92]},
                       'cfhtls' : {"u": {1:[0.78,3.16],2:[0.99,3.66],3:[0.85,2.65],4:[0.77,4.63],5:[0.93,3.73],6:[0.93,3.14],7:[0.90,3.34],8:[0.82,3.61],9:[0.81,2.70]},
                                   "g": {1:[0.81,3.43],2:[0.83,3.21],3:[0.82,4.59],4:[0.82,2.93],5:[0.82,3.39],6:[0.60,2.99],7:[0.86,3.81],8:[0.82,3.62],9:[0.87,3.41]},
                                   "r": {1:[0.62,3.02],2:[0.71,2.69],3:[0.71,2.81],4:[0.85,2.91],5:[0.82,2.98],6:[0.67,2.65],7:[0.82,3.02],8:[0.83,2.78],9:[0.67,2.58]},
                                   "i": {1:[0.53,3.46],2:[0.57,3.31],3:[0.73,3.10],4:[0.63,5.23],5:[0.70,3.09],6:[0.54,4.11],7:[0.60,3.47],8:[0.73,3.08],9:[0.75,3.44]},
                                   "z": {1:[0.73,3.25],2:[0.64,4.22],3:[0.77,2.94],4:[0.70,2.37],5:[0.66,3.23],6:[0.80,2.70],7:[0.61,3.02],8:[0.68,2.75],9:[0.63,3.26]}},
                       'uds'    : {'j':[0.76,2.95],'h':[0.77,3.00],'k':[0.71,3.02]},
                       'video'  : {'z':[0.73,2.25],'y':[0.75,2.40],'j':[0.74,3.05],'h':[0.73,3.91],'ks':[0.76,3.89]}}

    elif 'conv' in basis_type:

        # moffat_pars = {'hsc'    : {'g':[0.68,2.72],'r':[0.66,2.72],'i':[0.68,2.85],'z':[0.67,2.68],'y':[0.72,2.67]},
        #                'supcam' : {'b':[0.75,3.27],'v':[0.75,3.47],'r':[0.76,3.40],'i':[0.74,3.07],'z':[0.75,3.50]},
        #                'cfht'   : {'u':[0.74,2.67]},
        #                'cfhtls' : {'u':[0.74,3.03],'g':[9.99,9.99],'r':[9.99,9.99],'i':[9.99,9.99],'z':[9.99,9.99]},
        #                'uds'    : {'j':[0.72,3.06],'h':[0.73,3.12],'k':[0.70,3.08]},
        #                'video'  : {'z':[0.73,4.77],'y':[0.75,5.06],'j':[0.75,4.95],'h':[0.74,4.49],'ks':[0.77,4.35]},
        #                'irac'   : {'1':[0.70,2.80],'2':[0.70,2.80],'3':[0.70,2.80],'4':[0.70,2.80]}}

        moffat_pars = {'hsc'    : {'g':[0.71,2.98],'r':[0.73,2.92],'i':[0.71,2.94],'z':[0.66,2.72],'y':[0.72,2.85]},
                       'supcam' : {'b':[0.77,3.66],'v':[0.77,3.76],'r':[0.77,3.66],'i':[0.76,3.38],'z':[0.78,3.71]},
                       'cfht'   : {'u':[0.81,3.53]},
                       'cfhtls' : {'u':[0.75,3.25],'g':[0.74,3.54],'r':[0.74,3.31],'i':[0.78,2.68],'z':[0.76,2.95]},
                       'uds'    : {'j':[0.74,3.22],'h':[0.75,3.30],'k':[0.72,3.28]},
                       'video'  : {'z':[0.74,4.84],'y':[0.76,5.14],'j':[0.76,5.02],'h':[0.75,4.18],'ks':[0.77,4.45]}}

    else: raise Exception('Incorrect basis_type in psfex_moffat_pars().')

    return moffat_pars

def match_ra_dec(ra1,dec1,ra2,dec2,crit=0.5,maxmatch=1):

    h = esutil.htm.HTM(10)
    crit = crit/3600. # crit arcsec
    m1,m2,d12 = h.match(ra1,dec1,ra2,dec2,crit,maxmatch=maxmatch)
    return m1, m2, d12

def match_x_y(x1,y1,x2,y2,r,k=1):

    d1 = np.dstack((x1, y1))[0]
    d2 = np.dstack((x2, y2))[0]

    t = scipy.spatial.cKDTree(d2)
    d, idx = t.query(d1, k=k, eps=0, p=2, distance_upper_bound=r)
    return idx

def read_lephare_photoz(fname):

    catalog_zphot = np.genfromtxt(fname,
                            dtype=[('ID','>i4'),('Z_BEST','>f8'),('Z_BEST68_LOW','>f8'),('Z_BEST68_HIGH','>f8'),
                                   ('Z_ML','>f8'),('Z_ML68_LOW','>f8'),('Z_ML68_HIGH','>f8'),
                                   ('CHI_BEST','>f8'),('MOD_BEST','>i4'),('EXTLAW_BEST','>i4'),('EBV_BEST','>f8'),
                                   ('PDZ_BEST','>f8'),('SCALE_BEST','>f8'),('DIST_MOD_BEST','>f8'),
                                   ('NBAND_USED','>i4'),('NBAND_ULIM','>i4'),
                                   ('Z_SEC','>f8'),('CHI_SEC','>f8'),('MOD_SEC','>i4'),
                                   ('Z_QSO','>f8'),('CHI_QSO','>f8'),('MOD_QSO','>i4'),
                                   ('CHI_STAR','>f8'),('MOD_STAR','>i4'),
                                   ('CONTEXT','>i4'),('ZSPEC','>f4')])

    catalog_zphot["CHI_SEC"][catalog_zphot["CHI_SEC"]==1e+9] = 1e+10

    dof = 3

    cond_chi_best = (catalog_zphot["CHI_BEST"] != 1e+10)
    cond_chi_qso  = (catalog_zphot["CHI_QSO"]  != 1e+10)
    cond_chi_star = (catalog_zphot["CHI_STAR"] != 1e+10)
    cond_chi_sec  = (catalog_zphot["CHI_SEC"]  != 1e+10)

    cond_nband = (catalog_zphot["NBAND_USED"] > dof)
    catalog_zphot["CHI_BEST"][cond_chi_best & cond_nband] = catalog_zphot["CHI_BEST"][cond_chi_best & cond_nband] / (catalog_zphot["NBAND_USED"][cond_chi_best & cond_nband] - dof)
    catalog_zphot["CHI_QSO" ][cond_chi_qso  & cond_nband] = catalog_zphot["CHI_QSO" ][cond_chi_qso  & cond_nband] / (catalog_zphot["NBAND_USED"][cond_chi_qso  & cond_nband] - dof)
    catalog_zphot["CHI_STAR"][cond_chi_star & cond_nband] = catalog_zphot["CHI_STAR"][cond_chi_star & cond_nband] / (catalog_zphot["NBAND_USED"][cond_chi_star & cond_nband] - dof)
    catalog_zphot["CHI_SEC" ][cond_chi_sec  & cond_nband] = catalog_zphot["CHI_SEC" ][cond_chi_sec  & cond_nband] / (catalog_zphot["NBAND_USED"][cond_chi_sec  & cond_nband] - dof)

    cond_nband = (catalog_zphot["NBAND_USED"] <= dof)
    catalog_zphot["CHI_BEST"][cond_chi_best & cond_nband] = 1e+10
    catalog_zphot["CHI_QSO" ][cond_chi_qso  & cond_nband] = 1e+10
    catalog_zphot["CHI_STAR"][cond_chi_star & cond_nband] = 1e+10
    catalog_zphot["CHI_SEC" ][cond_chi_sec  & cond_nband] = 1e+10

    return catalog_zphot

def read_lephare_phys(fname):

    catalog_phys = np.genfromtxt(fname,
                            dtype=[('ID','>i4'),
                                   ('Z_BEST','>f8'),('CHI_BEST','>f8'),('MOD_BEST','>i4'),
                                   ('EXTLAW_BEST','>i4'),('EBV_BEST','>f8'),('PDZ_BEST','>f8'),
                                   ('SCALE_BEST','>f8'),('DIST_MOD_BEST','>f8'),
                                   ('NBAND_USED','>i4'),('NBAND_ULIM','>i4'),
                                   ('K_COR','>f8',(28,)),('MAG_ABS','>f8',(28,)),('MABS_FILT','>i4',(28,)),
                                   ('AGE_BEST','>f8'),('MASS_BEST','>f8'),('SFR_BEST','>f8'),
                                   ('SSFR_BEST','>f8'),('LUM_NUV_BEST','>f8'),('LUM_R_BEST','>f8'),('LUM_K_BEST','>f8')])

    dof = 3

    cond_chi_best = (catalog_phys["CHI_BEST"] != 1e+10)

    cond_nband = (catalog_phys["NBAND_USED"] > dof)
    catalog_phys["CHI_BEST"][cond_chi_best & cond_nband] = catalog_phys["CHI_BEST"][cond_chi_best & cond_nband] / (catalog_phys["NBAND_USED"][cond_chi_best & cond_nband] - dof)

    cond_nband = (catalog_phys["NBAND_USED"] <= dof)
    catalog_phys["CHI_BEST"][cond_chi_best & cond_nband] = 1e+10

    return catalog_phys

def view_fields(a, names):
    """
    `a` must be a numpy structured array.
    `names` is the collection of field names to keep.

    Returns a view of the array `a` (not a copy).
    """
    dt = a.dtype
    formats = [dt.fields[name][0] for name in names]
    offsets = [dt.fields[name][1] for name in names]
    itemsize = a.dtype.itemsize
    newdt = np.dtype(dict(names=names,
                          formats=formats,
                          offsets=offsets,
                          itemsize=itemsize))
    b = a.view(newdt)
    return b

def calc_fscale(zp0,zp1=23.93):
    """
    -2.5*log(f1) + zp1 = -2.5*log(f0) + zp0
    f1/f0 = 10**((zp1 - zp0) / 2.5)
    """
    fscale = 10**((zp1 - zp0) / 2.5)
    return fscale

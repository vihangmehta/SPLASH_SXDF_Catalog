import os, argparse
import numpy as np

import useful

cwd = "/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/scamp/"
chk_list = ["FGROUPS","DISTORTION","ASTR_REFERROR2D","ASTR_REFERROR1D","ASTR_CHI2","ASTR_REFSYSMAP"]

def mk_scamp(solve_astrom='Y'):

    def call(instr,filt=None,tile=None):

        if tile:
            cat = "%s/catalog_%s_%s_%i.ldac" % (instr,instr,filt,tile)
            chk_names = ",".join(["%s/tile_%s_%s_%i_%s" % (plots_dir,instr,filt,tile,x.lower()) for x in chk_list])
            solve_photom = 'Y' if solve_astrom=='Y' else 'N'
            xml_name = "%s/tile_%s_%s_%i.xml" % (xml_dir,instr,filt,tile)
        else:
            cat = "%s/catalog_%s.ldac" % (instr,instr)
            chk_names = ",".join(["%s/premosaic_%s_%s" % (plots_dir,instr,x.lower()) for x in chk_list])
            solve_photom = 'N'
            xml_name = "%s/premosaic_%s.xml" % (xml_dir,instr)

        call = "scamp %s " \
               "-MATCH Y -SOLVE_ASTROM %s -SOLVE_PHOTOM %s " \
               "-ASTREF_CATALOG SDSS-R9 " \
               "-CHECKPLOT_TYPE %s " \
               "-CHECKPLOT_NAME %s " \
               "-WRITE_XML Y -XML_NAME %s " \
               "-c config/config.scamp " % (cat,solve_astrom,solve_photom,",".join(chk_list),chk_names,xml_name)

        if instr=='hsc'  : call += "-ASTREFMAG_LIMITS 16,25 "
        if instr=='irac' : call += "-ASTREFMAG_LIMITS 16,21.5 "
        if instr=='video': call += "-DISTORT_DEGREES 7 "
        
        return call

    plots_dir = "plots/before" if solve_astrom!='Y' else "plots"
    xml_dir = "xml/before" if solve_astrom!='Y' else "xml"

    for instr in ['hsc','uds','video','cfht','cfhtls'][:1]:
        useful.run(call(instr=instr),cwd=cwd,verbose=True)

    # calls  = [call(instr='supcam',filt='b',tile=i) for i in np.arange(5)+1]
    # calls += [call(instr='supcam',filt='v',tile=i) for i in np.arange(5)+1]
    # calls += [call(instr='supcam',filt='r',tile=i) for i in np.arange(5)+1]
    # calls += [call(instr='supcam',filt='i',tile=i) for i in np.arange(5)+1]
    # calls += [call(instr='supcam',filt='z',tile=i) for i in np.arange(5)+1]
    # _ = [useful.run(_call,cwd=cwd,verbose=True) for _call in calls]

if __name__ == '__main__':
    
    mk_scamp(solve_astrom='Y')
    mk_scamp(solve_astrom='N')

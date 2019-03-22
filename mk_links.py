import os, argparse

import useful

def mk_links(instr,filt,src_dir,dst_dir,dry_run):

    _filt = 'k' if instr=='video' and filt=='ks' else filt

    file_types = ["img","wht"]

    if instr=='hsc' : file_types.append("omsk")
    if instr=='irac': file_types.append("std")

    for file_type in file_types:

        src = os.path.join(src_dir,"mosaic_%s_%s.%s.fits"%(instr,filt,file_type))
        dst = os.path.join(dst_dir,"mosaic_%s_%s.%s.fits"%(instr,filt,file_type))
        
        if os.path.isfile(src):
            print "[mk_links.py]  Linked: %s to \n%s" % (src,dst)
            if not dry_run: useful.force_symlink(src, dst)
        else:
            print "Warning: No input found for %s:%s (%s filetype)" % (instr,filt,file_type)

if __name__ == '__main__':
    
    cwd = "/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/"
    msc_dir = "/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/scamp/"
    #msc_dir = "/data/highzgal/PUBLICACCESS/SPLASH/MOSAICS/new_workdir/irac_cropped_completed/"
    orig_dir = "/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/data/orig/"
    conv_dir = "/data/highzgal/PUBLICACCESS/SPLASH/PROCESS/data/conv/"

    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--dry",help="dry run",action='store_true')
    args = parser.parse_args()

    for instr in useful.instr_used_list[:-1]:
        for filt in useful.filters[instr]:
            mk_links(instr=instr,filt=filt,src_dir=msc_dir,dst_dir=orig_dir,dry_run=args.dry)

    instr = 'conv_cfhtls'
    for filt in useful.filters['cfhtls']:
        mk_links(instr=instr,filt=filt,src_dir=msc_dir,dst_dir=conv_dir,dry_run=args.dry)
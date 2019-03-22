import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import astropy.io.fits as fitsio

from astropy.table import Table
from astropy.coordinates import Angle
from astropy.wcs import WCS

tile_dir = '/data/highzgal/PUBLICACCESS/SPLASH/DATA/HSC/Tiles/'

def too_close(all_ra,all_dec,xc,yc,dra,ddec,nearby=False):

    dist = np.sqrt((all_ra-xc)**2 + (all_dec-yc)**2)
    
    if nearby:
        cond = dist < 1.5*np.mean([dra,ddec])
    else:
        cond = dist < 0.8*np.mean([dra,ddec])
    
    return cond

def ingest_tiles():

    fig,axes = plt.subplots(2,3,figsize=(17,12),dpi=300)
    fig.subplots_adjust(left=0.04,right=0.98,bottom=0.05,top=0.95)
    axes = axes.flatten()
    axes[-1].set_visible(False)
    axes = axes[:-1]
    colors = plt.cm.gist_rainbow_r(np.linspace(0.2,0.95,len(axes)))

    with open("hsc_tiles.dat","w") as f:

        f.write("#%5s%10s%6s%6s%15s%15s%10s%10s%4s\n"%("filt","name","dx","dy","ra","dec","dra","ddec","flg"))

        for filt,ax,color in zip(['g','r','i','z','y'],axes,colors):

            tile_list = np.sort(np.array([x for x in os.listdir(tile_dir) if "calexp-HSC-%s-"%filt.upper() in x and ".fits" in x]))
            all_ra, all_dec, all_text = np.array([0,]), np.array([0,]), np.array([0,],dtype=object)

            for tile in tile_list:

                sys.stdout.write("\rProcessing %s ... \033[K" % tile)
                sys.stdout.flush()

                fit = fitsio.open(os.path.join(tile_dir,tile))
                hdr = fit[1].header
                wcs = WCS(hdr)

                dx = hdr['NAXIS1']
                dy = hdr['NAXIS2']
                x0,x1 = 0,dx
                y0,y1 = 0,dy

                _ra,_dec = wcs.all_pix2world(np.array([x0,x1]),np.array([y0,y1]),1)
                ra,dec = _ra[1],_dec[0]
                dra  =  _ra[0] -  _ra[1]
                ddec = _dec[1] - _dec[0]
                
                ax.add_patch(Rectangle(xy=[ra,dec],width=dra,height=ddec,facecolor=color,edgecolor='k',alpha=0.25))

                __ra   = 0.5*( _ra[0]+ _ra[1])
                __dec  = 0.5*(_dec[0]+_dec[1])
                __text = tile[13:-5]

                cond = too_close(all_ra,all_dec,__ra,__dec,dra,ddec)

                if any(cond):
                    i = np.where(cond)
                    all_text[i[0]] += "\n%s"%__text
                    overlapping = 1
                else:
                    all_ra   = np.append(all_ra  , __ra  )
                    all_dec  = np.append(all_dec , __dec )
                    all_text = np.append(all_text, __text)
                    overlapping = 0
                
                f.write("%6s%10s%6i%6i%15.6e%15.6e%10.6f%10.6f%4i\n"%(filt,__text,dx,dy,__ra,__dec,dra,ddec,overlapping))
            
            for ra,dec,text in zip(all_ra,all_dec,all_text):
                ax.text(ra,dec,text,color='k',fontsize=5.5,va='center',ha='center')

            ax.set_title("HSC-%s (%i tiles)"%(filt,len(tile_list)),color=color,fontsize=16,fontweight=800)
            ax.set_aspect(1.)
            ax.set_xlim(35.7,33.5)
            ax.set_ylim(-6.0,-3.7)
        
        print 

    fig.savefig("hsc_tiles.png")

def extract_extension(entry,ext,savename,wht_ext=False,tiles_info=None):

    fname = os.path.join(tile_dir,"calexp-HSC-%s-%s.fits"%(entry["filt"].upper(),entry["name"].strip()))

    with fitsio.open(fname, mode='readonly') as hdu:
        primary_header    = hdu[0].header
        extension_data    = hdu[ext].data
        extension_header  = hdu[ext].header
        extension_header += primary_header

        if wht_ext:
            extension_data = 1./extension_data
            extension_data = fix_weights(wht=extension_data,hdr=extension_header,entry=entry,tiles_info=tiles_info)

    if savename:
        fitsio.writeto(savename, extension_data, extension_header, overwrite=True) #output_verify='fix'

def fix_weights(wht,hdr,entry,tiles_info):

    factor = np.ones_like(wht,dtype=int)
    wcs = WCS(hdr)
    
    _tiles = tiles_info[(tiles_info["filt"]==entry["filt"]) & (tiles_info["name"]!=entry["name"]) & (tiles_info["flg"]==0)]
    nearby = too_close(_tiles["ra"],_tiles["dec"],entry["ra"],entry["dec"],entry["dra"],entry["ddec"],nearby=True)

    for i,_tile in enumerate(_tiles[nearby]):
        
        _input_file = os.path.join(tile_dir,"calexp-HSC-%s-%s.fits"%(_tile["filt"].upper(),_tile["name"].strip()))
        _hdr = fitsio.getheader(_input_file,3)
        _wcs = WCS(_hdr)

        ##### full edges
        pad = 6
        x0,x1 = pad, _hdr["NAXIS1"]-pad
        y0,y1 = pad, _hdr["NAXIS2"]-pad
        nx,ny = _hdr["NAXIS1"]*5, _hdr["NAXIS2"]*5

        # Edges of nearby tile (in pixels)
        _e1 = [np.linspace(x0,x1,nx), np.zeros_like(ny)+y0 ]
        _e2 = [np.linspace(x0,x1,nx), np.zeros_like(ny)+y1 ]
        _e3 = [np.zeros_like(nx)+x0 , np.linspace(y0,y1,ny)]
        _e4 = [np.zeros_like(nx)+x1 , np.linspace(y0,y1,ny)]

        # Edges of nearby tile (in WCS)
        __e1 = _wcs.all_pix2world(_e1[0],_e1[1],1)
        __e2 = _wcs.all_pix2world(_e2[0],_e2[1],1)
        __e3 = _wcs.all_pix2world(_e3[0],_e3[1],1)
        __e4 = _wcs.all_pix2world(_e4[0],_e4[1],1)

        # Edges of nearby tile projected on the main tile's WCS
        e1 = np.array(wcs.all_world2pix(__e1[0],__e1[1],1))
        e2 = np.array(wcs.all_world2pix(__e2[0],__e2[1],1))
        e3 = np.array(wcs.all_world2pix(__e3[0],__e3[1],1))
        e4 = np.array(wcs.all_world2pix(__e4[0],__e4[1],1))

        # Adjust for partial pixels
        e1[0],e1[1] = np.ceil(e1[0]), np.floor(e1[1])
        e2[0],e2[1] = np.ceil(e2[0]), np.floor(e2[1])
        e3[0],e3[1] = np.ceil(e3[0]), np.floor(e3[1])
        e4[0],e4[1] = np.ceil(e4[0]), np.floor(e4[1])

        e1 = e1.astype(int)
        e2 = e2.astype(int)
        e3 = e3.astype(int)
        e4 = e4.astype(int)

        # Trim down to only overlapping edges
        e1 = e1[:,((0<=e1[0,:]) & (e1[0,:]<hdr["NAXIS1"]) & (0<=e1[1,:]) & (e1[1,:]<hdr["NAXIS2"]))]
        e2 = e2[:,((0<=e2[0,:]) & (e2[0,:]<hdr["NAXIS1"]) & (0<=e2[1,:]) & (e2[1,:]<hdr["NAXIS2"]))]
        e3 = e3[:,((0<=e3[0,:]) & (e3[0,:]<hdr["NAXIS1"]) & (0<=e3[1,:]) & (e3[1,:]<hdr["NAXIS2"]))]
        e4 = e4[:,((0<=e4[0,:]) & (e4[0,:]<hdr["NAXIS1"]) & (0<=e4[1,:]) & (e4[1,:]<hdr["NAXIS2"]))]

        # Flag all overlapping pixels using the edges
        _factor = np.zeros_like(wht,dtype=int)
        for x,y in zip(e1[0,:],e1[1,:]): _factor[y:,x] = 1
        for x,y in zip(e2[0,:],e2[1,:]): _factor[:y,x] = 1
        for x,y in zip(e3[0,:],e3[1,:]): _factor[y,x:] = 1
        for x,y in zip(e4[0,:],e4[1,:]): _factor[y,:x] = 1
        
        factor += _factor
        #####

        ##### just corners
        # _x0,_x1 = 0,_hdr["NAXIS1"]-1
        # _y0,_y1 = 0,_hdr["NAXIS2"]-1

        # _ra,_dec = _wcs.all_pix2world(np.array([_x0,_x1]),np.array([_y0,_y1]),1)
        # [x0,x1],[y0,y1] = wcs.all_world2pix(_ra,_dec,1)

        # [x0,x1] = np.clip([x0,x1],0,hdr["NAXIS1"]-1).astype(int)
        # [y0,y1] = np.clip([y0,y1],0,hdr["NAXIS2"]-1).astype(int)

        # factor[y0:y1,x0:x1] = factor[y0:y1,x0:x1] + 1
        #####
        
        print "%10s%10s%4i/%1i" % (entry["name"],_tile["name"],i+1,np.sum(nearby))

    wht = wht / factor
    return wht

def setup_tiles():
    
    tiles_info = np.genfromtxt("hsc_tiles.dat",
                                dtype=[("filt",np.object),("name",np.object),
                                       ("dx",int),("dy",int),
                                       ("ra",float),("dec",float),
                                       ("dra",float),("ddec",float),
                                       ("flg",int)])
    tiles_info = tiles_info[tiles_info["flg"]==0]

    # cond = [(x["filt"]=='g') for x in tiles_info]
    # tiles_info = tiles_info[cond]

    for i,tile in enumerate(tiles_info):
    
        print "%s -> tile_hsc_%s_%s.*.fits (%3i/%3i)"%(tile["name"],tile["filt"],tile["name"],i+1,len(tiles_info))
        extract_extension(entry=tile, ext=1, savename='hsc/tile_hsc_%s_%s.img.fits'%(tile["filt"],tile["name"].replace(',','.')))
        extract_extension(entry=tile, ext=2, savename='hsc/tile_hsc_%s_%s.msk.fits'%(tile["filt"],tile["name"].replace(',','.')))
        extract_extension(entry=tile, ext=3, savename='hsc/tile_hsc_%s_%s.wht.fits'%(tile["filt"],tile["name"].replace(',','.')), wht_ext=True, tiles_info=tiles_info)

if __name__ == '__main__':
    
    # ingest_tiles()
    setup_tiles()
    # plt.show()
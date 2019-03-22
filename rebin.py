import numpy as np
import scipy.ndimage
from scipy import stats
from collections import Counter

def rebin(img,factor,mode):

    if isinstance(factor,float):
        if mode=='mean':
            # print "Factor provided is a float."
            return scipy.ndimage.zoom(img,1./factor)
        elif mode=='sum':
            # print "Factor provided is a float."
            img = scipy.ndimage.zoom(img,1./factor)
            img = img / np.sum(img)
            return img
        else:
            raise Exception("Only mode='mean' supported for floating factors.")

    if isinstance(factor,int):
        factor = np.array([factor,factor])
    else:
        try:
            _,_ = factor
        except TypeError:
            raise Exception("Provide a list,array or tuple for factor.")

    n = np.asarray(img.shape) / np.asarray(factor)

    if mode=='mean':
        rebin_img = img.reshape(n[0],factor[0],n[1],factor[1]).mean(1).mean(2)
    elif mode=='sum':
        rebin_img = img.reshape(n[0],factor[0],n[1],factor[1]).sum(1).sum(2)
    elif mode=='max':
        rebin_img = img.reshape(n[0],factor[0],n[1],factor[1]).max(1).max(2)
    elif mode=='mode':
        _img = img.reshape(n[0],factor[0],n[1],factor[1]).swapaxes(1,2).reshape(n[0],n[1],-1)
        rebin_img = stats.mode(_img,axis=-1)[0].sum(-1)
    elif mode=='mode-fix':
        _img = img.reshape(n[0],factor[0],n[1],factor[1]).swapaxes(1,2).reshape(n[0],n[1],-1)
        rebin_img = stats.mode(_img,axis=-1)[0].sum(-1)

        orig_objs  = np.unique(img)
        rebin_objs = np.unique(rebin_img)
        miss_objs  = orig_objs[~np.in1d(orig_objs,rebin_objs)]

        for miss_obj in miss_objs:
            idx = np.where(_img==miss_obj)
            ctr = Counter(zip(idx[0],idx[1]))
            _idx = [x for x,v in zip(ctr.keys(),ctr.values()) if v==np.max(ctr.values())]
            for __idx in _idx:
                if rebin_img[__idx]==0: rebin_img[__idx] = miss_obj

        # ix,iy = np.where((np.count_nonzero(_img,axis=-1) >= 2) & (rebin_img == 0))
        # cond = np.array([np.in1d(_img[_ix,_iy,:],miss_objs).any() for _ix,_iy in zip(ix,iy)],dtype=bool)
        # ix,iy = ix[cond],iy[cond]
        # rebin_img[ix,iy] = np.max(_img[ix,iy,:],axis=-1)

        cond = np.in1d(orig_objs,np.unique(rebin_img))
        if not cond.all():
            print ("Warning: %i (out of %i) objects not recovered in rebinned seg map." % (len(orig_objs[~cond]),len(orig_objs)))

    else:
        raise Exception("Invalid mode. Choose from [sum,mean,max]")
    return rebin_img

def upbin(img,factor,mode):

    if mode not in ['sci','rms']:
        raise Exception("Choose from ['sci','rms'] for mode.")

    if isinstance(factor,int):
        factor = np.array([factor,factor])
    else:
        try:
            _,_ = factor
        except TypeError:
            raise Exception("Provide a list,array or tuple for factor.")

    if not (isinstance(factor[0],int) and isinstance(factor[1],int)):
        raise Exception("Provide an integer for factor.")

    new_shape = np.asarray(img.shape) * np.asarray(factor)

    if   mode == 'sci':

        img = np.repeat(img[:,:,  np.newaxis],factor[0],axis=-1) / float(factor[0])
        img = np.repeat(img[:,:,:,np.newaxis],factor[1],axis=-1) / float(factor[1])
        img = img.swapaxes(1,2)
        img = img.reshape(new_shape)

    elif mode == 'rms':

        img = np.repeat(img[:,:,  np.newaxis],factor[0],axis=-1) / np.sqrt(float(factor[0]))
        img = np.repeat(img[:,:,:,np.newaxis],factor[1],axis=-1) / np.sqrt(float(factor[1]))
        img = img.swapaxes(1,2)
        img = img.reshape(new_shape)

    return img

if __name__ == '__main__':

    print ("No main() defined.")

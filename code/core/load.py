import numpy as np
import tifffile
import mat73
import json
import os



def load_data(path:str, src:str, datatype:str = 'LF', normalize:bool = True, ret_max_val:bool = False, ret_meta = False):
    """
    Load data from file.
    return: 
        (ret_meta == False)data: targets, Hs(, maxval)(, views)
        (ret_meta == True)meta
    """
    if datatype == 'LF':        # LF dataset (single scene)
        out = load_LF(path, src, normalize = normalize, ret_meta = ret_meta)
    elif datatype == 'LF-MS':   # multi-scene LF dataset
        out = load_LF_MS(path, src, normalize=normalize, ret_meta = ret_meta)
    else:
        raise NotImplementedError("Dataset type not supported!")
    if ret_meta:
        return out
    else:
        return out[:2]+out[3:] if not ret_max_val else out


def load_LF(basedir:str, src:str or dict, normalize:bool = True, ret_meta:bool = False):
    """
    Load default LF data.
    basedir: basedir
    src: (str)meta filename | (dict)scene meta
    normalize: normalize input images by their max value or not
    ret_meta: return meta rather than data for online fetching when training
    return:
        data: targets, Hs, maxval(, views)
    """
    meta = {}
    if type(src) is str:
        with open(os.path.join(basedir, src), 'r') as fp: meta_raw = json.load(fp)
    elif type(src) is dict:
        meta_raw = src
    else:
        raise TypeError("Type of argument:src should be str or dict")


    if 'stack_path' in meta_raw:
        stack_path = os.path.join(basedir, meta_raw['stack_path'])
    if 'views_path' in meta_raw:
        meta['views'] = os.path.join(basedir, meta_raw['views_path'])

    if 'H_path' in meta_raw:
        meta['Hs'] = os.path.join(basedir, meta_raw['H_path'])
    if 'H_cuts' in meta_raw:
        H_centers = np.array(meta_raw['H_cuts']['centers'], dtype=np.int32)
        H_radius = int(meta_raw['H_cuts']['radius'])
        meta['H_cuts'] = H_centers, H_radius

    if 'Ht_path' in meta_raw:
        meta['Hts'] = os.path.join(basedir, meta_raw['Ht_path'])
    elif 'Hs' in meta:
        meta['Hts'] = meta['Hs']


    if 'Ht_cuts' in meta_raw:
        H_centers = np.array(meta_raw['Ht_cuts']['centers'], dtype=np.int32)
        H_radius = int(meta_raw['Ht_cuts']['radius'])
        meta['Ht_cuts'] = H_centers, H_radius
    elif 'H_cuts' in meta and 'Ht_cuts' not in meta:
        meta['Ht_cuts'] = meta['H_cuts']


    if 'targets_path' in meta_raw:
        meta['targets'] = os.path.join(basedir, meta_raw['targets_path'])
    else:
        if 'stack_path' in locals(): meta['targets'] = stack_path

    if 'ssim' in meta_raw:
        meta['ssim'] = {
            'Hs': os.path.join(basedir, meta_raw['ssim']['H_path']),
            'targets': os.path.join(basedir, meta_raw['ssim']['targets_path'])
        }
        if 'H_cuts' in meta_raw['ssim']:
            H_centers = np.array(meta_raw['ssim']['H_cuts']['centers'], dtype=np.int32)
            H_radius = int(meta_raw['ssim']['H_cuts']['radius'])
            meta['ssim']['H_cuts'] = H_centers, H_radius


    if ret_meta:
        return meta
    else: 
        data = get_data_from_meta(meta, normalize=normalize, ret_maxval=True)
        if 'views' in data: return data['targets'], data['Hs'], data['maxval'], data['views']
        else: return data['targets'], data['Hs'], data['maxval']

def load_LF_MS(basedir:str, src:str, normalize:bool = True, ret_meta:bool = False):
    """
    Load multi-scene LF data.
    basedir: basedir
    src: meta filename
    normalize: normalize input images by their max value or not
    return:
        data: targets, Hs, maxval(, views)
    """
    metas = []
    with open(os.path.join(basedir, src), 'r') as fp: metas_raw = json.load(fp)
    for scene_meta in metas_raw['scenes']:
        metas.append(load_LF(basedir, scene_meta, normalize=normalize, ret_meta=True))
    if ret_meta:
        return metas
    else:
        targets_ms = []
        H_mtxs_ms = []
        maxval_ms = []
        views_ms = []
        for meta in metas:
            data = get_data_from_meta(meta, normalize=normalize, ret_maxval=True)
            targets_ms.append(data['targets'])
            H_mtxs_ms.append(data['Hs'])
            maxval_ms.append(data['maxval'])
            if 'views' in data: views_ms.append(data['views'])
        targets_ms = np.array(targets_ms, dtype=object)
        H_mtxs_ms = np.array(H_mtxs_ms, dtype=object)
        maxval_ms = np.array(maxval_ms, dtype=object)

        assert len(np.unique(np.array([targets_ms.shape[-1] for targets_ms in targets_ms]))) == 1,  "#Channels of the images should be the same!"
        if len(views_ms) > 0:
            views_ms = np.array(views_ms, dtype=object)
            assert len(np.unique(np.array([views_ms.shape[-1] for views_ms in views_ms]))) == 1,    "#Channels of the images should be the same!"
            return targets_ms, H_mtxs_ms, maxval_ms, views_ms 
        else:        
            return targets_ms, H_mtxs_ms, maxval_ms

def get_data_from_meta(meta:dict, normalize:bool=True, ret_maxval:bool=True, factor:(float or None)=None):
    """
    Read data from path provided by meta.
    return:
        data{'targets', 'Hs'(, 'maxval'(, 'views'))}
    """
    out = {}
    # get targets
    if 'targets' in meta:
        targets, maxval = readtif(meta['targets'], normalize = normalize)
        targets = targets[...,None] if len(targets.shape)==3 else targets   # (N,H,W,C)
        if factor != None:
            H, W = targets.shape[1:3]
            targets = targets[:, H // 2 - round(factor * H / 2):H // 2 + round(factor * H / 2),
                  W // 2 - round(factor * W / 2):W // 2 + round(factor * W / 2)]
    # get views
    if 'views' in meta:
        views, _ = readtif(meta['views'], normalize = normalize)
        views = views[...,None] if len(views.shape)==3 else views       # (N,H,W,C)
        if factor != None:
            H,W = views.shape[1:3]
            views = views[:,H//2-round(factor*H/2):H//2+round(factor*H/2),W//2-round(factor*W/2):W//2+round(factor*W/2)]
        out['views'] = views

    H_centers, H_radius = meta.get('H_cuts', (None, None))
    # get H matrice
    H_mtxs = getH(meta['Hs'], H_centers, H_radius)

    # get Ht matrice
    # if 'Hts' in meta:
    #     Ht_centers, Ht_radius = meta.get('Ht_cuts', (None, None))
    #     out['Hts'] = getHt(meta['Hts'], Ht_centers, Ht_radius)
    
    if 'targets' in locals():
        N,H,W,C = targets.shape
        Nm,D,Hm,Wm = H_mtxs.shape
        assert N == Nm ,"#images and #H_mtxs not matching!"
        out['targets'] = targets
        if ret_maxval: out['maxval'] = maxval
    out["Hs"] = H_mtxs

    if 'ssim' in meta:
        out['ssim'] = get_data_from_meta(meta['ssim'], normalize=normalize, ret_maxval=ret_maxval, factor=factor)

    return out






def readtif(path:str, normalize:bool = True, dtype = np.float32):
    """
    read tif or tiff image stack from file.
    Returns:
        out: image stack in ndarray; (N,H,W)
        maxval: max value in image, used for normalizing; float
    """
    out = tifffile.imread(path).astype(dtype)
    if len(out.shape) == 2:
        out = out[None,...]     # (N,H,W)
    # normalize
    if normalize:
        maxval = np.max(out)
        out /= maxval
    else:
        maxval = None
    return out, maxval

def getH(path:str, center=None, radius:int=None):
    """
    Get H for forward projection.
    """
    if path.endswith('.mat'):
        mat = mat73.loadmat(path)
        for key in mat:
            raw = mat[key]  # (H,W,D)
            break
    elif path.endswith(('.tif','.tiff')):
        raw = tifffile.imread(path)
    elif path.endswith(('.npz','.npy')):
        raw = np.load(path)
    else:
        raise ValueError("Not supported for this format, please use .tif/.tiff/.mat")
    if center is not None and radius is not None:
        print(f"Read {raw.shape} integrated raw H matrix from {path}")
        center = np.array(center, dtype=np.int32)
        radius = int(radius)
        Hs = []
        for cc in center:
            H = (raw[cc[0]-radius:cc[0]+radius+1,cc[1]-radius:cc[1]+radius+1,:]).astype(np.float32)
            H = H.transpose(2,0,1)
            Hs.append(H)
        H = np.array(Hs, dtype=np.float32)  # (N,D,H,W)
    else:
        # print(f"Read {raw.shape} H matrix from {path}")
        H = raw
    H = H.astype(np.float32)
    # print(f"H: {H.shape} {H.dtype}")
    return H

def getHt(path:str=None, center=None, radius:int=None, isHt:bool=False):
    """
    Get reversed H for back projection.
    isHt: the input data is Ht matrix or H matrix
    """
    if path.endswith('.mat'):
        mat = mat73.loadmat(path)
        for key in mat:
            raw = mat[key]  # (H,W,D)
            break
    elif path.endswith(('.tif','.tiff')):
        raw = tifffile.imread(path)
    elif path.endswith(('.npz','.npy')):
        raw = np.load(path)
    else:
        raise ValueError("Not supported for this format, please use .tif/.tiff/.mat")
    if center is None or radius is None:
        Hts = raw           # should be (N,D,H,W)
    else:
        center = np.array(center, dtype=np.int32)
        radius = int(radius)
        Hts = []
        rawt = np.rot90(raw, 2, (0,1)) if not isHt else raw
        for cc in center:
            Ht = (rawt[cc[0]-radius:cc[0]+radius+1,cc[1]-radius:cc[1]+radius+1,:]).astype(np.float32)
            Ht = Ht.transpose(2,0,1)
            Hts.append(Ht)
    Ht = np.array(Hts, dtype=np.float32)[:,::-1,...].copy()     # (N,D,H,W)
    return Ht

def Normalize_data(x,is_clip=False,cast_bitdepth=16):

    if is_clip:
        x[x>1.0]=1.0
        x[x<0]=0
    x = (x - x.min()) / (x.max()- x.min()+1e-7)
    if cast_bitdepth==16:
        max_=65535
        x = np.array(x*max_,np.uint16)
    elif cast_bitdepth==8:
        max_=255
        x = np.array(x * max_, np.uint8)
    else:
        x = x
    return x

import torch
import numpy as np



def forward_project(realspace:torch.Tensor, H_mtx:torch.Tensor, padding:str='same'):
    """
    Perform forward projection.
    realspace: object space volume voxel
    H_mtx: H matrix used for forward projection. (Related to PSF)
    padding: padding added to each dim and each side. See torch.nn.functional.pad().
    """
    D,H,W,C = realspace.shape
    Dm,Hm,Wm = H_mtx.shape
    assert D == Dm, "The Depth of realspace and H_matrix is not matched!"
    realspace_padded = realspace
    try:
        realspace_padded = torch.nn.functional.pad(realspace_padded, pad=padding, value=0)
        padding = 'valid'
    except: pass
    # out = torch.nn.functional.conv2d(realspace_padded, H_mtx[None,...], None, padding='valid')
    realspace_perm = realspace_padded.permute([-1,0,1,2])   # (C,D,H,W)
    H_perm = H_mtx[None,...]                                # (1,Dm,Hm,Wm)
    out = torch.nn.functional.conv2d(realspace_perm, H_perm, None, padding=padding).permute([1,2,3,0])[0]   # (H,W,C)
    return out

def DeConv(projections:torch.Tensor, Hts:torch.Tensor, weights:torch.Tensor or int = 1):
    """
    Perform back projection, by deconvolution.
    """
    N,H,W,C = projections.shape
    Nm,Dm,Hm,Wm = Hts.shape
    assert N == Nm, "#Hts and #projections is not matched!"
    weights = torch.tensor([weights/N], device=projections.device)
    projections_perm = (projections * weights.reshape([-1,1,1,1])).permute([3,0,1,2]) # (C,N,H,W)
    Hts_perm = Hts.permute([1,0,2,3])   # (D,N,H,W)
    realspace = torch.nn.functional.conv2d(projections_perm, Hts_perm, None, padding='same')    # (C,D,H,W)
    realspace = realspace.permute([1,2,3,0])    # (D,H,W,C)
    return realspace



if __name__=='__main__':
    import tifffile
    import os
    import json
    from tqdm import trange
    # from glbSettings import *
    # from core.load import getH, getHt


    print("Run rendering on CUDA: ", torch.cuda.is_available())
    torch.set_default_tensor_type('torch.cuda.FloatTensor') if torch.cuda.is_available() else torch.set_default_tensor_type("torch.FloatTensor")
    with torch.no_grad():

        H_line_path = './Shifted_Crop201/Hs.npy'
        H_path = "./Shifted_Crop201/Hs.npy"

        has_enhanced_view= False
        root_dir =  r'E:\YCQ\OceanCode_benchmark\data\LF\Zebrafish_27view_REP'
        ori_view_dir= r'./view_spikings_rep/2'
        deblur_view_dir= r'./view_spikings_rep/2/'
        save_path = r'E:\YCQ\OceanCode_benchmark\data\LF\Zebrafish_27view_REP'
        metas = {"scenes":[]}


        if has_enhanced_view:
            for f_ori,f_de in zip(sorted(os.listdir(os.path.join(root_dir,ori_view_dir))),sorted(os.listdir(os.path.join(root_dir,deblur_view_dir)))):
                if ('.tif' in f_ori) and ('.tif' in f_de):
                    ori_path=os.path.join(ori_view_dir,f_ori)
                    de_path =os.path.join(deblur_view_dir,f_de)
                    meta = {
                            "targets_path": ori_path,
                            "H_path": H_path,
                            "ssim": {
                                "targets_path": de_path,
                                "H_path":H_line_path
                            }
                            }
                    metas["scenes"].append(meta)
            metas_js = json.dumps(metas, indent=4)
            with open(os.path.join(save_path, "transforms_train.json"), "w") as f:
                f.write(metas_js)
        else:
            for f_ori in sorted(os.listdir(os.path.join(root_dir,ori_view_dir))):
                if ('.tif' in f_ori) :
                    ori_path=os.path.join(ori_view_dir,f_ori)
                    meta = {
                            "targets_path": ori_path,
                            "H_path": H_path,
                            }
                    metas["scenes"].append(meta)
            metas_js = json.dumps(metas, indent=4)
            with open(os.path.join(save_path, "transforms_train.json"), "w") as f:
                f.write(metas_js)


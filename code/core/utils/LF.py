import torch
import numpy as np
from core.utils.fft_conv import fft_conv
import torchvision.transforms.functional as TF
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
    # out = torch.nn.functional.conv2d(realspace_perm, H_perm, None, padding=padding).permute([1,2,3,0])[0]   # (H,W,C)
    out = fft_conv(realspace_perm, H_perm, None, padding=padding).permute([1, 2, 3, 0])[0]  # (H,W,C)
    return out


def forward_project_Transpose(realspace:torch.Tensor, H_mtx:torch.Tensor, padding:str='same'):
    """
    Perform forward projection.
    realspace: object space volume voxel
    H_mtx: H matrix used for forward projection. (Related to PSF)
    padding: padding added to each dim and each side. See torch.nn.functional.pad().
    """

    H_mtx = TF.rotate(H_mtx, angle=180)

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
    # out = torch.nn.functional.conv2d(realspace_perm, H_perm, None, padding=padding).permute([1,2,3,0])[0]   # (H,W,C)
    out = fft_conv(realspace_perm, H_perm, None, padding=padding).permute([1, 2, 3, 0])[0]  # (H,W,C)
    return out

def back_project(projections:torch.Tensor, Hts:torch.Tensor):
    """
    Perform back projection, by deconvolution.
    """
    N,H,W,C = projections.shape
    Nm,Dm,Hm,Wm = Hts.shape
    assert N == Nm, "#Hts and # projections is not matched!"
    projections_perm = projections.permute([3,0,1,2])    # (C,N,H,W)
    Hts_perm = Hts.permute([1,0,2,3])   # (D,N,H,W)
    realspace = torch.nn.functional.conv2d(projections_perm, Hts_perm, None, padding='same')    # (C,D,H,W)
    realspace = realspace.permute([1,2,3,0]) / N
    return realspace




if __name__=='__main__':
    import tifffile
    import os
    import json
    from tqdm import trange
    from glbSettings import *
    from core.load import getH, getHt


    print("Run rendering on CUDA: ", torch.cuda.is_available())
    torch.set_default_tensor_type('torch.cuda.FloatTensor') if torch.cuda.is_available() else torch.set_default_tensor_type("torch.FloatTensor")
    with torch.no_grad():
        O_dir = r"E:\YCQ\OceanCode_benchmark\code\core\utils/deconv_stack/deconv_iter100_Crop_shifted.tif"

        img_data = tifffile.imread(O_dir)
        img_data = np.asarray(img_data, dtype=np.float32)

        # filp vol:
        # img_data = img_data[::-1,...]

        H_dir = 'E:\YCQ\OceanCode_benchmark\data\LF\Zebrafish_27view\Shifted_Crop201\Hs.npy'
        out_dir = "./out"

        Hs = np.load(H_dir)
        Hs = np.array([H / np.sum(H) for H in Hs], dtype=np.float32)
        Hs = torch.from_numpy(Hs).to(DEVICE)    # (view,D,H,W)

        img_tensor = torch.from_numpy(img_data).to(DEVICE)
        img_tensor = img_tensor[...,None]

        # filp vol:
        # img_tensor = img_tensor.flip(0)
        pass

        prj_list = []
        for v_idx in range(Hs.shape[0]):
            local_psf = Hs[v_idx]
            # prj = forward_project(img_tensor, local_psf)
            prj = forward_project_Transpose(img_tensor, local_psf)
            prj_list.append(prj)
        prj_list = torch.stack(prj_list, dim=0)
        tifffile.imwrite('prj_list_TransHs.tif',np.squeeze(prj_list.cpu().numpy()))
        pass
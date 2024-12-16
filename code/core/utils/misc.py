import torch
import re
import numpy as np
from core.utils.pytorch_ssim import SSIM
from glbSettings import *
from scipy import optimize as sci_op
import imageio

def get_block(pts, model, embedder, post=torch.relu, chunk=32*1024, model_kwargs={'embedder':{}, 'post':{}}):
    sh = pts.shape[:-1]
    pts = pts.reshape([-1,pts.shape[-1]])
    # break down into small chunk to avoid OOM
    if model.type != 'NeRF_ex':
        outs = []
        for i in range(0, pts.shape[0], chunk):
            pts_chunk = pts[i:i+chunk]
            # eval
            out = model.run(pts_chunk, embedder, post, model_kwargs)
            outs.append(out)
        outs = torch.cat(outs, dim=0)
        outs = outs.reshape([*sh,-1])
    else:
        pts_chunk = pts[0:chunk]
        outs = model.run(pts_chunk, embedder, post, model_kwargs)
    return outs

def get_block_v2(px, py, pz, embedder, model, post_processor):
    if embedder is not None:
        _px = embedder.embed(px)
        _py = embedder.embed(py)
        _pz = embedder.embed(pz)
    else:
        _px,_py,_pz = px, py, pz
    block = model.model([_px,_py,_pz])
    block = post_processor(block)
    return block
class Edge_Loss(torch.nn.Module):
    def __init__(self, device):
        super(Edge_Loss, self).__init__()
        kernels =[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
        self.kernel_x=torch.from_numpy(np.asarray(kernels[0],np.float32)).to(device)
        self.kernel_y = torch.from_numpy(np.asarray(kernels[1], np.float32)).to(device)

    def forward(self, pred, target):
        '''
        :param pred: tensor of shape [View, C, H, W]
        '''
        pred = pred[:,0,...].unsqueeze(0)   # b,view, h, w
        target = target[:, 0, ...].unsqueeze(0)

        kernelsX_rep = self.kernel_x.reshape(-1, 1, 3, 3).repeat(pred.shape[1], 1, 1, 1)
        kernelsY_rep = self.kernel_y.reshape(-1, 1, 3, 3).repeat(pred.shape[1], 1, 1, 1)

        pred_edge_x = torch.nn.functional.conv2d(input=pred, weight=kernelsX_rep, padding=1, groups=pred.shape[1])
        pred_edge_y = torch.nn.functional.conv2d(input=pred, weight=kernelsY_rep, padding=1, groups=pred.shape[1])

        target_edge_x = torch.nn.functional.conv2d(input=target, weight=kernelsX_rep, padding=1, groups=pred.shape[1])
        target_edge_y = torch.nn.functional.conv2d(input=target, weight=kernelsY_rep, padding=1, groups=pred.shape[1])

        return (img2mse(pred_edge_x,target_edge_x)+img2mse(pred_edge_y,target_edge_y))/2



def get_regu_value(block,ratio_list=[0.1,0.01],device=DEVICE):
    # regularization
    l1_regu =torch.tensor(0, device=block.device)
    _TV_regu=torch.tensor(0, device=block.device)
    if ratio_list[0] != 0:
        l1_regu = block.abs().mean()
    if ratio_list[1] != 0:
        block_xx = block[:, :, 2:] + block[:, :, :-2] - 2 * block[:, :, 1:-1]
        block_yy = block[:, 2:, :] + block[:, :-2, :] - 2 * block[:, 1:-1, :]
        block_zz = block[2:, :, :] + block[:-2, :, :] - 2 * block[1:-1, :, :]
        block_xy = block[:-1, 1:, 1:] + block[:-1, :-1, :-1] - block[:-1, :-1, 1:] - block[:-1, 1:, :-1]
        block_xz = block[1:, :-1, 1:] + block[:-1, :-1, :-1] - block[:-1, :-1, 1:] - block[1:, :-1, :-1]
        block_yz = block[1:, 1:, :-1] + block[:-1, :-1, :-1] - block[:-1, 1:, :-1] - block[1:, :-1, :-1]
        _TV_regu = block_xx.abs().mean() + block_yy.abs().mean() + block_zz.abs().mean() + 2 * (
                block_xy.abs().mean() + block_xz.abs().mean() + block_yz.abs().mean())
    return ratio_list[0]* l1_regu +  ratio_list[1]* _TV_regu

def makeRectangle(x,y,w,h):
    y = y
    x = x
    return y,x,h,w

def Get_GeoPrj_Matrix(Hs,fitting_range=None):

    '''
    Input tensor: psf matrix [view,d,h,w]
    '''


    def _f_1(x, A, B):
        return A * x + B
    def _f_2(x, A, B):
        return A * x + B

    costumed_psf = np.array([
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ], np.float32)
    costumed_psf = torch.from_numpy(costumed_psf).to(DEVICE)



    Hs = Hs.cpu().detach().numpy() if torch.is_tensor(Hs) else Hs
    view_num,depth,heght,width = Hs.shape
    new_psf = torch.zeros(Hs.shape,device=DEVICE)
    if fitting_range is None:
        fitting_range = [depth//2-8,depth//2+8]
    for _v in range(view_num):
        local_psf = Hs[_v]
        coordinates = []
        loca_new_psf = new_psf[_v]
        for _d in range(local_psf.shape[0]):
            max_idx = np.argmax(local_psf[_d])
            h_idx = max_idx // local_psf.shape[1]
            w_idx = max_idx % local_psf.shape[2]
            z_idx = _d
            if h_idx!=0 and w_idx!=0:
                coordinates.append([h_idx,w_idx,z_idx])

        coordinates = np.asarray(coordinates)
        z1,z2=fitting_range[0],fitting_range[1]
        A_h, B_h = sci_op.curve_fit(_f_1, coordinates[z1:z2,2], coordinates[z1:z2,0])[0]
        A_w, B_w = sci_op.curve_fit(_f_2, coordinates[z1:z2,2], coordinates[z1:z2,1])[0]
        new_h = np.asarray(np.round(np.arange(local_psf.shape[0])*A_h+B_h),np.uint16)
        new_w = np.asarray(np.round(np.arange(local_psf.shape[0])*A_w+B_w),np.uint16)
        for _d in range(local_psf.shape[0]):
            loca_new_psf[_d][new_h[_d],new_w[_d]]=1
        loca_new_psf = torch.nn.functional.conv2d(loca_new_psf.unsqueeze(1),costumed_psf[None,None,...],padding='same')
        new_psf[_v]=loca_new_psf.squeeze(1)

    for _v in range(new_psf.shape[0]):
        new_psf[_v] = new_psf[_v] / torch.sum(new_psf[_v])

    return new_psf


def get_SSIM_map(center_view,center_preds,cut_FOV=485,save_folder=None):
    ssim_map = SSIM(window_size=11, size_average=False)
    # bg_area = makeRectangle(230, 20, 55, 55)
    with torch.no_grad():

        cal_map = ssim_map(center_view.squeeze(-1)[None, None, ...], center_preds.squeeze(-1)[None, None, ...])
        # print(cal_map.mean())
        diff_mapp = torch.ones_like(cal_map) - cal_map - cal_map.mean() * 0.5

        maxHW = max(center_view.shape[0],center_view.shape[1])
        mesh_xx, mesh_yy = np.meshgrid(np.arange(start=-maxHW // 2, stop=maxHW // 2),
                                            np.arange(start=-maxHW // 2, stop=maxHW // 2))
        mask_puipl = np.sqrt(mesh_xx ** 2 + mesh_yy ** 2) >= cut_FOV / 2

        diff_mapp = torch.squeeze(diff_mapp)
        diff_mapp[mask_puipl] = 0
        diff_mapp[diff_mapp < 0] = 0

        saliency_map = diff_mapp
        saliency_map_normalized = saliency_map / torch.sum(saliency_map)
    return saliency_map_normalized,cal_map


img2mse = lambda x, y : torch.mean((x - y) ** 2)
# img2ssim = SSIM(window_size=11)
img2ssim = SSIM(window_size=7)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.tensor([10.], device=DEVICE))
strsort = lambda l: int(re.findall('\d+', l)[0])
get_gradient_loss = Edge_Loss(device=DEVICE)


if __name__=='__main__':
    import tifffile
    import os
    import json
    from tqdm import trange
    from glbSettings import *

    view_num=7
    print("Run rendering on CUDA: ", torch.cuda.is_available())
    torch.set_default_tensor_type('torch.cuda.FloatTensor') if torch.cuda.is_available() else torch.set_default_tensor_type("torch.FloatTensor")
    with torch.no_grad():
        O_dir = r"E:\NetCode\Nerf_test\LF_INR_Final\data\OPT\Simu_apoferritin\projection_viewNum_07.tif"
        img_data = np.asarray(imageio.volread(O_dir), np.float32)
        img_tensor = torch.from_numpy(img_data).type(torch.float32)
        img_tensor = img_tensor[:,None,...].to(DEVICE)

        aa=get_gradient_loss(img_tensor,img_tensor)

        pass
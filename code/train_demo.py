import imageio
import torch
import numpy as np
from tqdm import tqdm, trange
from random import randint
import time
import tifffile
import os
from torch.utils.tensorboard import SummaryWriter
from collections.abc import Iterable

from core.load import get_data_from_meta, load_data
from core.models.Model import search_load_model
from core.utils.LF import *
from core.utils.misc import *
from core.load import Normalize_data
from core.utils.Coord import Coord
from glbSettings import *


def train_Static(Flags, meta: dict = None, models_created=None, dataset_preloaded=None,
                 rg=(lambda start, stop: trange(start, stop, desc='iter')), probability_map=None, writer=None,
                 **kwargs):
    global post_processor_old
    if 'seq_first' in kwargs:
        seq_first = kwargs['seq_first']
    else:
        seq_first = False
    ## data
    dataset = dataset_preloaded
    if dataset is None:
        if meta is None:
            # load data from file
            meta = load_data(Flags.datadir, 'transforms_train.json', datatype=Flags.datatype, normalize=True,
                             ret_max_val=True, ret_meta=True)
        dataset = get_data_from_meta(meta, normalize=True, ret_maxval=True, factor=Flags.centercut)

    targets, H_mtxs = dataset['targets'], dataset['Hs']
    H_mtxs = np.array([H / np.sum(H) for H in H_mtxs], dtype=np.float32)
    N, H, W, C = targets.shape
    Nm, D, Hm, Wm = H_mtxs.shape
    assert N == Nm, "The number of the target images and the H matrice should be the same"
    assert (Hm % 2 == 1 and Wm % 2 == 1), "H matrix should have odd scale"
    if dataset_preloaded is None: print(f'{Flags.datatype} dataset Loaded from {Flags.datadir}. Targets: ',
                                        targets.shape, 'H_mtxs: ', H_mtxs.shape)

    ## Grid building
    br = Flags.block_size // 2
    S = np.array([W, H, D], dtype=np.float32)
    maxHW = max(H, W)
    sc = np.full(3, 2 / (maxHW - 1), dtype=np.float32)
    dc = -((S // 2) * 2) / (maxHW - 1)

    dc[1] *= -1  # flip Y
    sc[1] *= -1

    dc[2] *= -1  # flip Z
    sc[2] *= -1

    Coord.set_idx2glb_scale(sc)
    Coord.set_idx2glb_trans(dc)
    Coord.shape = np.array([D, H, W])

    ## Create grid indice
    idx_glb_all = torch.stack(torch.meshgrid(
        torch.arange(D),
        torch.arange(H),
        torch.arange(W),
        indexing='ij'), dim=-1)  # (D,H,W,3)
    pts_glb_all = Coord.idx2glb(idx_glb_all)

    ## to GPU
    targets = torch.tensor(targets, device=DEVICE)
    H_mtxs = torch.tensor(H_mtxs, device=DEVICE)
    pts_glb_all = pts_glb_all.to(DEVICE)
    # get enhanced_views
    if 'ssim' in dataset:
        dataset['ssim']['Hs'] = torch.tensor(np.stack([H / np.sum(H) for H in dataset['ssim']['Hs']], axis=0),
                                             device=DEVICE)
        dataset['ssim']['targets'] = torch.tensor(dataset['ssim']['targets'], device=DEVICE)

    # Resampling
    DS = Flags.DSfactor
    re_block_size = Flags.block_size // DS + (1 - np.mod(Flags.block_size // DS, 2))
    re_br = re_block_size // DS

    # resample the H matrix and LF views
    Hmds, Wmds = Hm // DS // 2 * 2 + 1, Wm // DS // 2 * 2 + 1
    ds_H_mtxs = torch.nn.functional.interpolate(H_mtxs, size=(Hmds, Wmds), mode='bilinear')
    ds_H_mtxs = torch.stack([H / H.sum() for H in ds_H_mtxs])
    Hds, Wds = H // DS, W // DS
    ds_targets = torch.nn.functional.interpolate(targets[:, None, ..., 0], size=(Hds, Wds), mode='bilinear')[:, 0, ...,
                 None]

    # get the resampled grids
    resample_coordinates = pts_glb_all[:, ::DS, ::DS, :]
    resample_coordinates = resample_coordinates[:, :Hds, :Wds, :]

    # resample the H_line and deblurred views
    if Flags.add_ssim_loss:
        if 'ssim' in dataset:
            # contains enhanced view --> get matrix
            targs_ssim = dataset['ssim']['targets']
            H_mtxs_ssim = dataset['ssim']['Hs']
            ds_H_mtxs_ssim = Get_GeoPrj_Matrix(Hs=ds_H_mtxs)
            ds_targs_ssim = torch.nn.functional.interpolate(targs_ssim[:, None, ..., 0], size=(Hds, Wds),
                                                            mode='bilinear')[:, 0, ..., None]
        else:
            ds_targs_ssim = ds_targets
            ds_H_mtxs_ssim = ds_H_mtxs

    # search model
    if models_created is None:  # bypass when train across scenes
        ## Save config
        os.makedirs(os.path.join(Flags.basedir, Flags.expname), exist_ok=True)
        Flags.append_flags_into_file(os.path.join(Flags.basedir, Flags.expname, 'config.cfg'))  # @@@

        ## Create model
        model, embedder, post_processor, optimizer, start, \
            embedder_args, post_args = search_load_model(Flags, (D, H, W))
    else:
        model, embedder, post_processor, optimizer, start, \
            embedder_args, post_args = models_created

    # temp paras in loop
    if (Flags.postprocessortype == 'relu') and start < 500:
        post_processor_old, post_processor = post_processor, torch.nn.functional.leaky_relu
    l, r, n, f, padl, padr, padn, padf = 0, 0, 0, 0, 0, 0, 0, 0

    ## run
    if Flags.action.lower() == 'train': print("Start training...")
    if writer is None: writer = SummaryWriter(log_dir='%s/%s/Tensorboard' % (Flags.basedir,Flags.expname))

    N_steps_pre = Flags.N_steps_pre
    N_steps_fine = Flags.N_steps_fine
    Nsteps = Flags.N_steps

    boundary_range = [eval(_s) if isinstance(_s, str) else _s for _s in Flags.boundary_range]
    if len(boundary_range) != 0:
        idx_xx_range = np.linspace(start=boundary_range[0] // DS, stop=boundary_range[1] // DS, num=3, dtype=np.uint16)
        idx_yy_range = np.linspace(start=boundary_range[0] // DS, stop=boundary_range[1] // DS, num=3, dtype=np.uint16)
    else:
        idx_xx_range = np.arange(re_br, Wds - re_br, dtype=np.uint16)
        idx_yy_range = np.arange(re_br, Hds - re_br, dtype=np.uint16)

    os.makedirs(os.path.join(Flags.basedir, Flags.expname, 'MSE'), exist_ok=True)
    os.makedirs(os.path.join(Flags.basedir, Flags.expname, 'SSIM'), exist_ok=True)
    os.makedirs(os.path.join(Flags.basedir, Flags.expname, 'recon'), exist_ok=True)

    #################################
    ##          CORE LOOP          ##
    #################################

    print(10 * "-", "Prtraining training", 10 * "-")
    for step in rg(start + 1, N_steps_pre + 1):
        # get random block center (c_idx,c_idy)
        c_idx = idx_xx_range[np.random.randint(len(idx_xx_range))]
        c_idy = idx_yy_range[np.random.randint(len(idx_yy_range))]

        l, r, n, f = c_idx - re_br - Wmds // 2, \
                     c_idx + re_br + Wmds // 2, \
                     c_idy - re_br - Hmds // 2, \
                     c_idy + re_br + Hmds // 2
        padding = padl, padr, padn, padf = (max(0, -l), max(0, r - Wds + 1), max(0, -n), max(0, f - Hds + 1))
        padding = 0, 0, *padding  # C channel should have no padding

        pts_glb_re = resample_coordinates[:, n + padn:f - padf + 1, l + padl:r - padr + 1,
                     :]  # (D,2br+2pady+1,2br+2padx+1,3)
        block_re = get_block(pts_glb_re, model, embedder, post_processor, Flags.chunk)
        targs_re = ds_targets[:,
                   c_idy - re_br:c_idy + re_br + 1,
                   c_idx - re_br:c_idx + re_br + 1,
                   :]
        preds_re = torch.stack([forward_project_Transpose(block_re, _h, padding=padding) for _h in ds_H_mtxs], dim=0)

        # updating weights
        optimizer.zero_grad()
        img_loss = img2mse(preds_re.permute([0, -1, 1, 2]),
                           targs_re.permute([0, -1, 1, 2]))  # view,c,h,w --> view,h,w,c

        ssim_loss = torch.tensor(0, device=img_loss.device)
        if Flags.add_ssim_loss:
            if 'ssim' in dataset:
                preds_ssim_re = torch.stack(
                    [forward_project_Transpose(block_re, _h_ssim, padding=padding) for _h_ssim in ds_H_mtxs_ssim], dim=0)
                targs_ssim_re = ds_targs_ssim[:,
                                c_idy - re_br:c_idy + re_br + 1,
                                c_idx - re_br:c_idx + re_br + 1,
                                :]
            else:
                preds_ssim_re = preds_re
                targs_ssim_re = targs_re

            ssim_loss = (1. - img2ssim(preds_ssim_re.permute([0, -1, 1, 2]), targs_ssim_re.permute([0, -1, 1, 2])))

        loss = img_loss + Flags.ssim_loss_weight * ssim_loss

        psnr = mse2psnr(img_loss)
        loss.backward()
        optimizer.step()

        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = Flags.lrate_decay * 1000
        new_lrate = Flags.lrate * (decay_rate ** ((step - 1) / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        if step == 500 and 'post_processor_old' in locals(): post_processor = post_processor_old

        ## logging
        if (step % Flags.i_weights == 0 and seq_first) or (step % 50 == 0 and not seq_first):
            path = os.path.join(Flags.basedir, Flags.expname)
            model_dict = {
                'global_step': step,
                'network_fn_state_dict': model.get_state(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(model_dict, os.path.join(path, PFX_model + '%06d.tar' % step))
            if hasattr(embedder, 'get_state'): torch.save(embedder.get_state(),
                                                          os.path.join(path, PFX_embedder + '{:06d}.tar'.format(step)))
            if hasattr(post_processor, 'get_state'): torch.save(post_processor.get_state(), os.path.join(path,
                                                                                                         PFX_post + '{:06d}.tar'.format(
                                                                                                             step)))
            print('Saved checkpoints at', path)

        if (step % Flags.i_print == 0 and seq_first) or (step % 50 == 0 and not seq_first):
            with torch.no_grad():

                voxel = get_block(pts_glb_all, model, embedder, post_processor, chunk=Flags.chunk)
                image_rs = torch.stack([forward_project_Transpose(voxel, H_mtx) for H_mtx in H_mtxs], dim=0)
                targs = targets

                loss_avg = img2mse(image_rs.permute([0, -1, 1, 2]), targs.permute([0, -1, 1, 2]))
                psnr_avg = mse2psnr(loss_avg)
                loss_ssim = torch.tensor(0, device=loss_avg.device)
                if 'H_mtxs_ssim' in locals():
                    targs_ssim = dataset['ssim']['targets']
                    image_rs_ssim = torch.stack([forward_project_Transpose(voxel, H_mtx_ssim) for H_mtx_ssim in H_mtxs_ssim])
                    loss_ssim = 1. - img2ssim(image_rs_ssim.permute([0, -1, 1, 2]), targs_ssim.permute([0, -1, 1, 2]))
                else:
                    loss_ssim = 1. - img2ssim(image_rs.permute([0, -1, 1, 2]), targs.permute([0, -1, 1, 2]))
                tqdm_txt = f"[TRAIN] Iter: {step} Loss_fine: {loss_avg.item()} SSIM: {(1 - loss_ssim).item()} PSNR: {psnr_avg.item()}"
                tqdm.write(tqdm_txt)
                # write into file
                path = os.path.join(Flags.basedir, Flags.expname, 'logs.txt')
                with open(path, 'a') as fp:
                    fp.write(tqdm_txt + '\n')
                if Flags.shot_when_print:
                    tifffile.imwrite(os.path.join(Flags.basedir, Flags.expname, 'MSE', f'{(step - 1):06d}.tif'),
                                     Normalize_data(torch.relu(image_rs).cpu().numpy(), cast_bitdepth=16))
                    tifffile.imwrite(os.path.join(Flags.basedir, Flags.expname, 'recon', f'vox_{(step - 1):06d}.tif'),
                                     Normalize_data(torch.relu(voxel).cpu().numpy(), cast_bitdepth=16))
                    if 'image_rs_ssim' in locals():
                        tifffile.imwrite(
                            os.path.join(Flags.basedir, Flags.expname, 'SSIM', f'ssim_{(step - 1):06d}.tif'),
                            Normalize_data(torch.relu(image_rs_ssim).cpu().numpy(), cast_bitdepth=16))

                    # for i in range(len(image_rs)):
                    #     tifffile.imwrite(os.path.join(Flags.basedir, Flags.expname, f'{step-1}_{i}.tif'), torch.relu(image_rs[i]).cpu().numpy())
            del image_rs, voxel, loss_avg, psnr_avg, loss_ssim
        writer.add_scalar(tag="loss/train", scalar_value=loss,
                          global_step=step)
        del loss, img_loss, ssim_loss, psnr, preds_re, block_re
    print(10 * "-", "Prtraining Finished", 10 * "-")
    #################################
    ##          CORE LOOP          ##
    #################################

    print(10 * "-", "Finetuing", 10 * "-")
    boundary_range = [eval(_s) if isinstance(_s, str) else _s for _s in Flags.boundary_range]
    if len(boundary_range) != 0:
        idx_xx_range = np.linspace(start=boundary_range[0], stop=boundary_range[1], num=3, dtype=np.uint16)
        idx_yy_range = np.linspace(start=boundary_range[0], stop=boundary_range[1], num=3, dtype=np.uint16)
    else:
        idx_xx_range = np.arange(br, W - br, dtype=np.uint16)
        idx_yy_range = np.arange(br, H - br, dtype=np.uint16)

    #################################
    ##          CORE LOOP          ##
    #################################
    if start>=N_steps_pre:
        st_second = start
    else:
        st_second = N_steps_pre
    for step in rg(st_second + 1, Nsteps + 1):

        if step <= N_steps_fine:
            c_idx = idx_xx_range[np.random.randint(len(idx_xx_range))]
            c_idy = idx_yy_range[np.random.randint(len(idx_yy_range))]
        else:
            # calculate the erros
            if step % Flags.ErMap_update == 0 or step == N_steps_fine + 1:
                with torch.no_grad():
                    voxel = get_block(pts_glb_all, model, embedder, post_processor, chunk=Flags.chunk)
                    if 'ssim' in dataset:
                        center_view = dataset['ssim']['targets'][3, ...]
                        temp_pred = torch.stack([forward_project_Transpose(voxel, H_mtx_ssim) for H_mtx_ssim in H_mtxs_ssim])
                        center_preds = temp_pred[3, ...]
                    else:
                        center_view = targets[3, ...]
                        temp_pred = torch.stack([forward_project_Transpose(voxel, H_mtx) for H_mtx in H_mtxs], dim=0)
                        center_preds = temp_pred[3, ...]

                    if 'saliency_map_normalized' in locals(): del saliency_map_normalized
                    saliency_map_normalized, _ = get_SSIM_map(center_view, center_preds)
                    # save temporal data
                    if step % 1000 == 0 or step == N_steps_fine + 1:
                        # print('Current iter: %04d, SSIM mean: %.3f' % (step, cal_map.mean()))
                        save_folder = 'map_evaluation_Iter_%05d' % (step)
                        save_folder = os.path.join(Flags.basedir, Flags.expname, save_folder)
                        os.makedirs(save_folder, exist_ok=True)
                        imageio.imwrite(os.path.join(save_folder, 'View_whole.tif'),
                                        np.squeeze(center_view.cpu().numpy()))
                        imageio.imwrite(os.path.join(save_folder, 'Pred_whole.tif'),
                                        np.squeeze(center_preds.cpu().numpy()))
                        imageio.imwrite(os.path.join(save_folder, 'Diff_whole.tif'),
                                        np.squeeze(saliency_map_normalized.cpu().numpy()))

            flat_index = torch.multinomial(saliency_map_normalized.view(-1), 1).item()
            c_idy, c_idx = torch.tensor(
                [flat_index // saliency_map_normalized.shape[1], flat_index % saliency_map_normalized.shape[1]],
                dtype=torch.int64)
            c_idx = torch.clamp(c_idx, br, W - br - 1)  # Note: clamp will produce upper bound
            c_idy = torch.clamp(c_idy, br, H - br - 1)

        # get block
        l, r, n, f = c_idx - br - Wm // 2, c_idx + br + Wm // 2, c_idy - br - Hm // 2, c_idy + br + Hm // 2
        padding = padl, padr, padn, padf = (max(0, -l), max(0, r - W + 1), max(0, -n), max(0, f - H + 1))
        padding = 0, 0, *padding  # C channel should have no padding
        pts_glb = pts_glb_all[:, n + padn:f - padf + 1, l + padl:r - padr + 1, :]  # (D,2br+2pady+1,2br+2padx+1,3)
        block = get_block(pts_glb, model, embedder, post_processor, Flags.chunk)
        targs = targets[:, c_idy - br:c_idy + br + 1, c_idx - br:c_idx + br + 1, :]
        preds = torch.stack([forward_project_Transpose(block, H_mtx, padding=padding) for H_mtx in H_mtxs], dim=0)

        # updating weights
        optimizer.zero_grad()
        img_loss = img2mse(preds.permute([0, -1, 1, 2]), targs.permute([0, -1, 1, 2]))  # view,c,h,w --> view,h,w,c
        ssim_loss = torch.tensor(0, device=img_loss.device)
        if Flags.add_ssim_loss:
            if 'ssim' in dataset:
                H_mtxs_ssim = dataset['ssim']['Hs']
                targs_ssim = dataset['ssim']['targets'][:, c_idy - br:c_idy + br + 1, c_idx - br:c_idx + br + 1, :]
                preds_ssim = torch.stack(
                    [forward_project_Transpose(block, H_mtx_ssim, padding=padding) for H_mtx_ssim in H_mtxs_ssim], dim=0)
            else:
                preds_ssim, targs_ssim = preds, targs

            ssim_loss = (1. - img2ssim(preds_ssim.permute([0, -1, 1, 2]), targs_ssim.permute([0, -1, 1, 2])))

        loss = img_loss + Flags.ssim_loss_weight * ssim_loss + get_regu_value(block, ratio_list=[Flags.L1_regu_weight,
                                                                                                 Flags.TV_regu_weight])

        psnr = mse2psnr(img_loss)
        loss.backward()
        optimizer.step()

        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = Flags.lrate_decay * 1000
        new_lrate = Flags.lrate * (decay_rate ** ((step - 1) / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################
        if step == 500 and 'post_processor_old' in locals(): post_processor = post_processor_old
        ## logging
        if (step % Flags.i_weights == 0 and seq_first) or (step % 50 == 0 and not seq_first):
            path = os.path.join(Flags.basedir, Flags.expname)
            model_dict = {
                'global_step': step,
                'network_fn_state_dict': model.get_state(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(model_dict, os.path.join(path, PFX_model + '{:06d}.tar'.format(step)))
            if hasattr(embedder, 'get_state'): torch.save(embedder.get_state(),
                                                          os.path.join(path, PFX_embedder + '{:06d}.tar'.format(step)))
            if hasattr(post_processor, 'get_state'): torch.save(post_processor.get_state(), os.path.join(path,
                                                                                                         PFX_post + '{:06d}.tar'.format(
                                                                                                             step)))
            print('Saved checkpoints at', path)

        if (step % Flags.i_print == 0 and seq_first) or (step % 50 == 0 and not seq_first):
            with torch.no_grad():
                voxel = get_block(pts_glb_all, model, embedder, post_processor, chunk=Flags.chunk, )
                image_rs = torch.stack([forward_project_Transpose(voxel, H_mtx) for H_mtx in H_mtxs], dim=0)
                targs = targets
                loss_avg = img2mse(image_rs.permute([0, -1, 1, 2]), targs.permute([0, -1, 1, 2]))
                psnr_avg = mse2psnr(loss_avg)
                loss_ssim = torch.tensor(0, device=loss_avg.device)
                if 'H_mtxs_ssim' in locals():
                    targs_ssim = dataset['ssim']['targets']
                    image_rs_ssim = torch.stack([forward_project_Transpose(voxel, H_mtx_ssim) for H_mtx_ssim in H_mtxs_ssim])
                    loss_ssim = 1. - img2ssim(image_rs_ssim.permute([0, -1, 1, 2]), targs_ssim.permute([0, -1, 1, 2]))
                else:
                    loss_ssim = 1. - img2ssim(image_rs.permute([0, -1, 1, 2]), targs.permute([0, -1, 1, 2]))
                tqdm_txt = f"[TRAIN] Iter: {step} Loss_fine: {loss_avg.item()} SSIM: {(1 - loss_ssim).item()} PSNR: {psnr_avg.item()}"
                tqdm.write(tqdm_txt)
                # write into file
                path = os.path.join(Flags.basedir, Flags.expname, 'logs.txt')
                with open(path, 'a') as fp:
                    fp.write(tqdm_txt + '\n')
                if Flags.shot_when_print:
                    tifffile.imwrite(os.path.join(Flags.basedir, Flags.expname, 'MSE', f'{(step - 1):06d}.tif'),
                                     Normalize_data(torch.relu(image_rs).cpu().numpy(), cast_bitdepth=16))
                    tifffile.imwrite(os.path.join(Flags.basedir, Flags.expname, 'recon', f'vox_{(step - 1):06d}.tif'),
                                     Normalize_data(torch.relu(voxel).cpu().numpy(), cast_bitdepth=16))
                    if 'image_rs_ssim' in locals():
                        tifffile.imwrite(
                            os.path.join(Flags.basedir, Flags.expname, 'SSIM', f'ssim_{(step - 1):06d}.tif'),
                            Normalize_data(torch.relu(image_rs_ssim).cpu().numpy(), cast_bitdepth=16))
            del image_rs, voxel, loss_avg, psnr_avg, loss_ssim

        writer.add_scalar(tag="loss/train", scalar_value=loss,
                          global_step=step)
        del loss, img_loss, ssim_loss, psnr, targs, preds, block

    if Flags.action.lower() == 'train': writer.close()


def train_dynamic(Flags, meta: dict = None, models_created=None, dataset_preloaded=None,
                  rg=(lambda start, stop: trange(start, stop, desc='iter')), probability_map=None, writer=None,
                  **kwargs):
    if 'seq_first' in kwargs:
        seq_first = kwargs['seq_first']
    else:
        seq_first = False
    ## data
    dataset = dataset_preloaded
    if dataset is None:
        if meta is None:
            # load data from file
            meta = load_data(Flags.datadir, 'transforms_train.json', datatype=Flags.datatype, normalize=True,
                             ret_max_val=True, ret_meta=True)
        dataset = get_data_from_meta(meta, normalize=True, ret_maxval=True, factor=Flags.centercut)

    targets, H_mtxs = dataset['targets'], dataset['Hs']
    H_mtxs = np.array([H / np.sum(H) for H in H_mtxs], dtype=np.float32)
    N, H, W, C = targets.shape
    Nm, D, Hm, Wm = H_mtxs.shape
    assert N == Nm, "The number of the target images and the H matrice should be the same"
    assert (Hm % 2 == 1 and Wm % 2 == 1), "H matrix should have odd scale"
    if dataset_preloaded is None: print(f'{Flags.datatype} dataset Loaded from {Flags.datadir}. Targets: ',
                                        targets.shape, 'H_mtxs: ', H_mtxs.shape)


    ## Grid building
    br = Flags.block_size // 2
    S = np.array([W, H, D], dtype=np.float32)
    maxHW = max(H, W)
    sc = np.full(3, 2 / (maxHW - 1), dtype=np.float32)
    dc = -((S // 2) * 2) / (maxHW - 1)

    dc[1] *= -1  # flip Y
    sc[1] *= -1

    dc[2] *= -1  # flip Z
    sc[2] *= -1
    Coord.set_idx2glb_scale(sc)
    Coord.set_idx2glb_trans(dc)
    Coord.shape = np.array([D, H, W])

    ## Create grid indice
    idx_glb_all = torch.stack(torch.meshgrid(
        torch.arange(D),
        torch.arange(H),
        torch.arange(W),
        indexing='ij'), axis=-1)  # (D,H,W,3)
    pts_glb_all = Coord.idx2glb(idx_glb_all)

    ## to GPU
    targets = torch.tensor(targets, device=DEVICE)
    H_mtxs = torch.tensor(H_mtxs, device=DEVICE)
    pts_glb_all = pts_glb_all.to(DEVICE)

    if 'ssim' in dataset:
        dataset['ssim']['Hs'] = torch.tensor(np.stack([H / np.sum(H) for H in dataset['ssim']['Hs']], axis=0),
                                             device=DEVICE)
        dataset['ssim']['targets'] = torch.tensor(dataset['ssim']['targets'], device=DEVICE)


    # search model
    if models_created is None:  # bypass when train across scenes
        ## Save config
        os.makedirs(os.path.join(Flags.basedir, Flags.expname), exist_ok=True)
        Flags.append_flags_into_file(os.path.join(Flags.basedir, Flags.expname, 'config.cfg'))  # @@@

        ## Create model
        model, embedder, post_processor, optimizer, start, \
            embedder_args, post_args = search_load_model(Flags, (D, H, W))
    else:
        model, embedder, post_processor, optimizer, start, \
            embedder_args, post_args = models_created

    # post processor
    if (Flags.postprocessortype == 'relu') and start < 500:
        post_processor_old, post_processor = post_processor, torch.nn.functional.leaky_relu

    # argname->arggetter
    l, r, n, f, padl, padr, padn, padf = 0, 0, 0, 0, 0, 0, 0, 0


    os.makedirs(os.path.join(Flags.basedir, Flags.expname, 'MSE'), exist_ok=True)
    os.makedirs(os.path.join(Flags.basedir, Flags.expname, 'SSIM'), exist_ok=True)
    os.makedirs(os.path.join(Flags.basedir, Flags.expname, 'recon'), exist_ok=True)

    #################################
    ##          CORE LOOP          ##
    #################################
    ## run
    if Flags.action.lower() == 'train': print("Start training...")
    if writer is None: writer = SummaryWriter(log_dir='%s/%s/Tensorboard' % (Flags.basedir,Flags.expname))
    Nsteps = Flags.N_steps_seq

    boundary_range = [eval(_s) if isinstance(_s, str) else _s for _s in Flags.boundary_range]
    if len(boundary_range) != 0:
        idx_xx_range = np.linspace(start=boundary_range[0], stop=boundary_range[1], num=3, dtype=np.uint16)
        idx_yy_range = np.linspace(start=boundary_range[0], stop=boundary_range[1], num=3, dtype=np.uint16)
    else:
        idx_xx_range = np.arange(br, W - br, dtype=np.uint16)
        idx_yy_range = np.arange(br, H - br, dtype=np.uint16)

    for step in rg(start + 1, Nsteps + 1):
        c_idx = idx_xx_range[np.random.randint(len(idx_xx_range))]
        c_idy = idx_yy_range[np.random.randint(len(idx_yy_range))]
        # get block
        l, r, n, f = c_idx - br - Wm // 2, c_idx + br + Wm // 2, c_idy - br - Hm // 2, c_idy + br + Hm // 2
        padding = padl, padr, padn, padf = (max(0, -l), max(0, r - W + 1), max(0, -n), max(0, f - H + 1))
        padding = 0, 0, *padding  # C channel should have no padding
        pts_glb = pts_glb_all[:, n + padn:f - padf + 1, l + padl:r - padr + 1, :]  # (D,2br+2pady+1,2br+2padx+1,3)
        block = get_block(pts_glb, model, embedder, post_processor, Flags.chunk)
        targs = targets[:, c_idy - br:c_idy + br + 1, c_idx - br:c_idx + br + 1, :]
        preds = torch.stack([forward_project_Transpose(block, H_mtx, padding=padding) for H_mtx in H_mtxs], dim=0)

        # updating weights
        optimizer.zero_grad()
        img_loss = img2mse(preds.permute([0, -1, 1, 2]), targs.permute([0, -1, 1, 2]))  # view,h,w,c --> view,c,h,w
        ssim_loss = torch.tensor(0, device=img_loss.device)
        if Flags.add_ssim_loss:
            if 'ssim' in dataset:
                H_mtxs_ssim = dataset['ssim']['Hs']
                targs_ssim = dataset['ssim']['targets'][:, c_idy - br:c_idy + br + 1, c_idx - br:c_idx + br + 1, :]
                preds_ssim = torch.stack(
                    [forward_project_Transpose(block, H_mtx_ssim, padding=padding) for H_mtx_ssim in H_mtxs_ssim], dim=0)
            else:
                preds_ssim, targs_ssim = preds, targs

            ssim_loss = (1. - img2ssim(preds_ssim.permute([0, -1, 1, 2]), targs_ssim.permute([0, -1, 1, 2])))
        loss = img_loss + Flags.ssim_loss_weight * ssim_loss + get_regu_value(block, ratio_list=[Flags.L1_regu_weight,
                                                                                                 Flags.TV_regu_weight])

        psnr = mse2psnr(img_loss)
        loss.backward()
        optimizer.step()

        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = Flags.lrate_decay * 1000
        new_lrate = Flags.lrate * (decay_rate ** ((step - 1) / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################
        if step == 500 and 'post_processor_old' in locals(): post_processor = post_processor_old
        ## logging
        if (step % Flags.i_weights == 0 and seq_first) or (step % Flags.i_weights_trans == 0 and not seq_first):
            path = os.path.join(Flags.basedir, Flags.expname)
            model_dict = {
                'global_step': step,
                'network_fn_state_dict': model.get_state(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(model_dict, os.path.join(path, PFX_model + '{:06d}.tar'.format(step)))
            if hasattr(embedder, 'get_state'): torch.save(embedder.get_state(),
                                                          os.path.join(path, PFX_embedder + '{:06d}.tar'.format(step)))
            if hasattr(post_processor, 'get_state'): torch.save(post_processor.get_state(), os.path.join(path,
                                                                                                         PFX_post + '{:06d}.tar'.format(
                                                                                                             step)))
            print('Saved checkpoints at', path)
        if (step % Flags.i_print == 0 and seq_first) or (step % Flags.i_weights_trans == 0 and not seq_first):
            with torch.no_grad():
                voxel = get_block(pts_glb_all, model, embedder, post_processor, chunk=Flags.chunk, )
                image_rs = torch.stack([forward_project_Transpose(voxel, H_mtx) for H_mtx in H_mtxs], dim=0)
                targs = targets
                loss_avg = img2mse(image_rs.permute([0, -1, 1, 2]), targs.permute([0, -1, 1, 2]))
                psnr_avg = mse2psnr(loss_avg)
                loss_ssim = torch.tensor(0, device=loss_avg.device)
                if 'H_mtxs_ssim' in locals():
                    targs_ssim = dataset['ssim']['targets']
                    image_rs_ssim = torch.stack([forward_project_Transpose(voxel, H_mtx_ssim) for H_mtx_ssim in H_mtxs_ssim])
                    loss_ssim = 1. - img2ssim(image_rs_ssim.permute([0, -1, 1, 2]), targs_ssim.permute([0, -1, 1, 2]))
                else:
                    loss_ssim = 1. - img2ssim(image_rs.permute([0, -1, 1, 2]), targs.permute([0, -1, 1, 2]))
                tqdm_txt = f"[TRAIN] Iter: {step} Loss_fine: {loss_avg.item()} SSIM: {(1 - loss_ssim).item()} PSNR: {psnr_avg.item()}"
                tqdm.write(tqdm_txt)
                # write into file
                path = os.path.join(Flags.basedir, Flags.expname, 'logs.txt')
                with open(path, 'a') as fp:
                    fp.write(tqdm_txt + '\n')
                if Flags.shot_when_print:

                    # 20241128
                    tifffile.imwrite(os.path.join(Flags.basedir, Flags.expname, 'MSE', f'{(step - 1):06d}.tif'),
                                     image_rs.cpu().numpy())
                    tifffile.imwrite(os.path.join(Flags.basedir, Flags.expname, 'recon', f'vox_{(step - 1):06d}.tif'),
                                     voxel.cpu().numpy())
                    if 'image_rs_ssim' in locals():
                        tifffile.imwrite(
                            os.path.join(Flags.basedir, Flags.expname, 'SSIM', f'ssim_{(step - 1):06d}.tif'),
                           image_rs_ssim.cpu().numpy())

                    # tifffile.imwrite(os.path.join(Flags.basedir, Flags.expname, 'MSE', f'{(step - 1):06d}.tif'),
                    #                  Normalize_data(torch.relu(image_rs).cpu().numpy(), cast_bitdepth=16))
                    # tifffile.imwrite(os.path.join(Flags.basedir, Flags.expname, 'recon', f'vox_{(step - 1):06d}.tif'),
                    #                  Normalize_data(torch.relu(voxel).cpu().numpy(), cast_bitdepth=16))
                    # if 'image_rs_ssim' in locals():
                    #     tifffile.imwrite(
                    #         os.path.join(Flags.basedir, Flags.expname, 'SSIM', f'ssim_{(step - 1):06d}.tif'),
                    #         Normalize_data(torch.relu(image_rs_ssim).cpu().numpy(), cast_bitdepth=16))



            del image_rs, voxel, loss_avg, psnr_avg, loss_ssim

        writer.add_scalar(tag="loss/train", scalar_value=loss,
                          global_step=step)
        del loss, img_loss, ssim_loss, psnr, targs, preds, block

    if Flags.action.lower() == 'train': writer.close()


def train_seq(Flags):
    """
    Sequential-scene training.
    """
    from copy import deepcopy
    ## load data (T,N,H,W,C), T for time idx

    print(10 * '-', 'Exp: %s' % Flags.expname, 10 * '-')
    print('Data Dir: %s' % Flags.datadir)
    metas = load_data(Flags.datadir, 'transforms_train.json', datatype=Flags.datatype, normalize=True,
                      ret_max_val=True, ret_meta=True)
    # skip
    if Flags.skip > 1:
        metas = metas[::Flags.skip]

    local_normalization = Flags.local_normalization
    dataset_preloaded = []
    for scence_idx, meta in enumerate(metas):
        dataset_preloaded.append(
            get_data_from_meta(meta, normalize=local_normalization, ret_maxval=True, factor=Flags.centercut))
        if not 'ssim' in meta:
            print('[%03d/%03d] Scences:%s Model:%s' % (
                scence_idx, len(metas), os.path.basename(meta['targets']), os.path.basename(meta['Hs'])))
        else:
            print('[%03d/%03d] Scences:%s Model:%s; Enhanced Scence:%s Model:%s' % (scence_idx, len(metas),
                                                                                    os.path.basename(meta['targets']),
                                                                                    os.path.basename(meta['Hs']),
                                                                                    os.path.basename(
                                                                                        meta['ssim']['targets']),
                                                                                    os.path.basename(
                                                                                        meta['ssim']['Hs']),
                                                                                    )
                  )
    # image normalization
    if not local_normalization:
        if 'targets' in dataset_preloaded[0]:
            # maxval_ori = np.max(np.stack([dataset['targets'] for dataset in dataset_preloaded], axis=0),axis=(1,2,3,4))
            maxval_ori = np.max(np.stack([dataset['targets'] for dataset in dataset_preloaded], axis=0))
        if 'ssim' in dataset_preloaded[0]:
            maxval_ssim = np.stack([dataset['ssim']['targets'] for dataset in dataset_preloaded], axis=0).max()
        for dataset in dataset_preloaded:
            if 'targets' in dataset:
                dataset['targets'] = dataset['targets'] / maxval_ori
            if 'ssim' in dataset:
                dataset['ssim']['targets'] = dataset['ssim']['targets'] / maxval_ssim

    T = len(metas)
    C = Flags.sigch
    H, W, D = 0, 0, 0  # H,W,D will acturally never been used since 'Grid' type model is not supported

    ## Save config
    os.makedirs(os.path.join(Flags.basedir, Flags.expname), exist_ok=True)
    Flags.append_flags_into_file(os.path.join(Flags.basedir, Flags.expname, 'config.cfg'))  #

    ## Create model
    models = search_load_model(Flags, (D, H, W),initial_ckpt=True)
    start = models[4]
    lrate = models[3].param_groups[0]['lr']

    # perform standard training process on the first timepoint scene, for a good start point
    if start < Flags.N_steps:
        print(f"Start training on timepoint {0}...")
        Flags_0 = deepcopy(Flags)
        Flags_0.action = 'TRAIN'

        # train static: tri-sampling
        train_Static(Flags_0, meta=metas[0], models_created=models,
                     dataset_preloaded=dataset_preloaded[0],
                     seq_first=True)

        # 2024 05 06 Test random sampling
        # train_dynamic(Flags_0, meta=metas[0], models_created=models,
        #                      dataset_preloaded=dataset_preloaded[0],
        #                      seq_first=True)

        print(f"Timepoint {0} finished.")
        Time_0_path = os.path.join(Flags.basedir, f'{Flags.expname}/timepoint_{0:03d}')
        os.makedirs(Time_0_path, exist_ok=True)
        model, embedder, post_processor, optimizer, start, \
            embedder_args, post_args = models
        model_dict = {
            'global_step': Flags.N_steps,
            'network_fn_state_dict': model.get_state(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(model_dict, os.path.join(Time_0_path, PFX_model + '{:06d}.tar'.format(Flags.N_steps)))

    for t_idx in range(1, T):
        # update
        Flags_t = deepcopy(Flags)
        Flags_t.expname = f'{Flags.expname}/timepoint_{t_idx:03d}'

        # refresh the start and get the initial learning rate
        start = 0
        models = models[:4] + tuple([start]) + models[5:]
        for param_group in models[3].param_groups:
            param_group['lr'] = lrate  # assign the learning rate (the first timepoint)

        # Save config
        os.makedirs(os.path.join(Flags_t.basedir, Flags_t.expname), exist_ok=True)
        Flags_t.append_flags_into_file(os.path.join(Flags_t.basedir, Flags_t.expname, 'config.cfg'))
        # load ckpts: Important!
        models_load = search_load_model(Flags_t, (D, H, W), create_ok=False)
        if models_load != None and models_load[4] != 0:
            models = models_load
        print(f"Start training on timepoint {t_idx}...")
        start = models[4]
        print('Learning rate: %06f' % param_group['lr'])
        ## drop the first and the last layer
        # print('Drop the first and the last layer')
        # torch.nn.init.xavier_uniform_(models[0].get_state()['pts_linear.0.weight'])
        # torch.nn.init.constant_(models[0].get_state()['pts_linear.0.bias'], 0)
        # torch.nn.init.constant_(models[0].get_state()['pts_linear.5.weight'], 0)
        # torch.nn.init.constant_(models[0].get_state()['pts_linear.5.bias'], 0)

        # torch.nn.init.constant_(models[0].get_state()['pts_linear.6.weight'], 0)
        # torch.nn.init.constant_(models[0].get_state()['pts_linear.6.bias'], 0)

        # torch.nn.init.constant_(models[0].get_state()['pts_linear.7.weight'], 0)
        # torch.nn.init.constant_(models[0].get_state()['pts_linear.7.bias'], 0)
        #
        # torch.nn.init.xavier_uniform_(models[0].get_state()['output_linear.weight'])
        # torch.nn.init.constant_(models[0].get_state()['output_linear.bias'], 0)
        if start < Flags.N_steps_seq:
            train_dynamic(Flags_t, meta=metas[t_idx], models_created=models,
                          dataset_preloaded=dataset_preloaded[t_idx])
        print(f"Timepoint {t_idx} finished.")
    print('END.')

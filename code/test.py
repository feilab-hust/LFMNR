import torch
import numpy as np
from tqdm import trange
import tifffile
import os
from core.models.Model import create_model,loading_4D_weights,search_load_model
from core.utils.misc import *
from core.utils.Coord import Coord
from core.load import Normalize_data
from glbSettings import *
rg=(lambda start, stop: trange(start, stop, desc='iter'))

def test(Flags):
    with (torch.no_grad()):
        try:
            H,W,D = Flags.render_size
            H,W,D = int(H),int(W),int(D)
        except:
            raise ValueError("--render_size should be None or in format \"H W D\"!")

        if Flags.render_t_range is not None:
            t_min,t_max = [eval(_s) if isinstance(_s, str) else _s for _s in Flags.render_t_range]
            if t_min > t_max:
                raise ValueError("the t_min should be less than the t_max!")
            assert t_min>=1, "t_min should be greater than or equal to 1!"
            t_min = t_min - 1
            t_max = t_max
            t_seq_enable = 1
        else:
            # print('\"render_t_range\" is None! Defaulting to [1,1]')
            t_min = 0
            t_max = 0
            t_seq_enable = 0

        # build grid
        path = os.path.join(Flags.save_dir, Flags.expname)
        os.makedirs(path, exist_ok=True)
        ## constants
        S = np.array([W,H,D], dtype=np.float32)
        maxHW = max(H,W)
        sc = np.full(3,2/(maxHW-1), dtype=np.float32)
        dc = -((S//2)*2)/(maxHW-1)

        dc[1] *= -1     # flip Y
        sc[1] *= -1

        dc[2] *= -1  # flip Z
        sc[2] *= -1

        Coord.set_idx2glb_scale(sc)
        Coord.set_idx2glb_trans(dc)
        idx_glb_all = torch.stack(torch.meshgrid(
            torch.arange(D),
            torch.arange(H),
            torch.arange(W),
            indexing='ij'), axis=-1)  # (D,H,W,3)
        pts_glb_all = Coord.idx2glb(idx_glb_all)
        pts_glb_all = pts_glb_all.to(DEVICE)

        # seach models
        if Flags.weights_path is not None and Flags.weights_path != 'None':
            ckpts = [Flags.weights_path]
        else:
            ckpts = [os.path.join(Flags.basedir, Flags.expname, f) for f in
                     sorted(os.listdir(os.path.join(Flags.basedir, Flags.expname))) if 'cp_' in f]
        print('Found ckpts', ckpts)
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print('Reloading from', ckpt_path)
            if not t_seq_enable:
                ## fetch the first scene from weights
                ckpts_dict = {'model': ckpt_path}
                model, embedder, post_processor, start, embedder_args, post_args= create_model(Flags,
                                                                                 shape=(D,H,W),
                                                                                 img_ch=Flags.sigch,
                                                                                 model_type=Flags.modeltype,
                                                                                 embedder_type=Flags.embeddertype,
                                                                                 post_processor_type=Flags.postprocessortype,
                                                                                 weights_path=ckpts_dict,
                                                                                 create_optimizer=False)
                voxel = get_block(pts_glb_all, model, embedder, post_processor, chunk=Flags.chunk)
                tifffile.imwrite(os.path.join(path,"Scene_Rendering.tif"),
                                 Normalize_data(torch.relu(voxel).cpu().numpy(),cast_bitdepth=16))
                print('Rendering done!')
            else:

                if Flags.ROI_Box is not None and len(Flags.ROI_Box) !=0:
                    local_coordinates = [eval(_s) if isinstance(_s, str) else _s for _s in Flags.ROI_Box]
                    pt_x=local_coordinates[0]
                    pt_y=local_coordinates[1]
                    pt_z=local_coordinates[2]
                    local_w=local_coordinates[3]
                    local_h=local_coordinates[4]
                    local_d=local_coordinates[5]
                    assert pt_y + local_h - 1 <= H, "out of boundary (y-axis)"
                    assert pt_z + local_d - 1 <= D, "out of boundary (z-axis)"
                    assert pt_x + local_w - 1 <= W, "out of boundary (x-axis)"
                    local_points = pts_glb_all[
                                    pt_z-1:pt_z + local_d - 1,
                                    pt_y-1:pt_y + local_h - 1,
                                    pt_x-1: pt_x + local_w - 1,
                                   ]
                else:
                    local_points = pts_glb_all
                ckpt_path = ckpts[-1]
                Weights_4D_data = torch.load(ckpt_path,map_location=DEVICE)  # loading all weights onto designated device
                for time_idx in rg(t_min,t_max):
                    local_paras = Weights_4D_data[time_idx]
                    model, embedder, post_processor= loading_4D_weights(Flags,
                                                                                     model_type=Flags.modeltype,
                                                                                     embedder_type=Flags.embeddertype,
                                                                                     post_processor_type=Flags.postprocessortype,
                                                                                     weights_paras=local_paras)


                    voxel = get_block(local_points, model, embedder, post_processor, chunk=Flags.chunk)
                    norm_data = Normalize_data(torch.relu(voxel).cpu().numpy(),cast_bitdepth=16) if not Flags.retain_gray else torch.relu(voxel).cpu().numpy()
                    tifffile.imwrite(os.path.join(path, "Scene_Rendering_T_%s.tif"%local_paras['Timepoint']),norm_data)


                print('Rendering done!')
        else:
            print("No ckpts found. Do nothing.")
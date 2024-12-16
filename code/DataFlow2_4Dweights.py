''''
This code was used to package the mutiple trained weights of 3D videos into one 4D-weights

input:
log_path: The folder path that stored the trained weights
desired_ckpt: The iteration number of desired trained weights
t_range: The time range of output packaged weight
output_dir: The folder path where to save output 4D weights

output:
4D weights
'''

import numpy as np
import torch
import os


desired_ckpt = 100
t_range = [0,149]
Log_path = r'G:\LFMNR_official\Github_version\LFMNR\results\Exp_mito_view7_MultiScenes_equal_tile'
output_dir = r'G:\LFMNR_official\Github_version\LFMNR\results\Exp_mito_view7_MultiScenes_equal_tile\4d_ckpt'

time_prefix = 'timepoint'
time_folder_list = ['%s_%03d'%(time_prefix,_s) for _s in range(t_range[0],t_range[1])]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
folder_list = []
for _f in os.listdir(Log_path):
    if time_prefix in _f\
            and _f in  time_folder_list:
        folder_list.append(_f)

saved_weights = []
for _idx,_f in enumerate(folder_list):
    temp_folder = os.path.join(Log_path,_f)

    if _idx == 0:
        ckpt_list = [_ckpt for _ckpt in os.listdir(temp_folder) if ('.tar' in _ckpt)]
        ckpt_list = ckpt_list[-1:]
    else:
        ckpt_list = [_ckpt for _ckpt in os.listdir(temp_folder) if ('.tar' in _ckpt) and ('%d'%desired_ckpt in _ckpt)]

    if len(ckpt_list)!=0:
        ckpt_name = ckpt_list[0]
        tar_file = os.path.join(temp_folder,ckpt_name)
        params1 = torch.load(tar_file)['network_fn_state_dict']
        new_params = {}
        for param in params1:
            diff = params1[param]
            new_params[param] = diff
        new_params1 = {'network_fn_state_dict': new_params}
        new_params1.update({'Timepoint':_f})
        saved_weights.append(new_params1)
        print('Converting %d/%d'%(_idx,len(folder_list)))
    else:
        print('No ckpt found at %s'%_f)
    pass
# np.save(os.path.join(output_dir,'cp_4DWeights.npy'),saved_weights)
torch.save(saved_weights, os.path.join(output_dir,'cp_4DWeights.tar'))
pass
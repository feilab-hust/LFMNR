from glbSettings import *
## Dependences
import torch
import os
from absl import flags, app



## Config Settings
# general
flags.DEFINE_enum('action', 'TRAIN',
                  ['TRAIN','TRAIN-MS','TRAIN-DP','TEST',
                   ],
                  'Action: TRAIN, TRAIN-MS or TEST.')

flags.DEFINE_string('basedir', '', 'Input data path.')
flags.DEFINE_string('expname', 'exp', help='Experiment name.')
flags.DEFINE_string('datadir', './data/LF/exp', 'Input data path.')
flags.DEFINE_string('save_dir', 'results', 'output data path for validation.')


# embedder configs
flags.DEFINE_enum('embeddertype', 'None', ['None','PositionalEncoder','RadiaPosiEncoder'], 'Encoder type.')
# PositionalEncoder
flags.DEFINE_integer('multires', 10, 'log2 of max freq for positional encoding.')
flags.DEFINE_integer('multiresZ', None, 'log2 of max freq for positional encoding on z axis for anisotropic encoding.')

# model configs
flags.DEFINE_enum('modeltype', 'NeRF', ['NeRF', 'NeRF_ex'], 'Model type.')
flags.DEFINE_boolean('no_reload', False, 'Do not reload weights from saved ckpt.')
flags.DEFINE_list('weights_path', None, 'Weight paths to be loaded from: [model_weights(, embedder_weights(, post_processor_weights))]. Should be strings. "None" for ignore.')
flags.DEFINE_integer('sigch', 1, '#channels of the signal to be predicted.')
flags.DEFINE_boolean('compile', False, 'Compile model for faster training using new pytorch 2.0 feature.')
# NeRF-like
flags.DEFINE_integer('netdepth', 8, '#layers.')
flags.DEFINE_integer('netwidth', 256, '#channels per layer.')
flags.DEFINE_list('skips', [4], 'Skip connection layer indice.')


# postprocessor configs
flags.DEFINE_enum('postprocessortype', 'relu', ['linear','relu','leakyrelu'], 'Post processor type.')

# data options
flags.DEFINE_enum('datatype', 'LF', ['LF','LF-MS','OPT','FPM'], 'Dataset type: LF, LF-MS.')
flags.DEFINE_float('centercut', None, 'Only use center part.')
flags.DEFINE_boolean('local_normalization',False,'Whether to act frame-wise normalization.(True:Ti/Ti.max(),False: Ti/T.max()')


#Coarse to Fine training
flags.DEFINE_integer('N_steps_pre', 0, 'Coarse to Fine: Pre-Training Step')
flags.DEFINE_integer('DSfactor', 2, 'Coarse to Fine downsampled factor')
flags.DEFINE_integer('N_steps_fine', 2, 'Coarse to Fine: Fine-training Step')
flags.DEFINE_integer('ErMap_update', 10, 'Coarse to Fine: the Error Map updating step')
flags.DEFINE_list('boundary_range', [100,400], 'Learn rate decay step.')

# training options
flags.DEFINE_integer('N_steps', 10000, '#training steps for each scene.')
flags.DEFINE_integer('N_epochs', 1, '#training epochs (for cross-scene training).')
flags.DEFINE_integer('N_steps_seq', 100, '#training steps for subsequencial scenes (in train-seq).')
flags.DEFINE_float('lrate', 5e-4, 'Learn rate.')
flags.DEFINE_integer('lrate_decay', 250, 'Exponential learning rate decay (in 1000 steps).')
flags.DEFINE_integer('block_size', 129, 'Block Size trained one time. Should be an odd number.')
flags.DEFINE_integer('N_rand',32*32*4, '# random points per gradient step(only used in sequential training)')


flags.DEFINE_boolean('add_ssim_loss', False, 'Add ssim loss or not.')
flags.DEFINE_float('ssim_loss_weight', 0.01, 'Weight of ssim loss.')
flags.DEFINE_float('L1_regu_weight', 0.1, 'Weight of l1 regularization term.')
flags.DEFINE_float('TV_regu_weight', 0.01, 'Weight of gradients regularization term.')

flags.DEFINE_integer('skip', 1, 'Load 1/N scenes from train set.')
# rendering options
flags.DEFINE_integer('chunk', MAXINT, '#pts sent through network in parallel, decrease if running out of memory.')
flags.DEFINE_list('render_size', None, 'Size of the extracted volume, should be in format "H,W,D"')
flags.DEFINE_list('render_t_range', None, 'Time range of rendered 3D videos, should be in format [t_min,t_max].')
flags.DEFINE_list('ROI_Box', None, 'ROI box of rendered scene, should be in format [x,y,z,w,h,d].')
flags.DEFINE_boolean('retain_gray', False, 'Keep rendered 3D videos grayscale.')

flags.DEFINE_string('Hdir', None, 'H_mtx path for rendering projections. Omit when render_proj is set to False. None for reproducing training inputs.')
# finetuning options
flags.DEFINE_integer('N_steps_ft', 2000, '#finetuning steps for each scene.')
flags.DEFINE_float('lrate_ft', 0.004, 'Learn rate for finetuning.')
flags.DEFINE_list('lrate_decay_ft', [1000, 1500, 1800], 'Learn rate decay step.')
flags.DEFINE_float('lrate_decay_gamma', 0.5, 'Learn rate decay amount.')
# logging
flags.DEFINE_integer('i_weights', 10000, 'Weight ckpt saving interval.')
flags.DEFINE_integer('i_print', 1000, 'Printout and logging interval.')
flags.DEFINE_boolean('shot_when_print', False, 'Save snapshot when logging.')
flags.DEFINE_integer('i_weights_trans', 100, 'Transfer-learning Weight ckpt saving interval.')

FLAGS=flags.FLAGS

def main(argv):
    del argv

    ## User Defined Modules
    if FLAGS.action.lower() == 'train':
        from train_demo import train_seq
        train_seq(FLAGS)
    if FLAGS.action.lower() == 'train-dp':
        from train_demo_DP_2 import train_seq as train_seqDP
        train_seqDP(FLAGS)
    elif FLAGS.action.lower() == 'test':
        from test import test
        test(FLAGS)
    else:
        print("Unrecognized action. Do nothing.")



if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # np.random.seed(0)
    if DEVICE.type == 'cuda':
        print(f"Run on CUDA:{GPU_IDX} -- Name: {torch.cuda.get_device_name(GPU_IDX)}")
        device = torch.device(f"cuda:{GPU_IDX}")
        torch.cuda.set_device(device)
    else:
        print("Run on CPU")
    torch.set_default_tensor_type('torch.FloatTensor')
    app.run(main)
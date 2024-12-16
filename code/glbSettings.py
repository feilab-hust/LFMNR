import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch

GPU_IDX = 0
DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    DEVICE = torch.device(f"cuda:{GPU_IDX}")
    torch.cuda.set_device(DEVICE)

# ckpt file prefix
PFX_model = 'cp_'
PFX_embedder = 'em_'
PFX_post = 'ps_'
PFX_feat = 'feat_'
# constants
from sys import maxsize as MAXINT
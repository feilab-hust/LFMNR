
## FLFM_simualtion
- Tools under this directory are used to conduct numerical simulation of FLFM, which are modified from olaf [1].  <br>
Ref: \
[1] Stefanoiu, A., Page, J. & Lasser, T. "olaf: A flexible 3d reconstruction framework for light field microscopy". (2019). https://mediatum.ub.tum.de/1522002
## Functions
- Point spread function (PSF) of FLFM simulation
- Light-field forward projection
- FLFM RL-Deconvolution

## Example usages

* FLFM PSF generation

  - Click "main_FLFM_Simu.m" to generate PSF
  - Waiting for several minutes. PSF will be saved both in forms of ".mat" and ".tif"
* Forward projection

  - Click "ForwardBatch_GPU.m"
  - Choose the 3D stack for generating LF projection (*e.g. The 3D microtubule data at '/data/Trained_Weights_Data/Tube_view7_simuProjection/Ref_3D_GT'*)
  - Waiting for several seconds. The 2D LF will be saved at the child folder "Hex_projection"
* Deconvolution for FLFM 3D reconstruction
  - Click "ReconstructionGPU.m"
  - Choose the 2D LF images
  - Waiting for several minutes. Deconvolution results will be saved at the child folder "Deconv"
  \
  *Note: Users can set the iterative times in "ReconstructionGPU.m"*
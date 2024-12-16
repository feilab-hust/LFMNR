

## High-fidelity, generalizable light-field reconstruction of biological dynamics with physics-informed meta neural representation


This is a repository for the self-supervised light-field microscopy reconstruction approach, termed Light-Filed Meta Neural Representation (LFMNR).
LFMNR incorporates the physics model with implicit neural representation, offering high-fidelity 3D reconstruction.

## Features 
* Self-supervised FLFM 3D reconstruction with resolution improvement and artifacts removal
* Meta-learning acceleration (~100-fold)
* Extendability on other tomographic 3D-reconstruction (e.g. FPM data, OPT data, Cryo-ET data)

## System Requirements
- Windows 10. Linux should be able to run the Python-based codes.
- Graphics: Nvidia GPU with >16 GB graphic memory (RTX 4090 recommended or better)
- Memory: >=32 GB RAM 
- Hard Drive: ~50GB free space (SSD recommended)

## Dependencies
```
  - python=3.10.8
  - cudatoolkit=11.8
  - pytorch=2.0.1
  - torchvision=0.15.2
  - tensorboard==2.10.1
  - scipy==1.12.0
  - imageio
  - tifffile
  - six==1.16
  - tqdm==4.64.1
  - mat73==0.59
```

***Note: We provide a packages' list ("create_environment.yml"), containing the required dependencies can help users to fast install the running environment.
To use this, run the following commend inside a [conda](https://docs.conda.io/en/latest/) console***
  ```
  conda env create -f create_environment.yml
  ```

## Overview of repository
```
├── code:
    The source codes of LFMNR 
    ├── configs:
            The configuration files of different sample.
    ├── core: 
            Network model and image-processing codes
    ├── source 
            Example images used in "Readme"
    '.py' files:
    ├── DataFlow2_4Dweights.py  (package the trained weights)
    ├── glbSettings.py  (Device settings)
    ├── main.py (main function)
    ├── test.py (network inference with trained weights)
    ├── train_demo.py (train LFMNR)
    ├── train_demo_DP.py (train LFMNR with multi-GPUs)
    
├── FLFM_simulation
    The codes for FLFM PSF simulation
├── data
    └── LF: training data for 3D scene
    └── LF-T: training data for 4D scene
    └── Trained_Weights_Data: trained weights and corresponding results for validation
├── results: 
    stores the output files (model weights & logs when training, 3D & 4D reconstruction results)
```

## Get Started 
#### Training and test data preparation: ####
We provide example data for network training and testing. Please download from [Google Drive](https://drive.google.com/drive/folders/1mcP4AzC2waJfwcpjPuQ1vObnjq_mm9aq?usp=sharing)

### Commands for Training

#### Example 1: Single scene training
This command was used to represent one scene (mitochondria, here) with LFINR. The parameters and corresponding data have been included in the config file ('*LF_mito_scene1.cfg*').
 Users can run the following command in console
  ```
  python ./code/main.py --flagfile=code/configs/LF_mito_scene1.cfg
  ```
  #### Example 2: Multi-scenes training
  For fast evaluation the transfer-learning ability of LFMNR, we provided the network weights of the first scene to guide the optimization of remained frames. Users can type the following command in console to start multi-scenes optimization. 
  ```
  python ./code/main.py --flagfile=code/configs/LF-T_Matrix_Seq.cfg
  ```

### Commands for Test
We provided multiple LFMNR weights which stored different scenes containing light-field 3D/4D reconstruction, OPT/FPM/Cryo-ET reconstructions. The following commands are used to
query the images' data with coordinates input.

#### Example 1: Microtubule (Semi-synthetic) 3D-reconstruction 
* Weight & Data path: "*data/Trained_Weights_Data/Tube_view7_simuProjection*", which contains the Deconv./VE-Deconv results, corresponding "Ref-3D" ground truth and LFINR weights.
* LFINR results: After running the command, the LFINR result will be automatically saved at the folder of "*results/Tube_view7_simuProjection*". 
* Commands:
  ```
  python ./code/main.py --flagfile=code/configs/LF_Tube_view7_3D_test.cfg
  ```
#### Example 2: Mitochondria Matrix 3D-reconstruction 
* Weight & Data path: "*data/Trained_Weights_Data/Mito_view7_3D*", which contains the Deconv./VE-Deconv results and LFINR weights. 
* LFINR results: After running the command, the LFINR result will be automatically saved at the folder of "*results/Mito_view7_3D*". 
* Commands:
  ```
  python ./code/main.py --flagfile=code/configs/LF_Mito_view7_3D_test.cfg
  ```

#### Example 3: Mitochondria Matrix 4D-reconstruction 
* Weight path: "*data/LF/Trained_Weights_Data/Mito_view7_4D*", which contains LFMNR weights across 800 timepoints.
* LFMNR results: After running the command, the LFINR result will be automatically saved at the child folder of "*Mito_view7_4D*". 
* Commands:\
  Note: Because the whole 4D image needs large storage space, here, we set the t_range as [1,30] and ROI-box as [90,105,1,363,182,51],
  t_range should be in the form of \[t<sub>start</sub>, t<sub>end</sub>].ROI-box should be in the form of [x,y,z,w,h,d], where x&y&z is
  the 3D coordinates of top-left corner and w&h&d is the 3D size of ROI. Users can customize the 4D ROI in "*configs/LF_Mito_view7_4D_test.cfg*"
  ```
  python ./code/main.py --flagfile=code/configs/LF_Mito_view7_Apo_4D_test.cfg
  ```

#### Example 4: Other tomographic-like 3D-reconstruction
① *** Optical Projection Tomography Data*** 
* Weight & Data path: "*data/Trained_Weights_Data/OtherModalities/Opt*", which contains the SIRT-3D results and INR-net weight under 3 & 7 & 13 views.
* INR-based results: After running the command, the INR-based result will be automatically saved at the child folder of "*results/Opt_view_{view_number}*". 
* Commands:
  Note: Users need to change the parameter "expname" in "*configs/OPT_3D_test.cfg*" to obtain INR results under specified views number.\
  For example, the default value is "Opt_view_03". Running the following command will yield INR result under 3 views. Users can change to "Opt_view_03" or "Opt_view_13"
  ```
  python ./code/main.py --flagfile=code/configs/OPT_3D_test.cfg
  ```
  
② ***3D-reconstruction for Fourier Ptychographic Microscopy data*** 
* Weight & Data path: "*data/Trained_Weights_Data/OtherModalities/FPM*", which contains the classical model based results 
  and INR-net weights under 20 & 40 & 60 views .
* INR-based results: After running the command, the INR-based result will be automatically saved at the child folder of "*FPM_view_{view_number}*". 
* Commands:
 Like OPT reconstruction, users can also change the "expname" in "*configs/Cryo_3D_test.cfg*" to obtain INR reconstruction under different view number. Default: 20. (40,60 available)
  ```
  python ./code/main.py --flagfile=code/configs/FPM_worm_3D_test.cfg
  ```

③ ***3D-reconstruction for Cryogenic Electron Tomography Data*** 
* Weight & Data path: "*data/Trained_Weights_Data/OtherModalities/Cryo-ET*", which contains the SIRT-3D results and INR-net weights under 7 & 13 & 21 views.
* LFINR results: After running the command, the INR-based result will be automatically saved at the child folder of "CryoET_view_{view_number}". Default: 7. (13,21 available). 
* Commands:
  ```
  python ./code/main.py --flagfile=code/configs/Cryo_3D_test.cfg
  ```

## Citation
If you use this code and relevant data, please cite the corresponding paper where original methods appeared:
Yi, C., Ma, Y., Sun, M., Sun, J.  et al. High-fidelity, generalizable light-field reconstruction of biological dynamics with physics-informed meta neural representation. 

## Correspondence
Should you have any questions regarding this code and the corresponding results, please contact Chengqiang Yi (cqyi@hust.edu.cn)

## TODO
* Writing detailed user manual
* Re-organize the codes for readability
* Graphical User Interface


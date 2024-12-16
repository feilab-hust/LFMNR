% oLaF - a flexible 3D reconstruction framework for light field microscopy
% Copyright (c)2017-2020 Anca Stefanoiu

function [H, Ht] = FLFM_computeLFMatrixOperators(Camera, Resolution,ax2ca,mla2axicon,is_axicon)

%% Sensor and ML space coordinates
IMGsizeHalf_y = floor(Resolution.sensorSize(1)/2);
IMGsizeHalf_x = floor(Resolution.sensorSize(2)/2);
Resolution.yspace = Resolution.sensorRes(1)*linspace(-IMGsizeHalf_y, IMGsizeHalf_y, Resolution.sensorSize(1));  %sensor plane coordinates
Resolution.xspace = Resolution.sensorRes(2)*linspace(-IMGsizeHalf_x, IMGsizeHalf_x, Resolution.sensorSize(2));
Resolution.yMLspace = Resolution.sensorRes(1)* [- ceil(Resolution.Nnum(1)/2) + 1 : 1 : ceil(Resolution.Nnum(1)/2) - 1];   %local lenslet coordinates
Resolution.xMLspace = Resolution.sensorRes(2)* [- ceil(Resolution.Nnum(2)/2) + 1 : 1 : ceil(Resolution.Nnum(2)/2) - 1];

%% Compute LFPSF operators

% compute the wavefront distribution incident on the MLA for every depth
fprintf('\nCompute the PSF stack at the back aperture stop of the MO.')
psfSTACK = FLFM_calcPSFAllDepths(Camera, Resolution);

% compute LFPSF at the sensor plane
fprintf('\nCompute the LFPSF stack at the camera plane:')
tolLFpsf = 0.001; % clap small valueds in the psf to speed up computations

for ii=1:length(ax2ca)
    d_camera=ax2ca(ii);
    [H, Ht] = FLFM_computeLFPSF(psfSTACK, Camera, Resolution, tolLFpsf,d_camera,mla2axicon,is_axicon);
    if is_axicon
        save('H_axicon.mat','H','-v7.3');
        save('Ht_axicon.mat','Ht','-v7.3');
    else
        save('H_WF.mat','H','-v7.3');
        save('Ht_WF.mat','Ht','-v7.3');
    end
    [h,w]=size(H{1,1,1});
    psf_img=zeros(h,w,length(Resolution.depths));
    for i = 1:length(Resolution.depths)
        psf_img(:,:,i)=H{1,1,i};
    end

    psf_wf= abs(double(psfSTACK).^2);
    
    psf_wf = mat2gray(psf_wf)*255;
    psf_img = mat2gray(psf_img)*255;
    write3d(psf_wf,'Simu_m100_1.4_p3250_[m3.4-3.4]_step_0.17_WF.tif',8);
    write3d(psf_img,'Simu_m100_1.4_p3250_[m3.4-3.4]_step_0.17_FLFM.tif',8);
end
% %% Import dependecies
import2ws();
addpath ./util
clc;
clear all;
% Import configurations of FLFM setup
Config.gridType =  'Reg';
Config.NA =1.4;
Config.M=100;
Config.f1 = 180000;
Config.fobj = Config.f1/Config.M;         %focal length of the objective lens. Camera.fobj = Camera.ftl/Camera.M

Config.f2 = 400000;        
Config.fm =120000;           %fm - focal length of the micro-lens.
Config.mla2sensor = Config.fm;   %distance between the MLA plane and camera sensor plane.

Config.lensPitch = 3250;
Config.WaveLength = 510*1e-3;

% pixel_size=6.5;

Config.spacingPixels=499;  %the number of pixels between two horizontally neighboring lenslets.
Config.pixelPitch =Config.lensPitch /Config.spacingPixels; % sensor pixel pitch (in ?m).

% Config.lensPitch = Config.pixelPitch*Config.spacingPixels;

Config.immersion_n=1.518;
Config.n = 1;
Config.SensorSize = [2901,2901]; %the size of the input light field image,1075
Config.X_center = ceil(Config.SensorSize(1)/2);
Config.Y_center = ceil(Config.SensorSize(2)/2);
Config.depthStep = 0.17;
Config.depthRange = [-3.4,3.4];

Config.MLAnumX = 3;
Config.MLAnumY = 3;


D_pupuil=Config.f2*Config.NA*2/Config.M;
occupy_ratio=(Config.MLAnumX*Config.lensPitch)/D_pupuil;
FOV=Config.lensPitch*Config.f2/Config.fm/Config.M;
system_magnification=Config.f1/Config.fobj*Config.fm/Config.f2;
rxy=Config.WaveLength*Config.fm/Config.lensPitch/system_magnification;
rz=Config.WaveLength/(sqrt(1))*(Config.f2/Config.lensPitch*Config.fobj/Config.f1)^2;

sr_=rxy/(Config.pixelPitch/system_magnification);
N_=rxy*2.*Config.NA/Config.WaveLength;

tan_theta=Config.lensPitch/Config.f2;
DOF_ge_ideal=Config.lensPitch*Config.f2/Config.fm/(tan_theta*Config.M^2);
DOF_wave_ideal=N_^2*Config.WaveLength/Config.NA^2*(1+1/(2*sr_));

fprintf('[!]System Magnification: %.4f Optical Resolution (Rxy):%.4f (Rz): %.4f Voxel_size:(%.3f,%.3f)\n',system_magnification,rxy,rz,Config.pixelPitch/system_magnification,Config.depthStep);
fprintf('[!]System fov:%.3f DOF:%.3f D_pupuil:%.1f occupy_ratio:%.2f\n',FOV,DOF_wave_ideal,D_pupuil,occupy_ratio);
fprintf('[!]System Nnum:%d mla_num (%d-%d) Pitch size: %.2f\n',Config.spacingPixels,Config.MLAnumY,Config.MLAnumX,Config.pixelPitch);


%%axicon
Config.n_axicon=1.534;
Config.theta=2*pi/180;
mla2axicon=0;

beta_=atan(Config.lensPitch/2/Config.fm);
h_=Config.lensPitch/2-mla2axicon*Config.lensPitch/2/Config.fm;
gamma=asin(  Config.n_axicon *sin( beta_+Config.theta  ))-Config.theta;
NA_gamma=1*sin(gamma);
Rxy_gamma=Config.WaveLength/2/NA_gamma/system_magnification;
z_focus=h_/tan(gamma);
z_inference_max=Config.lensPitch*Config.fm/(2*Config.fm*(Config.n_axicon-1)*Config.theta+Config.lensPitch)-mla2axicon;
% ax2ca=z_inference_max;

superResFactor =1; % superResFactor controls the lateral resolution of the reconstructed object. It is interpreted as a multiple of the lenslet resolution (1 voxel/lenslet). superResFactor =’default’ means the object is reconstructed at sensor resolution, while superResFactor = 1 means lenslet resolution.
[Camera,LensletGridModel] = Compute_camera(Config,superResFactor);
Resolution = Compute_Resolution(Camera,LensletGridModel);

% is_axicon=0;
% [H_raw, Ht_raw] = FLFM_computeLFMatrixOperators(Camera, Resolution,ax2ca,mla2axicon,is_axicon);



is_axicon=0;
if ~is_axicon
    ax2ca=Config.fm;
else
    ax2ca=18258776810.2/1e6;
end
[H, Ht] = FLFM_computeLFMatrixOperators(Camera, Resolution,ax2ca,mla2axicon,is_axicon);




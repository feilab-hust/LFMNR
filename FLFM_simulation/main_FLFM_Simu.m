% %% Import dependecies
import2ws();
addpath ./util
clc;
clear all;
% Import configurations of FLFM setup
Config.NA =1.4;
Config.M=100;
Config.f1 = 180000;
Config.fobj = Config.f1/Config.M;   %focal length of the objective lens. Camera.fobj = Camera.ftl/Camera.M


Config.MLAnumX = 3; % deprecated
Config.MLAnumY = 3; % deprecated


Config.f2 = 400000;        
Config.fm =120000;           %fm - focal length of the micro-lens.
Config.mla2sensor = Config.fm;   %distance between the MLA plane and camera sensor plane.

Config.lensPitch = 3250;
Config.WaveLength = 510*1e-3;

Config.spacingPixels=499;  %the number of pixels between two horizontally neighboring lenslets.
Config.pixelPitch =Config.lensPitch /Config.spacingPixels; % sensor pixel pitch (in um).

Config.immersion_n=1.518;
Config.n = 1;
Config.SensorSize = [2901,2901]; %the size of the input light field image
Config.X_center = ceil(Config.SensorSize(1)/2);
Config.Y_center = ceil(Config.SensorSize(2)/2);
Config.depthStep = 0.17;
Config.depthRange = [-3.4,3.4];


[Camera,LensletGridModel] = Compute_camera(Config,1);
Resolution = Compute_Resolution(Camera,LensletGridModel);
[H, Ht] = FLFM_computeLFMatrixOperators(Camera, Resolution,Config.fm,0,0);



% oLaF - a flexible 3D reconstruction framework for light field microscopy
% Copyright (c)2017-2020 Anca Stefanoiu

function [H, Ht] = WF_computeLFMatrixOperators(Camera, Resolution,ax2ca,TL2axicon,is_axicon)

%% Sensor and ML space coordinates
IMGsizeHalf_y = floor(Resolution.sensorSize(1)/2);
IMGsizeHalf_x = floor(Resolution.sensorSize(2)/2);
Resolution.yspace = Resolution.sensorRes(1)*linspace(-IMGsizeHalf_y, IMGsizeHalf_y, Resolution.sensorSize(1));  %sensor plane coordinates
Resolution.xspace = Resolution.sensorRes(2)*linspace(-IMGsizeHalf_x, IMGsizeHalf_x, Resolution.sensorSize(2));
%% Compute LFPSF operators

% compute the wavefront distribution incident on the MLA for every depth
fprintf('\nCompute the PSF stack at the back aperture stop of the MO.')
psfSTACK = zeros(length(Resolution.yspace), length(Resolution.xspace), length(Resolution.depths));


% [XX,YY]=meshgrid(Resolution.xspace,Resolution.yspace);

%% Wavefront at the MO's fron focal plane
A = 1; % some amplitude
kn = 2*pi/Camera.WaveLength*Camera.immersion_n; % wave number
k0 = 2*pi/Camera.WaveLength;
% choose unevent number of points to have a mid point
N = 16381; % samples -> relatively large to make sure the field is not undersampled
midPoint = (N + 1) / 2;
NoutputY = length(Resolution.yspace);
NoutputX = length(Resolution.xspace);

% physical length of the sampled input field at the NOP in micrometers
LU0 = 500;

% coordinates of the NOP
x = linspace(-LU0/2, LU0/2, N);
y = x;
[X,Y] = meshgrid(x,y);
% dx=x(2)-x(1);
% dy=x(2)-x(1);
% res=[dx,dy];
for i = 1:length(Resolution.depths)
    p1=0;
    p2=0;
    p3=Resolution.depths(i);
    if p3 == 0
        p3 = 1e-8;
    end
    
    % distance from point source to the MO
    r = sqrt((X-p1).^2.+(Y-p2).^2.+p3.^2);
    
    % when p3>0, propagate back to the lens
    if p3 > 0
        r = -1*r;
    end
    % field at the NOP
    U0 = -1i*A*kn/2/pi./r .* exp(1i.*kn.*r);
    
    %% Wavefront at the MO's front focal plane
    % due to the FFT there is a scaling factor and LU1 is the length of the field at the the back aperture stop of the MO
    [U1, LU1] = FLFM_lensProp(U0, LU0, Camera.WaveLength, Camera.fobj);
    coeffU1minus = -1i*exp(1i*kn*Camera.fobj)/Camera.WaveLength/Camera.fobj;
    U1 = coeffU1minus.*U1;
    % back aperture stop
    circ = @(x,y,r) (x.^2.+y.^2.<r.^2); % pupil function
    dobj = 2*Camera.fobj*Camera.NA;
    U1 = U1.*circ(X./LU0.*LU1, Y./LU0.*LU1, dobj./2);
    dx1=Camera.WaveLength*Camera.fobj/LU0;          %dx1 
    res=[dx1,dx1];
    xx1 = linspace(-LU1/2, LU1/2, N);
    [XX1,YY1]=meshgrid(xx1,xx1);
%     
%     Mrelay = 1;
%     % we have to reshape U1 to match the spacing and extent of the xspace and yspace
%     cut = [round(Resolution.yspace(end) / (Mrelay*LU1/2) * (N+1)/2), round(Resolution.xspace(end) / (Mrelay*LU1/2) * (N+1)/2)];
%     psf = U1(midPoint - cut(1) : midPoint + cut(1), midPoint - cut(2) : midPoint + cut(2));
%     % downsample the psf the back focal plane of the MO
%     back_focal = imresize(psf, [NoutputY NoutputX], 'bicubic');
%     imshow(mat2gray(abs(back_focal.^2)))

    %% TubelensPlane
    tube_plane = prop2Sensor(U1, res, Camera.f1, Camera.WaveLength, 0);
    tube_plane = tube_plane.*exp(-1i*k0*(XX1.^2+YY1.^2)./(2*Camera.f1));
    
    %% AxiconPlane
    axicon_plane= prop2Sensor(tube_plane, res, TL2axicon, Camera.WaveLength, 0);
    axicon_plane= axicon_plane.*exp(-1i.*k0.*(Camera.n_axicon-1).*sqrt(XX1.^2+YY1.^2).*Camera.theta);

    %% ImgPlane
    img_plane=prop2Sensor(axicon_plane, res, ax2ca, Camera.WaveLength, 0);
    %% Cut
    Mrelay = 1;
    cut = [round(Resolution.yspace(end) / (Mrelay*LU1/2) * (N+1)/2), round(Resolution.xspace(end) / (Mrelay*LU1/2) * (N+1)/2)];
    psf = img_plane(midPoint - cut(1) : midPoint + cut(1), midPoint - cut(2) : midPoint + cut(2));
    psf = imresize(psf, [NoutputY NoutputX], 'bicubic');
    psfSTACK(:,:,i)=psf;
end
tolLFpsf=0.05;
H=psfSTACK;

for c = 1: size(H,3)
    for aa = 1:size(H,1)
        for bb = 1:size(H,2)
            temp = H(aa,bb,c);
            max_slice = max(temp(:));
            % Clamp values smaller than tol 
            temp(temp < max_slice*tol) = 0;
            sum_temp = sum(temp(:));
            if(sum_temp==0)
                continue
            end
            % and normalize per individual PSF such that the sum is 1.
            temp = temp/sum_temp;
            H(aa,bb,c) = sparse(temp);
        end
    end
end

Ht=0;

% % %% Compute backward light transport 
% % disp('Computing backward light propagation')
% 
% % backward patterns in this case are just rotated forward patterns
% Ht = cell(1, 1, length(Resolution.depths));
% for i = 1:length(Resolution.depths)
%     Ht{1,1,i} = imrotate(H{1,1,i}, 180);
% end
% 
% % make sure the application of the inverse psf (through convolution) preserves the object energy (See Richardson Lucy algorithm)
% Ht = normalizeHt(Ht); 

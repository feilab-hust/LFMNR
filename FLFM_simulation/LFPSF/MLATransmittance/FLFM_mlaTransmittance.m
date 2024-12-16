% oLaF - a flexible 3D reconstruction framework for light field microscopy
% Copyright (c)2017-2020 Anca Stefanoiu 

function MLARRAY = FLFM_mlaTransmittance(Resolution, ulensPattern)

%% Compute the ML array as a grid of phase/amplitude masks corresponding to mlens
ylength = length(Resolution.yspace);
xlength = length(Resolution.xspace);

MLcenters = zeros(ylength, xlength);




%%%%%%% MLA arrangement%%%%%%%
d=size(ulensPattern,1);
d1=0;
center=ceil(ylength/2);
fprintf('\n %d size  -- center %d',d,center);

% %% HEX 7
MLcenters(center-ceil(sqrt(3)*(d+d1)/2),center-ceil((d+d1)/2))=1;
MLcenters(center-ceil(sqrt(3)*(d+d1)/2),center+ceil((d+d1)/2))=1;

MLcenters(center,center-ceil((d+d1)))=1;
MLcenters(center,center)=1;
MLcenters(center,center+ceil((d+d1)))=1;

MLcenters(center+ceil(sqrt(3)*(d+d1)/2),center-ceil((d+d1)/2))=1;
MLcenters(center+ceil(sqrt(3)*(d+d1)/2),center+ceil((d+d1)/2))=1;

%%%%%%%%%%%%%%%%%%%%%

% apply the mlens pattern at every ml center
MLARRAY  = conv2(MLcenters, ulensPattern, 'same');


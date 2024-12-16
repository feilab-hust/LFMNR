% oLaF - a flexible 3D reconstruction framework for light field microscopy
% Copyright (c)2017-2020 Anca Stefanoiu 

function MLARRAY = FLFM_mlaTransmittance(Resolution, ulensPattern)

%% Compute the ML array as a grid of phase/amplitude masks corresponding to mlens
ylength = length(Resolution.yspace);
xlength = length(Resolution.xspace);

% LensletCenters(:,:,1) = round(Resolution.LensletCenters(:,:,2));%[110,324,538,752,966] size=5x5
% LensletCenters(:,:,2) = round(Resolution.LensletCenters(:,:,1));

% activate lenslet centers -> set to 1

MLcenters = zeros(ylength, xlength);

% d=333;
% 
% MLcenters(833,1499)=1;
% MLcenters(833,1166)=1;
% MLcenters(833,833)=1;
% MLcenters(833,500)=1;
% MLcenters(833,167)=1;
% 
% MLcenters(833-ceil(sqrt(3)*d/2),833-ceil(3*d/2))=1;
% MLcenters(833-ceil(sqrt(3)*d/2),833-ceil(d/2))=1;
% MLcenters(833-ceil(sqrt(3)*d/2),833+ceil(d/2))=1;
% MLcenters(833-ceil(sqrt(3)*d/2),833+ceil(3*d/2))=1;
% 
% MLcenters(833+ceil(sqrt(3)*d/2),833-ceil(3*d/2))=1;
% MLcenters(833+ceil(sqrt(3)*d/2),833-ceil(d/2))=1;
% MLcenters(833+ceil(sqrt(3)*d/2),833+ceil(d/2))=1;
% MLcenters(833+ceil(sqrt(3)*d/2),833+ceil(3*d/2))=1;
% 
% 
% MLcenters(833-ceil(sqrt(3)*d),833-d)=1;
% MLcenters(833-ceil(sqrt(3)*d),833)=1;
% MLcenters(833-ceil(sqrt(3)*d),833+d)=1;
% 
% MLcenters(833+ceil(sqrt(3)*d),833-d)=1;
% MLcenters(833+ceil(sqrt(3)*d),833)=1;
% MLcenters(833+ceil(sqrt(3)*d),833+d)=1;


%% 460 size
% fprintf('\n460 size');
% MLcenters(352,521)=1;
% MLcenters(352,981)=1;
% 
% MLcenters(751,291)=1;
% MLcenters(751,751)=1;
% MLcenters(751,1211)=1;
% 
% MLcenters(1150,521)=1;
% MLcenters(1150,981)=1;


% 501 size
% fprintf('\n501 size');
% d = 501;
% center=756;
% 
% MLcenters(center-ceil(d*sqrt(3)/2),center-floor(d/2))=1;
% MLcenters(center+ceil(d*sqrt(3)/2),center-floor(d/2))=1;
% 
% MLcenters(center,center-d)=1;
% MLcenters(center,center)=1;
% MLcenters(center,center+d)=1;
% 
% MLcenters(center-ceil(d*sqrt(3)/2),center+floor(d/2))=1;
% MLcenters(center+ceil(d*sqrt(3)/2),center+floor(d/2))=1;


% %% 383 size
% fprintf('\n383 size');
% d=383;
% MLcenters(751-ceil(sqrt(3)*d/2),751-ceil(d/2))=1;
% MLcenters(751-ceil(sqrt(3)*d/2),751+ceil(d/2))=1;
% 
% MLcenters(751,751-d)=1;
% MLcenters(751,751)=1;
% MLcenters(751,751+d)=1;
% 
% MLcenters(751+ceil(sqrt(3)*d/2),751-ceil(d/2))=1;
% MLcenters(751+ceil(sqrt(3)*d/2),751+ceil(d/2))=1;



%% 383 size
% 
% d=461;
% center=752;
% fprintf('\n %d size  -- center %d',d,center);
% MLcenters(center-ceil(sqrt(3)*d/2),center-ceil(d/2))=1;
% MLcenters(center-ceil(sqrt(3)*d/2),center+ceil(d/2))=1;
% 
% MLcenters(center,center-d)=1;
% MLcenters(center,center)=1;
% MLcenters(center,center+d)=1;
% 
% MLcenters(center+ceil(sqrt(3)*d/2),center-ceil(d/2))=1;
% MLcenters(center+ceil(sqrt(3)*d/2),center+ceil(d/2))=1;

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

%% hex 3
% MLcenters(center-ceil(sqrt(3)*(d+d1)/2),center-ceil((d+d1)/2))=1;
% MLcenters(center-ceil(sqrt(3)*(d+d1)/2),center+ceil((d+d1)/2))=1;
% 
% MLcenters(center,center-ceil((d+d1)))=1;
% MLcenters(center,center)=1;
% MLcenters(center,center+ceil((d+d1)))=1;
% 
% MLcenters(center+ceil(sqrt(3)*(d+d1)/2),center-ceil((d+d1)/2))=1;
% MLcenters(center+ceil(sqrt(3)*(d+d1)/2),center+ceil((d+d1)/2))=1;


%% Hex5  sampling  rotate
% MLcenters(center-ceil(2*sqrt(3)*(d+d1)/2)-1,center-d-d1)=1;
% MLcenters(center-ceil(2*sqrt(3)*(d+d1)/2)-1,center)=1;
% MLcenters(center-ceil(2*sqrt(3)*(d+d1)/2)-1,center+d+d1)=1;
% 
% 
% MLcenters(center-ceil(sqrt(3)*(d+d1)/2),center-ceil(3*(d+d1)/2))=1;
% MLcenters(center-ceil(sqrt(3)*(d+d1)/2),center+ceil(3*(d+d1)/2))=1;
% 
% MLcenters(center-ceil(sqrt(3)*(d+d1)/2),center-ceil((d+d1)/2))=1;
% MLcenters(center-ceil(sqrt(3)*(d+d1)/2),center+ceil((d+d1)/2))=1;
% 
% 
% MLcenters(center,center-2*(d+d1))=1;
% MLcenters(center,center-d-d1)=1;
% MLcenters(center,center)=1;
% MLcenters(center,center+d+d1)=1;
% MLcenters(center,center+2*(d+d1))=1;
% 
% MLcenters(center+ceil(sqrt(3)*(d+d1)/2),center-ceil((d+d1)/2))=1;
% MLcenters(center+ceil(sqrt(3)*(d+d1)/2),center+ceil((d+d1)/2))=1;
% 
% 
% MLcenters(center+ceil(sqrt(3)*(d+d1)/2),center-ceil(3*(d+d1)/2))=1;
% MLcenters(center+ceil(sqrt(3)*(d+d1)/2),center+ceil(3*(d+d1)/2))=1;
% 
% 
% 
% MLcenters(center+ceil(2*sqrt(3)*(d+d1)/2)+1,center-d-d1)=1;
% MLcenters(center+ceil(2*sqrt(3)*(d+d1)/2)+1,center)=1;
% MLcenters(center+ceil(2*sqrt(3)*(d+d1)/2)+1,center+d+d1)=1;




% MLcenters= imrotate(MLcenters,90);
% for a = 1:size(LensletCenters,1) 
%     for b = 1:size(LensletCenters,2) 
% %         if( (LensletCenters(a,b,1) < size(ulensPattern,1)/2) || (LensletCenters(a,b,1) > ylength - size(ulensPattern,1)/2) || ...
% %               (LensletCenters(a,b,2)< size(ulensPattern,2)/2) || (LensletCenters(a,b,2) > xlength - size(ulensPattern,2)/2)  )
% %           continue
% %         end
%         MLcenters( LensletCenters(a,b,1), LensletCenters(a,b,2)) = 1;
%     end
% end

% apply the mlens pattern at every ml center
MLARRAY  = conv2(MLcenters, ulensPattern, 'same');


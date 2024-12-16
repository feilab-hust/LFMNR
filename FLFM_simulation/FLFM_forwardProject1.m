function Projection = FLFM_forwardProject1(H, realSpace)
% forwardProjectFLFM: Forward projects a volume to a lenslet image by applying the LFPSF

Projection = zeros([size(realSpace,1), size(realSpace,2)]);
for j = 1:size(H,3)
    fprintf('[%d/%d] projection\n',j,size(H,3))
    Projection = Projection + conv2(realSpace(:,:,j), H(:,:,j),'same');
end
end

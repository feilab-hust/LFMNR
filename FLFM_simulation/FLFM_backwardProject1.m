function BackProjection = FLFM_backwardProject1(Ht, projection)
% backwardProjectFLFM: back projects a lenslet image into a volume.

BackProjection = zeros([size(projection,1), size(projection,2), size(Ht,3)]);
for j = 1:size(Ht,3)
    fprintf('[%d/%d] projection\n',j,size(H,3))
    BackProjection(:,:,j) = conv2(projection , Ht(:,:,j),'same');
end
end
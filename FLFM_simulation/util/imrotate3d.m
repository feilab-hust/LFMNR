function img_rotate=imrotate3d(ZsampledVol,angle_xy,angle_z,angle_zY,method,bbox)
% % %
% input:img stack shape =[height, width,depth]
%  rotate 'angle_xy' in an anticlockwise direction pivoting 'depth'   
%  rotate 'angle_z' in an anticlockwise direction pivoting 'width'
% % %
    assert(ndims(ZsampledVol)==3, sprintf('Sample volume shape erro \n'))
    %xy rotate   [ height, width,depth]
    VolRot = imrotate(ZsampledVol, angle_xy, method, bbox);
    vol_rearrange=permute(VolRot,[1 3 2]);
    vol_rearrange = permute(vol_rearrange,[2 1 3]);
    
    %z rotate     depth height  width
    vol_rearrange = imrotate(vol_rearrange, angle_z, method, bbox);
    %convert to [height width depth]
    vol_rearrange = permute(vol_rearrange,[2 1 3]);
    vol_rearrange=permute(vol_rearrange,[1 3 2]);
    
    %z rotate     width depth  height
    vol_rearrange = permute(vol_rearrange,[2 3 1]);
%     vol_rearrange = permute(vol_rearrange,[2 3 1]);
    vol_rearrange = imrotate(vol_rearrange, angle_zY, method, bbox);
%     vol_rearrange = permute(vol_rearrange,[2 3 1]);
    vol_rearrange = permute(vol_rearrange,[2 3 1]);
    img_rotate = permute(vol_rearrange,[2 3 1]);
%     img_rotate = flip(img_rotate, 3);
end
 


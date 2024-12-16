function temp_map=interView_convolution(map_size,local_psf_collect,padding_g)
%% interView_convolution
height=map_size(1);
width=map_size(2);
depth=map_size(3);
temp_map=zeros(map_size);
for h_idx=1:height
    for w_idx=1:width
        pixel_idx=(h_idx-1)*width+w_idx;
        fprintf('[%d/%d] calculating\n',pixel_idx,height*width);
        
        local_psf=local_psf_collect.local_psf;
        
        for d_idx=1:depth
            rt_kernel=rot90(local_psf(:,:,d_idx),2);
            offset_hw=[padding_sz,padding_sz];
            local_img=padding_g(h_idx+offset_hw(1)-padding_sz:h_idx+offset_hw(1)+padding_sz,w_idx+offset_hw(2)-padding_sz:w_idx+offset_hw(2)+padding_sz,d_idx);
            local_kernel=rt_kernel;
            local_value= sum(sum(local_img.*local_kernel));
            temp_map(h_idx,w_idx,d_idx)=local_value;
        end
    end
end
% write3d(conv_map,'tublin_corner_convIdentical.tif',32);
end
function [view_stack] = LFP2Stack(signal,n_num,varargin)
    if nargin == 3
        is_scale = varargin{1};
    end
    if nargin == 2
        is_scale = false;
    end

    sr_factor=15;
    [h,w]=size(signal);
    base_h=floor(h/n_num);
    base_w=floor(w/n_num);
    if is_scale
        view_stack=zeros(base_h*sr_factor,base_w*sr_factor,n_num*n_num);
    else
        view_stack=zeros(base_h,base_w,n_num*n_num);
    end
    for i=1:n_num
        for j=1:n_num
            view_idx=(i-1)*n_num+j;
            temp_view=signal(i: n_num: end, j: n_num: end);
            if is_scale
                temp_view=imresize(temp_view,sr_factor,'bicubic');
            end
            view_stack(:,:,view_idx)=temp_view;
        end
    end
end


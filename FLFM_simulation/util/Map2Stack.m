function [view_stack] = Map2Stack(signal,n_num,varargin)
    if nargin == 3
        is_scale = varargin{1};
    end
    if nargin == 2
        is_scale = false;
    end

    sr_factor=4;
    [h,w]=size(signal);
    base_h=floor(h/n_num);
    base_w=floor(w/n_num);
    view_stack=zeros(base_h,base_w,n_num*n_num);
    for i=1:n_num
        for j=1:n_num
            view_idx=(i-1)*n_num+j;
            view_stack(:,:,view_idx)=signal( (i-1)*base_h+1:i*base_h, (j-1)*base_w+1:j*base_w);
        end
    end
end


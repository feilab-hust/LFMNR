function [view_map] = Stack2Map(signal,n_num)

    [base_h,base_w,d]=size(signal);
    h=floor(base_h*n_num);
    w=floor(base_w*n_num);
    view_map=zeros(h,w);
    for i=1:n_num
        for j=1:n_num
            view_idx=(i-1)*n_num+j;
            view_map((i-1)*base_h+1:i*base_h,(j-1)*base_w+1:j*base_w)=signal(:,:,view_idx);
        end
    end
end
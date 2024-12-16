function [LFP] = Stack2LFP(signal,n_num)

    [base_h,base_w,d]=size(signal);
    h=floor(base_h*n_num);
    w=floor(base_w*n_num);
    LFP=zeros(h,w);
    for i=1:n_num
        for j=1:n_num
            view_idx=(i-1)*n_num+j;
            LFP(i: n_num: end, j: n_num: end)=signal(:,:,view_idx);
        end
    end
end
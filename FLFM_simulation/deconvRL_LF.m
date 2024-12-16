function [recon] = deconvRL_LF(forwardFUN, backwardFUN, img, iter, init)
addpath ./utils

Xguess = init; 
for i=1:iter,
    tic;
    
    HXguess = forwardFUN(Xguess);
    HXguessBack = backwardFUN(HXguess);
    errorBack = img./HXguessBack;
    Xguess = Xguess.*errorBack; 
    Xguess(isnan(Xguess)) = 0;
    ttime = toc;
    disp(['  iter ' num2str(i) ' | ' num2str(iter) ', took ' num2str(ttime) ' secs']);
%     filename=sprintf('Iter_%d.tif',i);
%     write3d(gather(Xguess),filename,32);
end
recon=Xguess;
    
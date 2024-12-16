clc
clear
addpath ./util
addpath ./solvers
addpath ./LFPSF
addpath ./projectionOperators

%% loading H
% H=psf;
GPU_enable=1;
H=load('L:\LFMNR_official\Github_version\FLFM_simulation_hex7\SimuPSF\Hex7_sz2901_100x\H_WF.mat');
H=H.H;

% Ht=load('H:\Code\Extended_dof\extended_hex\HEX\5X5\hex_5x5\WF\Ht_WF.mat');
% Ht=Ht.Ht;
H_size=size(full(H{:,:,1}));

desired_size=2901;
desired_depth=41;

H_depth=size(H,3);
center_idx=ceil(H_depth/2);
sig_half_depth=floor(desired_depth/2);
H = H(:,:,center_idx-sig_half_depth:center_idx+sig_half_depth);
volumeSize = [desired_size,desired_size,desired_depth];
file_name=sprintf('Hex_projection');

%%

if ~GPU_enable
    forwardFUN = @(volume) FLFM_forwardProject(H, volume);
    backwardFUN = @(projection) FLFM_backwardProject(Ht, projection);
else
    forwardFUN = @(volume) FLFM_forwardProjectGPU(H, volume);
    backwardFUN = @(projection) FLFM_backwardProjectGPU(Ht, projection);
    global zeroImageEx;
    global exsize;
    xsize = [volumeSize(1), volumeSize(2)];
    msize = [H_size(1), H_size(2)];
    mmid = floor(msize/2);
    exsize = xsize + mmid;
    exsize = [ min( 2^ceil(log2(exsize(1))), 128*ceil(exsize(1)/128) ), min( 2^ceil(log2(exsize(2))), 128*ceil(exsize(2)/128) ) ];
    zeroImageEx = gpuArray(zeros(exsize, 'single'));
    disp(['FFT size is ' num2str(exsize(1)) 'X' num2str(exsize(2))]);
end

%% generate_projection
[hr_file_name,hr_filepath] = uigetfile('*.tif','Select HR Volumes','MultiSelect','on');
if ~iscell(hr_file_name)
    hr_file_name = {hr_file_name};
end

save_Path=fullfile(hr_filepath,file_name);
if exist(save_Path,'dir')==7
    ;
else
    mkdir(save_Path);
end

for img_idx=1:length(hr_file_name)
    img_name=hr_file_name{img_idx};
    fprintf('\npreocessing %s',img_name);
    path=fullfile(hr_filepath,img_name);
    [img,bitDepth]=imread3d(path);
    img_size=size(img);
    h=img_size(1);
    w=img_size(2);
    sig_len=img_size(3);
    if ~all(img_size==volumeSize)
        raidus=[floor(h/2),floor(w/2)];
        start_cordi=ceil(desired_size/2)-raidus;
        new_h=desired_size;
        new_w=desired_size;
        if start_cordi(1)<0
            incre=0;
            step=Nnum;
            while(start_cordi(1)+incre*step<=0)
                incre=incre+1;
            end
            start_cordi(1)=start_cordi(1)+incre*step;
            new_h=incre*step*2+desired_size;
        end
        
        if start_cordi(2)<0
            incre=0;
            step=Nnum;
            while(start_cordi(2)+incre*step<=0)
                incre=incre+1;
            end
            start_cordi(2)=start_cordi(2)+incre*step;
            new_w=incre*step*2+desired_size;
        end
        
        padding=zeros(new_h,new_w,sig_len);
        for i=1:sig_len
            for j=1:h
                for k=1:w
                    padding(start_cordi(1)+j-1,start_cordi(2)+k-1,i)=img(j,k,i);
                end
            end
        end
    else
        padding=img;
    end

    %%padding axial
    

    beads_projecion=FLFM_forwardProjectGPU(H, padding);
    beads_projecion = gather(beads_projecion);
    save_file=fullfile(save_Path,img_name);
    write3d(beads_projecion,save_file,32);
end







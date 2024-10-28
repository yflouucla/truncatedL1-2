%% tips
%you can set detail=1 to follow intermediate results of each iteration, or set detail=0 to save time
%whether detail is set to 0 or 1, you can always find detailed output of each trial in res.outc and res.outs
%please read the description of batchtest.m for more information
%% test Gaussian sensing matrix for CS without/with noise
clear;
detail=0;
maxtest=5;%number of trials
tau=0;%0.01 0.03 0.05 0.1 noise level
switch tau
    case 0
        s=10;%10:2:32 sparsity for noiseless case
        lambda=1e-6;
    case 0.01
        s=15;%sparsity for noisy case
        lambda=8e-4;
    case 0.03
        s=15;%sparsity for noisy case
        lambda=2e-3;
    case 0.05
        s=15;%sparsity for noisy case
        lambda=3.6e-3;
    case 0.1
        s=15;%sparsity for noisy case
        lambda=1.1e-2;
end
prob=struct('type','cs_gaussian','size',[64 256],'s',s,'tau',tau,'detail',detail);
res=batchtest(prob,maxtest,lambda);
fprintf('\n\n')
%% test over-sampled DCT sensing matrix for CS without/with noise
clear;
detail=0;
maxtest=3;%number of trials
tau=0;%0.01 0.03 0.05 0.1 noise level
switch tau
    case 0
        s=5;%5:2:35 sparsity for noiseless case
        F=20;%coherence level for noiseless case
        lambda=1e-6;
    case 0.01
        s=15;%sparsity for noisy case
        F=10;%coherence level for noisy case
        lambda=2e-4;
    case 0.03
        s=15;%sparsity for noisy case
        F=10;%coherence level for noisy case
        lambda=5e-4;
    case 0.05
        s=15;%sparsity for noisy case
        F=10;%coherence level for noisy case
        lambda=1e-3;
    case 0.1
        s=15;%sparsity for noisy case
        F=10;%coherence level for noisy case
        lambda=2.2e-3;
end
prob=struct('type','cs_dct','size',[100 1500],'s',s,'F',F,'tau',tau,'detail',detail);
res=batchtest(prob,maxtest,lambda);
fprintf('\n\n')
%% test execution time for CS
clear;
detail=0;
maxtest=5;%number of trials
siz=1;%1:7 prblem size
size_temp=[64 256]*2^(siz-1);
lambda=1e-8;
prob=struct('type','cs_time','size',size_temp,'s',size_temp(1)/8,'tau',0,'detail',detail);
res=batchtest(prob,maxtest,lambda);
fprintf('\n\n')
%% test 100X100 exact low rank matrix for MC without/with noise
clear;
detail=0;
maxtest=5;%number of trials
tau=0;%0.01 0.03 0.05 0.1 noise level
switch tau
    case 0
        s=10;%10:29 rank for noiseless case
        lambda=1e-6;
    case 0.01
        s=15;%rank for noisy case
        lambda=2.3e-3;
    case 0.03
        s=15;%rank for noisy case
        lambda=7e-3;
    case 0.05
        s=15;%rank for noisy case
        lambda=1.5e-2;
    case 0.1
        s=15;%rank for noisy case
        lambda=2.5e-2;
end
p=0.5;%sampling ratio
prob=struct('type','mc_exact','size',[100 100],'s',s,'p',p,'tau',tau,'detail',detail);
res=batchtest(prob,maxtest,lambda);
fprintf('\n\n')
%% test 500X500 exact low rank matrix for MC without noise
clear;
detail=0;
maxtest=1;%number of trials
s=[50 70 90 110];%rank
p=[0.2235 0.3063 0.3854 0.4607];%sampling ratio
num=1;%1:4 problem number
lambda=1e-6;
prob=struct('type','mc_exact','size',[500 500],'s',s(num),'p',p(num),'tau',0,'detail',detail);
res=batchtest(prob,maxtest,lambda);
fprintf('\n\n')
%% test 500X500 approximate low rank matrix for MC without noise
clear;
detail=0;
maxtest=1;%number of trials
p=0.3;%0.04 0.08 0.15 0.3 sampling ratio
lambda=1e-6;
prob=struct('type','mc_approximate','size',[500 500],'p',p,'tau',0,'detail',detail);
res=batchtest(prob,maxtest,lambda);
fprintf('\n\n')
%% test MC time
clear;
detail=0;
maxtest=5;%number of trials
siz=1;%1:7 prblem size
size_temp=[64 64]*2^(siz-1);
p=0.5;%sampling ratio
lambda=1e-10;
prob=struct('type','mc_time','size',size_temp,'s',size_temp(1)/16,'p',p,'tau',0,'detail',detail);
res=batchtest(prob,maxtest,lambda);
fprintf('\n\n')
%% test image inpainting without/with noise
clear;
detail=0;
maxtest=5;%number of trials
img_num=1;%1:5 patch number
tau=0;%0 0.1 noise level
switch tau
    case 0
        lambda=1e-6;
    case 0.1
        lambda=[1e-2 1e-2 1.2e-2 1.2e-2 1.3e-2];
        lambda=lambda(img_num);
end
p=0.5;%sampling ratio
prob=struct('type','inpainting','size',img_num,'p',p,'tau',tau,'detail',detail);
res=batchtest(prob,maxtest,lambda);
fprintf('\n\n')
%% test MRI reconstruction
clear;
detail=0;
maxtest=1;%number of trials
size=256;%size of Phantom image
lines=22;%number of radial lines
lambda=1e-10;
prob=struct('type','mri','size',size,'s',lines,'tau',0,'detail',detail);
res=batchtest(prob,maxtest,lambda);
fprintf('\n\n')
%% customized test
% %you may want to do a customized test
% %in this case, you need to input your own data A (operator) and B (measurements)
% %note that for matrix completion and inpainting, we suggest you to
% %  normalize the original data to have unit 2-norm and then generate the
% %  measurements; otherwise, you may need to tune beta in truncatedL1L2 for better performance
% %for par, at least the following two fields are needed:
% clear;clc;close all;
% par.prob='cs';%cs mc mri inpainting
% par.lambda=1e-6;%please see truncatedL1L2 for more information
% %other fields of par are set to their default settings
% %you can tune them to better fit your problem
% %please read the description of truncatedL1L2 for more information
% [X,outc,outs]=truncatedL1L2(A,B,par);
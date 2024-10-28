function [A,X,B,tau_norm,norm_X]=make_data(prob,seed)
%% generate test data
%%%%%%%%%%%%%%%%%%%%
%input: prob,seed
% prob: problem description
%  prob.type: 
%   'cs_gaussian': compressed sensing using Gaussian sensing matrix
%   'cs_dct': compressed sensing using over-sampled DCT sensing matrix
%   'cs_time': compressed sensing with execution time test
%   'mri': MRI reconstruction
%   'mc_exact': matrix completion using exact low-rank matrix
%   'mc_approximate': matrix completion using approximate low-rank matrix 
%   'mc_time': matrix completion with execution time test
%   'inpainting': matrix completion using image patch
%  prob.size:
%   size of sensing matrix for CS and MRI
%   size of ground-truth for MC
%   number of image patch for inpainting
%  prob.s: 
%   sparsity of ground-truth for CS
%   number of radial lines for MRI
%   rank of ground-truth for MC
%   useless for inpainting
%  prob.F: refinement factor for over-sampled DCT sensing matrix
%   the higher F, the higher coherence
%  prob.p: sampling ratio for MC and inpainting
%  prob.tau: noise level
% seed: random seed
%%%%%%%%%%%%%%%%%%%%
%output: A,X,B,tau_norm,norm_X
% A: sensing matrix
% X: ground-truth
% B: measurement
% tau_norm: norm of the noise
% norm_X: 2-norm of ground-truth for inpainting  
%%%%%%%%%%%%%%%%%%%%
type=prob.type;
if length(prob.size)==1
    M=prob.size;N=M;
else
    M=prob.size(1);N=prob.size(2);
end
if isfield(prob,'s');s=prob.s;end
if isfield(prob,'F');F=prob.F;end
if isfield(prob,'p');p=prob.p;end
if ~isfield(prob,'tau');tau=0;else tau=prob.tau;end
if ~exist('seed','var')
    seed_temp=clock;
    seed=floor(seed_temp(end)*1000);
end
randn('seed',seed);
rand('seed',seed);
norm_X=1;
switch type
    case {'cs_gaussian','cs_time'}
        A=randn(M,N);
        A=A/norm(A,2);
        X=zeros(N,1);
        supp=make_supp(N,s,0);
        X(supp)=randn(s,1);
        B=A*X;
        e=randn(M,1);
    case 'cs_dct'
        A=sqrt(2/M)*cos(2*pi*rand(M,1)*(0:N-1)/F);
        A=A/norm(A,2);
        supp=make_supp(N,s,2*F-1);
        X=zeros(N,1);
        X(supp)=randn(s,1);
        B=A*X;
        e=randn(M,1);
    case 'mri'
        X=phantom(M);
        A=ifftshift(MRImask(M,s));
        B=fft2(X);B=B(A)/sqrt(M*N);
        e=randn(length(B),1)+1i*randn(length(B),1);
    case {'mc_exact','mc_time'}
        %X=(rand(M,s)-0.5)*(rand(s,N)-0.5);X=X/norm(X,2);
        X=randn(M,s)*randn(s,N);X=X/norm(X,2);
        A=ones(M,N);
        supp=make_supp(M*N,floor((1-p)*M*N),0);%noisy supp
        A(supp)=0;%known supp
        A=logical(A);
        B=X;B(~A)=0;
        e=randn(M,N);e(~A)=0;
    case 'mc_approximate'
        X1=orth(randn(M));X2=orth(randn(M));tempr=min(size(X1,2),size(X2,2));
        X3=diag(exp(-0.3*[1:tempr]));
        X=X1(:,1:tempr)*X3*X2(:,1:tempr)';X=X/norm(X,2);
        A=ones(M,N);
        supp=make_supp(M*N,floor((1-p)*M*N),0);%noisy supp
        A(supp)=0;
        A=logical(A);
        B=X;B(~A)=0;
        e=randn(M,N);e(~A)=0;
    case 'inpainting'%for inpainting, size is the image number
        data=load('inpaintingdata');
        X=data.final{prob.size(1)};
        [M,N]=size(X);A=ones(M,N);
        supp=make_supp(M*N,floor((1-p)*M*N),0);%noisy supp
        A(supp)=0;
        A=logical(A);
        norm_X=norm(X,2);X=X/norm_X;
        B=X;B(~A)=0;
        e=randn(size(X));e(~A)=0; 
end
tau_norm=tau*norm(B,'fro');%norm of the noise
tau_var=tau_norm/norm(e,'fro');%variance of the noise
B=B+tau_var*e;
end

function supp=make_supp(N,s,space)
if ~exist('space','var')
    space=0;
end
if N-space*(s-1)<s
    error('wrong s and F')
end
rand_sample=rand(1,N-space*(s-1));
[~,temp]=sort(rand_sample);
supp=sort(temp(1:s))+(0:s-1)*space;%supp
end
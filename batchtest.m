function res=batchtest(prob,maxtest,lambda)
%% description
%%%%%%%%%%%%%%%%%%%%
%input: prob,maxtest,lambda
%prob:
% prob.detail: show waitbar and print intermediate results to the screen or not, set to 0 to save time
% other fields: the same as those in make_data.m
%maxtest: number of trials, for example:
% maxtest=100; test 100 trials with seeds 1 to 100
% maxtest={10,20,30}; test 3 trials with seeds 10, 20, 30
%lambda: regularization parameter
%%%%%%%%%%%%%%%%%%%%
%output: res with the following fields
% prob/lambda: the input prob/lambda of batchtest.m
% par: parameter settings of truncatedL1L2
% x/x_t/x_n: solution/ground-truth/measurement, inpainting only
% good_trials/success_rate: successful trials/success rate
% outc/outs: output of truncatedL1L2 in cell/structure format
% iter/iter_average: (average) total number of ADMM iteration
% t: final t of truncatedL1L2
% fval_true/fval_end: objective value of the ground-truth/solution
% ReErr/err_average: (average) relative error
% psnr: PSNR of MRI and inpainting
% time: execution time
%%%%%%%%%%%%%%%%%%%%
%% initialize par
switch prob.type
    case 'cs_gaussian'
        par.prob='cs';
        fprintf('It is a test for compressed sensing using Gaussian sensing matrices ...\n')
        fprintf('The sensing matrix is of size %dX%d ...\n',prob.size(1),prob.size(2))
        fprintf('The sparsity of the ground-truth is %d ...\n',prob.s)
        fprintf('The noise level on the measurements is %.2f ...\n',prob.tau)
        disp_time=prob.s>25;
    case 'cs_dct'
        par.prob='cs';
        fprintf('It is a test for compressed sensing using over-sampled DCT sensing matrices ...\n')
        fprintf('The sensing matrix is of size %dX%d with refinement factor %d ...\n',prob.size(1),prob.size(2),prob.F)
        fprintf('The sparsity of the ground-truth is %d ...\n',prob.s)
        fprintf('The noise level on the measurements is %.2f ...\n',prob.tau)
        disp_time=prob.s>=20;
    case 'cs_time'
        par.prob='cs';
        par.test_time=1;
        par.t=prob.s-1;
        fprintf('It is a test for compressed sensing using Gaussian sensing matrices (time testing mode) ...\n')
        fprintf('The sensing matrix is of size %dX%d ...\n',prob.size(1),prob.size(2))
        fprintf('The sparsity of the ground-truth is %d ...\n',prob.s)
        fprintf('The noise level on the measurements is %.2f ...\n',prob.tau)
        disp_time=prob.size(2)>4000;
    case 'mri'
        par.prob='mri';
        fprintf('It is a test for MRI reconstruction from partial Fourier measurements ...\n')
        fprintf('The testing Shepp-Logan phantom image is of size %dX%d ...\n',prob.size(1),prob.size(1))
        fprintf('The number of radial lines is %d ...\n',prob.s)
        fprintf('The noise level on the measurements is %.2f ...\n',prob.tau)
        disp_time=1;
    case 'mc_exact'
        par.prob='mc';
        fprintf('It is a test for matrix completion using exact low rank matrices ...\n')
        fprintf('The testing matrix is of size %dX%d ...\n',prob.size(1),prob.size(2))
        fprintf('The rank of the ground-truth is %d ...\n',prob.s)
        fprintf('The sampling ratio (SR) is %.4f ...\n',prob.p)
        fprintf('The degree of freedom ratio (FR) is %.2f ...\n',prob.s*(prob.size(1)+prob.size(2)-prob.s)/(prob.p*prob.size(1)*prob.size(2)))
        fprintf('The noise level on the measurements is %.2f ...\n',prob.tau)
        disp_time=prob.s>25;
    case 'mc_approximate'
        par.prob='mc';
        fprintf('It is a test for matrix completion using approximate low rank matrices ...\n')
        fprintf('The testing matrix is of size %dX%d ...\n',prob.size(1),prob.size(2))
        fprintf('The sampling ratio (SR) is %.4f ...\n',prob.p)
        fprintf('The degree of freedom ratio (FR) is %.2f (calculated using rank=10) ...\n',10*(prob.size(1)+prob.size(2)-10)/(prob.p*prob.size(1)*prob.size(2)))
        fprintf('The noise level on the measurements is %.2f ...\n',prob.tau)
        disp_time=1;
    case 'mc_time'
        par.prob='mc';
        par.test_time=1;
        par.t=prob.s-1;
        fprintf('It is test for matrix completion using exact low rank matrices (time testing mode) ...\n')
        fprintf('The testing matrix is of size %dX%d ...\n',prob.size(1),prob.size(2))
        fprintf('The rank of the ground-truth is %d ...\n',prob.s)
        fprintf('The sampling ratio (SR) is %.4f ...\n',prob.p)
        fprintf('The degree of freedom ratio (FR) is %.2f ...\n',prob.s*(prob.size(1)+prob.size(2)-prob.s)/(prob.p*prob.size(1)*prob.size(2)))
        fprintf('The noise level on the measurements is %.2f ...\n',prob.tau)
        disp_time=prob.size(2)>500;
    case 'inpainting'
        par.prob='inpainting';
        disp1='image inpainting';
        fprintf('It is test for image inpainting using image patches ...\n')
        fprintf('The testing patch is #%d of size 50X50 ...\n',prob.size(1))
        fprintf('The sampling ratio (SR) is %.4f ...\n',prob.p)
        fprintf('The degree of freedom ratio (FR) is 0.38 (calculated using rank=5) ...\n')
        fprintf('The noise level on the measurements is %.2f ...\n',prob.tau)
        disp_time=0;
end
if disp_time
    fprintf(2,'Warning: this test may take a bit long ...\n')
end
par.lambda=lambda;
par.detail=prob.detail;
%% begin batch test
time1=0;
if iscell(maxtest)
    test_num=cell2mat(maxtest);
else
    test_num=1:maxtest;
end
test_number=length(test_num);
res.good_trials=zeros(1,test_number);
for i=1:test_number
    switch i
        case 1
            disp_th='st';
        case 2
            disp_th='nd';
        case 3
            disp_th='rd';
        otherwise
            disp_th='th';
    end
    if ~prob.detail
        fprintf([' Taking the %d' disp_th ' of total %d test(s) ... '],i,test_number)
    else
        fprintf([' Taking the %d' disp_th ' of total %d test(s) ... \n'],i,test_number)
    end
    %generate test data
    [A,x_t,B,tau_norm,norm_X]=make_data(prob,test_num(i));
    %call truncatedL1L2
    time_temp=tic;
    par.norm_X=norm_X;par.tau=tau_norm;par.x_t=x_t;
    [x,outc,outs]=truncatedL1L2(A,B,par);
    time_temp=toc(time_temp);
    time1=time1+time_temp;
    %generate res
    res.outc{i}=outc;
    res.outs{i}=outs;
    res.par=outs.par;
    res.iter(i)=sum(outs.iter_in);
    res.t(i)=outs.t_end;
    res.fval_true(i)=outs.fval_true;
    res.fval_end(i)=outs.fval_end;
    res.ReErr(i)=norm(x-x_t,'fro')/norm(x_t,'fro');
    if any(strcmp(prob.type,{'mri','inpainting'}))
        res.psnr(i)=PSNR(x_t*norm_X,real(x)*norm_X);
    end
    if res.ReErr(i)<1e-3
        res.good_trials(i)=1;
    end
    if strcmp(prob.type,'inpainting')
        res.x{i}=x*norm_X;res.x_t{i}=x_t*norm_X;res.x_n{i}=B*norm_X;
    end
    fprintf('done in %.2fs with ReErr=%.4e\n',time_temp,res.ReErr(i))
end
res.prob=prob;res.lambda=lambda;
res.success_rate=sum(res.good_trials)/test_number;
res.time=time1;
if isfield(res,'iter')
    res.iter_average=sum(res.iter)/test_number;
end
res.err_average=mean(res.ReErr);
fprintf('All tests are done in %.2fs with success rate %.2f%% and average ReErr %.4e\n',res.time,res.success_rate*100,res.err_average)
end
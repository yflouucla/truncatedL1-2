function [X,outc,outs,X_all]=truncatedL1L2(A,B,par)
%% description
%for solving model
%min_X : lambda*(||X||_1-||X||_{t,1+2})+F(A*X-B)
%F(X)=\delta_{||X||_F<=tau}(X) constrained
%F(X)=1/2||X||_F^2 unconstrained
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%input:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%A: operator (needed)
%  for compressed sensing: sensing matrix
%  for matrix completion or inpainting: matrix with 0 for unknown and 1 for known, double or logical are both supported
%  for mri reconstruction: mask on FFT measurements
%B: measurements (needed)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%par: parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  parameters for the model (fields prob and lambda are needed, tau is needed for the constrained model)
%    par.prob: (needed)
%      'cs' for compressed sensing
%      'mc' for matrix completion
%      'mri' for magnetic resonance imaging
%      'inpainting' for image inpainting
%    par.lambda: regularization parameter (needed)
%      for noiseless cases, a small lambda (e.g., 1e-6) is suitable
%      for noisy cases, one needs to tune lambda to obtain the best performance
%    par.reg: regularization term (optional)
%      'L1': l_1, i.e., ||x||_1
%      'L1-L2': l_{1-2}, i.e., ||x||_1-||x||_2
%      'TL1-L2': truncated l_{1-2}, i.e., ||x||_{t,1-2} (default)
%    par.type:  (optional)
%      'con' for the constrained model (incompletely tested)
%      'uncon' for the unconstrained model (default)
%    par.tau: noise var (needed for constrained model)
%    par.sensingmatrix: sensing matrix for CS and MRI (optional)
%      0 for random sensing matrix (default for CS)
%      1 for partial FFT matrix (default for MRI)
%      2 for partial DCT matrix (untested)
%    par.regCS: transform domain for CS (optional)
%      'L1' (default for CS)
%      'HaarM': 2D Haar (untested)
%      'HaarV': 1D Haar (untested)
%      'TV': total variation (default for MRI)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  parameters for TL1-L2 (all optional except par.t for fixed selection of t)
%    par.alpha: PL1-alpha*L2 (default 1)
%    par.t_c: 0 for fixed selection of t and 1,2,3 for adaptive selection of t (default 1)
%      1: use linearly growing thresholdings as in the paper (default)
%      2: use customized thresholdings
%      3: use customized t values
%    par.t: truncated number (needed if use fixed t)
%    par.t_t: target thresholding, \theta in (4.11) of the paper (default values vary for different problems)
%      t_c=1: target thresholding, e.g., par.t_t=0.9
%      t_c=2: customized thresholdings, e.g., par.t_t=[0.3 0.6 0.7 0.8 0.85 0.9]
%      t_c=3: customized t values, e.g., par.t_t=[100 200 300 350 400 425 450 475 485 490 495 500]
%    par.t_d: step for t, \mu in (4.11) of the paper (default values vary for different problems)
%    par.iter_out_min: maximal iteration number for the continuation on t, k_0 in (4.11) of the paper (default values vary for different problems)
%    par.t_mean: compute t using normalized data, untested (default 0)
%    par.t_min: impose minimal t, untested (default 0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  parameters for the DCA and ADMM (optional)
%    par.c: DCA parameter, c in (4.2) of the paper (default 0)
%    par.beta: penalty parameters in the augmented Lagrangian for ADMM (default values vary for different problems)
%      par.beta(1): \beta in (4.7) of the paper
%      par.beta(2): for the unconstrained model, the weight for the fidelity term, i.e., par.beta(2)=1
%                   for the constrained model, the penalty parameter for the fidelity constraint, i.e., par.beta(2)=par.beta(1)
%    par.tol: stopping tolerances for outer and inner loops (default [1e-5,1e-5])
%    par.maxiter: maximal outer and inner iteration numbers (default values vary for different problems)
%    par.protect: avoid false convergence
%      'false_converge': not stop until reaching stopping criteria in small iteration numbers 3 times (untested)
%      'iter_in_min': impose minimal inner iteration number; see (4.13) of the paper (default)
%      0: do nothing, for time testing modes, i.e., cs_time and mc_time
%    par.iter_in_min: minimal inner iteration number if par.protect='iter_in_min', i.e., l_min in (4.13) of the paper (default 0.2*par.maxiter(2))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  other parameters (optional)
%    par.x_t: ground-truth, evaluation purpose only
%    par.sparsity: true sparsity/rank
%    par.eps_min: a small constant for robustness (default 1e-10)
%    par.proj: project the solution to [0,1] or not (default 0)
%    par.norm_X: if normalize the ground-truth to have unit 2-norm, norm_X is the 2-norm of the ground-truth,
%                if not normalize, norm_X=1 (default)
%                we suggest the users to do normalization for matrix completion and inpainting
%    par.show_fval: compute objective function values (default 1)
%    par.show_sparse: show support sets or singular values of intermediate solutions (default 0)
%    par.detail: show waitbar and print intermediate results to the screen or not, set to 0 to save time (default 1)
%    par.show_images: show images for mri and inpainting (default 1)
%    par.test_time: enter time testing mode, this mode needs par.x_t, par.t, and par.test_thre (default 0)
%    par.test_thre: stopping criterion for time testing mode (default 1e-5)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%output:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%X: solution
%outc/outs: output in cell/structure format with the following fields:
%  par: parameter setting
%  A/B/X: input A/input B/solution
%  i: outer iteration number
%  time_DCA: execution time
%  con_end: ||B-A*X||_F
%  spar_end: sparsity/rank of the solution
%  t_end: final t
%  fval_true/fval_end: objective function values of the ground-truth/solution
%  ReErr/psn_ini/psn_end/mean_org/mean_end: final ReErr/initial PSNR/final PSNR/mean of the ground-truth/mean of the solution
%  err/psn: ReErr/PSNR of intermediate solutions
%  cr_x: outer stopping criterion of intermediate solutions, i.e., {||X^{k+1}-X^k||F/||X^k||F}
%  cr_f: outer objective function value stopping criterion of intermediate solutions, i.e., {|F(X^{k+1})-F(X^k)|/|F(X^k)|}
%  iter_in: inner ADMM iteration numbers of intermediate solutions
%  cr_in: inner stopping criterion of intermediate solutions, i.e., {||X^{k,l+1}-X^{k,l}||F/||X^{k,l}||F}
%  t_all: values of t of intermediate solutions
%  con_Y/con_Z: check whether constraints in ADMM are satisfied, i.e., ||X-Y||_F for con_Y and ||Z-A*Y+B||_F for con_Z
%  fval: objective function values of intermediate solutions, i.e., {F(X^k)}
%X_all: support sets or singular values of intermediate solutions, need par.show_sparse=1
%% parameter extraction
time_DCA=tic;
prob=par.prob;if ~any(strcmp(prob,{'cs','mc','mri','inpainting'}));error('wrong input: par.prob');end
lambda=par.lambda;if lambda<0;error('wrong input: par.lambda');end
if ~isfield(par,'reg');par.reg='TL1-L2';end;
reg=par.reg;if ~any(strcmp(reg,{'L1','L1-L2','TL1-L2'}));error('wrong input: par.reg');end
if ~isfield(par,'type');par.type='uncon';end;
type=par.type;if ~any(strcmp(type,{'con','uncon'}));error('wrong input: par.type');end
switch prob
    case 'cs'
        if ~isfield(par,'sensingmatrix');par.sensingmatrix=0;end
        if ~isfield(par,'regCS');par.regCS='L1';end
        if ~isfield(par,'t_t');par.t_t=0.95;end
        if ~isfield(par,'t_d');par.t_d=0.1;end
        if ~isfield(par,'iter_out_min');par.iter_out_min=10;end
        beta_temp=lambda*100;
        if ~isfield(par,'maxiter');par.maxiter=[50,5000];end
        if ~isfield(par,'show_images');par.show_images=0;end
    case 'mc'
        A=logical(A);
        if ~isfield(par,'sensingmatrix');par.sensingmatrix=0;end
        if ~isfield(par,'regCS');par.regCS='L1';end
        if ~isfield(par,'t_t');par.t_t=0.95;end
        if ~isfield(par,'t_d');par.t_d=0.2;end
        if ~isfield(par,'iter_out_min');par.iter_out_min=5;end
        beta_temp=lambda*100;
        if ~isfield(par,'maxiter');par.maxiter=[100,1000];end
        if ~isfield(par,'show_images');par.show_images=0;end
    case 'mri'
        A=logical(A);
        prob='cs';
        if ~isfield(par,'sensingmatrix');par.sensingmatrix=1;end
        if ~isfield(par,'regCS');par.regCS='TV';end
        if ~isfield(par,'t_t');par.t_t=0.95;end
        if ~isfield(par,'t_d');par.t_d=0.03;end
        if ~isfield(par,'iter_out_min');par.iter_out_min=20;end
        beta_temp=lambda*10;
        if ~isfield(par,'maxiter');par.maxiter=[100,5000];end
        if ~isfield(par,'show_images');par.show_images=1;end
    case 'inpainting'
        A=logical(A);
        prob='mc';
        if ~isfield(par,'sensingmatrix');par.sensingmatrix=0;end
        if ~isfield(par,'regCS');par.regCS='L1';end
        if ~isfield(par,'t_t');par.t_t=0.7;end
        if ~isfield(par,'t_d');par.t_d=0.2;end
        if ~isfield(par,'iter_out_min');par.iter_out_min=5;end
        beta_temp=lambda*100;
        if ~isfield(par,'maxiter');par.maxiter=[50,1000];end
        if ~isfield(par,'show_images');par.show_images=1;end
end
SM=par.sensingmatrix;if ~ismember(SM,[0,1,2]);error('wrong input: par.sensingmatrix');end
regCS=par.regCS;if ~any(strcmp(regCS,{'L1','HaarM','HaarV','TV'}));error('wrong input: par.regCS');end
if ~isfield(par,'alpha');par.alpha=1;end;if(par.alpha<0 || par.alpha>1);error('wrong input: par.alpha');end
if ~isfield(par,'t_c');par.t_c=1;end;if ~ismember(par.t_c,[0,1,2,3]);error('wrong input: par.t_c');end
if(isfield(par,'t') && par.t<0);error('wrong input: par.t');end
if ~isfield(par,'t_mean');par.t_mean=0;end
if ~isfield(par,'t_min');par.t_min=0;end
if ~isfield(par,'c');par.c=0;end;c=par.c;if(par.c<0);error('wrong input: par.c');end
if ~isfield(par,'beta');par.beta(1)=beta_temp;
    if strcmp(type,'con');par.beta(2)=beta_temp;else par.beta(2)=1;end
elseif length(par.beta)==1;
    if strcmp(type,'con');par.beta(2)=par.beta(1);else par.beta(2)=1;end
end;beta1=par.beta(1);beta2=par.beta(2);if (beta1<0 || beta2<0);error('wrong input: par.beta');end
if ~isfield(par,'tol');par.tol=[1e-5,1e-5];end;tol_out=par.tol(1);tol_in=par.tol(2);
maxiter_out=par.maxiter(1);maxiter_in=par.maxiter(2);
if ~isfield(par,'protect');par.protect='iter_in_min';end
if(~any(strcmp(par.protect,{'false_converge','iter_in_min'})) && ~isequal(par.protect,0));error('wrong input: par.protect');end
if ~isfield(par,'iter_in_min');par.iter_in_min=ceil(0.2*par.maxiter(2));end
if ~isfield(par,'eps_min');par.eps_min=1e-10;end
if ~isfield(par,'proj');par.proj=0;end;proj=par.proj;
if ~isfield(par,'norm_X');par.norm_X=1;end
if ~isfield(par,'show_fval');par.show_fval=1;end;show_fval=par.show_fval;
if ~isfield(par,'show_sparse');par.show_sparse=0;end;show_sparse=par.show_sparse;
if ~isfield(par,'detail');par.detail=1;end
if ~isfield(par,'test_time');par.test_time=0;end
if par.test_time
    show_sparse=0;par.detail=0;par.show_images=0;
    par.t_c=0;par.protect=0;par.iter_in_min=0;
    if ~isfield(par,'test_thre');par.test_thre=1e-5;end
end
switch regCS
    case 'L1'
        W=@(x)x;WT=@(x)x;
    case 'HaarM'
        W=@(x)mdwt(x,[sqrt(2)/2 sqrt(2)/2]);
        WT=@(y)midwt(y,[sqrt(2)/2 sqrt(2)/2]);
    case 'HaarV'
        W=@(x)reshape(mdwt(reshape(x,[sqrt(length(x)) sqrt(length(x))]),[sqrt(2)/2 sqrt(2)/2]),[length(x) 1]);
        WT=@(x)reshape(midwt(reshape(x,[sqrt(length(x)) sqrt(length(x))]),[sqrt(2)/2 sqrt(2)/2]),[length(x) 1]);
    case 'TV'
        W=@(x)[Dxforward(x) Dyforward(x)];
        WT=@(x)-Dxbackward(x(:,1:size(x,2)/2))-Dybackward(x(:,size(x,2)/2+1:end));
end
par.W=W;par.WT=WT;
%% initialization1
cr_in=zeros(1,maxiter_out);
cr_x=zeros(1,maxiter_out);
cr_f=zeros(1,maxiter_out);
iter_in=zeros(1,maxiter_out);
con_Y=zeros(1,maxiter_out);
con_Z=zeros(1,maxiter_out);
err=zeros(1,maxiter_out);
psn=zeros(1,maxiter_out);
fval=zeros(1,maxiter_out);
t_all=zeros(1,maxiter_out);
if show_sparse
    switch prob
        case 'cs'
            X_all=cell(size(A,2),maxiter_out);
        case 'mc'
            X_all=cell(min(size(A)),maxiter_out);
    end
else
    X_all=[];
end
%% initialization2
switch prob
    case 'cs'
        [M,N]=size(A);
        if ~strcmp(regCS,'TV')
            if ~SM
                sizeA=[N,1];
                L=chol(eye(M)/beta2+A*A'/(beta1+c),'Lower');U=L';
            else
                sizeA=[M N];
                denom=beta2*A+beta1+c;
            end
            S=zeros(sizeA);V=zeros(length(B),1);
        else%(c+beta2*A+beta1*|otfDx|^2+beta1*|otfDy|^2)
            sizeA=[M N];
            denom=beta2*A+beta1*(abs(psf2otf([1 -1 0],sizeA)).^2+abs(psf2otf([1;-1;0],sizeA)).^2)+c;
            S=zeros(size(W(zeros(sizeA))));
        end
        B=B(:);Z=zeros(length(B),1);
        X=zeros(sizeA);
    case 'mc'
        sizeA=size(A);
        if ~isequal(size(B),sizeA)
            B_temp=zeros(sizeA);
            B_temp(A)=B;
            B=B_temp;
        end
        X=zeros(sizeA);%X(A)=B(A);
        S=zeros(sizeA);V=zeros(sizeA);Z=zeros(sizeA);
end
if ~isfield(par,'x_t');par.x_t=X;end
t_now=0;
switch par.protect
    case 'false_converge'
        false_converge_limit=3;
        false_converge=0;
        iter_in_min=ceil(maxiter_in*0.1);
    case 'iter_in_min'
        iter_in_min=par.iter_in_min;
    case 0
        iter_in_min=0;
end
if strcmp(reg,'TL1-L2') && par.t_c
    iter_out_min=par.iter_out_min;
else
    iter_out_min=0;
end
%% DCA begin
for i=1:maxiter_out
    Xpout=X;false_converge_c=0;
    %% compute P
    switch reg
        case 'L1'
            P=zeros(sizeA);
        case 'L1-L2'
            norm_X=norm(X,'fro');
            if norm_X<par.eps_min
                P=zeros(sizeA);
            else
                P=X/norm_X;
            end
        case 'TL1-L2'
            switch prob
                case 'cs'
                    X_P=W(X);sizeP=size(X_P);X_P=X_P(:);
                    supp_temp=abs(X_P)>par.eps_min;%supp of X
                    X_temp=X_P(supp_temp);
                    [X_abs_sorted,ind_temp]=sort(abs(X_temp),'descend');
                    X_sorted=X_temp(ind_temp);
                    [~,ind_inv]=sort(ind_temp);
                    r_temp=length(X_temp);
                case 'mc'
                    [U_mc,S_temp,V_mc]=svd(X,'econ');
                    if all(size(S_temp)-1)
                        S_temp=diag(S_temp);
                    end
                    S_temp=S_temp(:);
                    r_temp=sum(S_temp>par.eps_min);%rank of X
                    if par.t_mean
                        S_temp_t=svd(X-mean(mean(X)),'econ');
                        r_temp_t=sum(S_temp_t>par.eps_min);
                        X_abs_sorted=S_temp_t(par.t_min+1:r_temp_t);%compute t from par.t_min
                    else
                        X_abs_sorted=S_temp(par.t_min+1:r_temp);%compute t from par.t_min
                    end
            end
            if par.t_c
                if i<=iter_out_min+1
                    switch par.t_c
                        case 1
                            t_now=find_t(X_abs_sorted,par.t_t-(iter_out_min-i+1)*par.t_d)+par.t_min;
                        case 2
                            t_now=find_t(X_abs_sorted,par.t_t(i))+par.t_min;
                        case 3
                            t_now=par.t_t(i);
                    end
                    if isfield(par,'sparsity')
                        t_now=min(t_now,par.sparsity);
                    end
                end
            else
                t_now=par.t;
            end
            switch prob
                case 'cs'
                    if r_temp<=t_now
                        P_temp=sign(X_sorted(1:r_temp));
                    else
                        P_temp=[sign(X_sorted(1:t_now));...
                            par.alpha*X_sorted(t_now+1:r_temp)/norm(X_sorted(t_now+1:r_temp),'fro')];
                    end
                    P_temp=P_temp(ind_inv);
                    P=zeros(sizeP);
                    P(supp_temp)=P_temp;
                    P=WT(P);
                case 'mc'
                    if r_temp<=t_now
                        P_temp=ones(r_temp,1);
                    else
                        P_temp=[ones(t_now,1);...
                            par.alpha*S_temp(t_now+1:r_temp)/norm(S_temp(t_now+1:r_temp),'fro')];
                    end
                    P=U_mc(:,1:r_temp)*diag(P_temp)*V_mc(:,1:r_temp)';
            end
            P=P+c/lambda*X;
    end
    %% ADMM
    if par.detail
        hbar=waitbar(0,'ADMM start');
    end
    for j=1:maxiter_in
        Xpin=X;
        %for the unconstrained/constrained model:
        %lambda*(||WX||_1-<P,Y>)+delta(Z)+c/2*||Y||_2^2 s.t. X=Y Z=A*Y-B
        %L(X,Y,Z,S,V)=lambda*(||WX||_1-<P,Y>)+delta(Z)+c/2*||Y||_2^2
        %             +beta1/2*||X-Y+S||_2^2+beta2/2*||Z-A*Y+B+V||_2^2
        % unconstrained: delta(Z)=0, Z=0 (fixed), V=0 (fixed), beta2=1
        % constrained: delta(Z)=\delta_{||Z||_F<=tau}(Z), Z and V are updated, beta2=beta1
        %for the MRI model:
        %lambda*(||DX||_1-<P,X>)+c/2*||X||_2^2+1/2*||A*X-B||_2^2
        %lambda*(||Y||_1-<P,X>)+c/2*||X||_2^2+1/2*||A*X-B||_2^2 s.t. Y=DX
        %L(X,Y,S)=lambda*(||Y||_1-<P,X>)+c/2*||X||_2^2+beta2/2*||A*X-B||_2^2+beta1/2*||Y-DX+S||_2^2
        %% compute Y
        switch prob
            case 'cs'
                if ~strcmp(regCS,'TV')
                    %-lambda*<P,Y>+c/2*||Y||_2^2+beta1/2*||X-Y+S||_2^2+beta2/2*||Z-A*Y+B+V||_2^2
                    switch SM
                        case 0%A is Gaussian matrix
                            Y_temp=beta1*(X+S)+lambda*P+beta2*A'*(Z+B+V);
                            %Y=((beta1+c)*I+beta2*A'A)^(-1)*Y_temp
                            %(A+UBV)^(-1)=A^(-1)-A^(-1)U(B^(-1)+VA^(-1)U)^(-1)VA^(-1)
                            %A=(beta1+c)*I,U=A',B=beta2*I,V=A
                            %(beta2*A'A+(beta1+c)*I)^(-1)=1/(beta1+c)*I-A'*(1/beta2*I+AA'/(beta1+c))^(-1)*A/(beta1+c)/(beta1+c)
                            %Y=Y_temp-A'*(A_temp*(A*Y_temp));
                            %test
                            %a=30;b=120;A=randn(a,b);beta1=rand;beta2=rand;B=randn(b,1);
                            %e1=inv(beta2*A'*A+beta1*eye(b))*B;
                            %L=chol(eye(a)/beta2+A*A'/beta1,'Lower');U=L';
                            %e2=B/beta1-A'*(U\(L\(A*B)))/beta1/beta1;
                            %norm(e1-e2,'fro')
                            Y=Y_temp/(beta1+c)-A'*(U\(L\(A*Y_temp)))/(beta1+c)/(beta1+c);
                        case 1%A is FFT matrix
                            %-lambda*<P,Y>+c/2*||Y||_2^2+beta1/2*||X-Y+S||_2^2+beta2/2*||Z-A*Y+B+V||_2^2
                            %-lambda*<FP,FY>+c/2*||FY||_F^2+beta1/2*||FY-F(X+S)||_F^2+beta2/2*||A.*FY-sqrt(MN)(Z+B+V)||_F^2
                            %-lambda*FP+c*FY+beta1*(FY-F(X+S))+beta2*(A.*FY-A^-1sqrt(MN)(Z+B+V))=0
                            %(beta2*A+beta1+c)FY=lambda*FP+beta1*F(X+S)+beta2*A^-1sqrt(MN)(Z+B+V)
                            Y_temp1=zeros(sizeA);Y_temp1(A)=sqrt(M*N)*(B+Z+V);
                            Y_temp=fft2(beta1*(X+S)+lambda*P)+beta2*Y_temp1;
                            %Y=real(ifft2(Y_temp./denom));
                            Y=(ifft2(Y_temp./denom));
                        case 2%A is DCT matrix
                            Y_temp1=zeros(sizeA);Y_temp1(A)=(B+Z+V);
                            Y_temp=dct2(beta1*(X+S)+lambda*P)+beta2*Y_temp1;
                            Y=idct2(Y_temp./denom);
                    end
                else%lambda*||Y||_1+beta1/2*||Y-DX+S||_2^2
                    Y=thresholdings(W(X)-S,lambda/beta1,'L1');
                end
            case 'mc'
                %-lambda*<P,Y>+c/2*||Y||_2^2+beta1/2*||X-Y+S||_2^2+F(A.*Y-B)
                Y_temp=beta1*(X+S)+lambda*P;
                Y=zeros(sizeA);
                Y(~A)=Y_temp(~A)/(beta1+c);
                switch type
                    case 'con'
                        if par.tau==0
                            Y(A)=B(A);
                        else
                            Y(A)=B(A)+(Y_temp(A)/(beta1+c)-B(A))*par.tau/max(par.tau,norm(Y_temp(A)/(beta1+c)-B(A),'fro'));
                        end
                    case 'uncon'
                        Y(A)=(B(A)+Y_temp(A))/(beta1+c+1);
                end
                if proj
                    Y=min(max(0,Y),1/par.norm_X);
                end
        end
        %% compute X
        switch prob
            case 'cs'
                if ~strcmp(regCS,'TV')
                    %lambda*||WX||_1+beta1/2*||X-(Y-S)||_2^2
                    %lambda*||WX||_1+beta1/2*||WX-W(Y-S)||_2^2
                    X=WT(thresholdings(W(Y-S),lambda/beta1,'L1'));
                else%-lambda<P,X>+c/2*||X||_2^2+beta2/2*||A*FX/sqrt(MN)-B||_2^2+beta1/2*||Y-DX+S||_2^2
                    %-lambda<FP,FX>+c/2*||FX||_2^2+beta2/2*||A*FX-sqrt(MN)*B||_2^2
                    %  +beta1/2*||otfDx*FX-FY1-FS1||_2^2+beta1/2*||otfDy*FX-FY2-FS2||_2^2
                    %-lambda*FP+cFX+beta2*A.*(A.*FX-sqrt(MN)*B)+beta1*conj(otfDx)*(otfDx*FX-FY1-FS1)
                    %  +beta1*conj(otfDy)*(otfDy*FX-FY2-FS2)=0
                    %(c+beta2*A+beta1*|otfDx|^2+beta1*|otfDy|^2)FX=
                    %   lambda*FP+beta2*sqrt(MN)*B+beta1*conj(otfDx)*(FY1+FS1)+beta1*conj(otfDy)*(FY2+FS2)
                    %(c+beta2*A+beta1*|otfDx|^2+beta1*|otfDy|^2)FX=beta2*sqrt(MN)*B+F(lambda*P+beta1*WT(Y+S))
                    X_temp1=zeros(sizeA);X_temp1(A)=sqrt(M*N)*B;
                    X_temp=fft2(beta1*WT(Y+S)+lambda*P)+beta2*X_temp1;
                    X=(ifft2(X_temp./denom));
                end
            case 'mc'
                X=SVT(Y-S,lambda/beta1,'L1');
        end
        %% compute Z
        if strcmp(prob,'cs') && strcmp(type,'con')
            %delta(Z)+beta2/2*||Z-A*Y+B+V||_2^2
            if par.tau==0
                Z=zeros(M,1);
            else
                switch SM
                    case 0
                        Z=(A*Y-B-V)*par.tau/max(par.tau,norm(A*Y-B-V,'fro'));
                    case 1
                        Z_temp=fft2(Y);Z_temp=Z_temp(A)/sqrt(M*N);
                        Z=(Z_temp-B-V)*par.tau/max(par.tau,norm(Z_temp-B-V,'fro'));
                    case 2
                        Z_temp=dct2(Y);Z_temp=Z_temp(A);
                        Z=(Z_temp-B-V)*par.tau/max(par.tau,norm(Z_temp-B-V,'fro'));
                end
            end
        end
        %% compute S and V
        if ~strcmp(regCS,'TV')
            S=S+X-Y;
        else
            S=S+Y-W(X);
        end
        if strcmp(prob,'cs') && strcmp(type,'con')
            switch SM
                case 0
                    V=V+Z-A*Y+B;
                case {1,2}
                    V=V+Z-Z_temp+B;
            end
        end
        %% stop inner loop
        cr_in1=norm(X-Xpin,'fro')/max(norm(Xpin,'fro'),par.eps_min);
        %cr_in1=norm(real(X-Xpin),'fro')/max(norm(real(Xpin),'fro'),par.eps_min);
        if par.detail
            waitbar(j/maxiter_in,hbar,sprintf('Out: %d, In: %d, Cr: %.4e',i,j,cr_in1));
        end
        if cr_in1<tol_in
            stop_flag=1;
            if strcmp(reg,'TL1-L2')
                switch par.protect
                    case 'false_converge'
                        if j<iter_in_min && i>=iter_out_min+1 && ~false_converge_c && false_converge<=false_converge_limit
                            false_converge=false_converge+1;
                            false_converge_c=1;
                            %[i j false_converge]
                        end
                        stop_flag= (j>=iter_in_min || i<iter_out_min+1 || false_converge>=false_converge_limit+1);
                    case 'iter_in_min'
                        stop_flag= (i<iter_out_min+1 || j>=iter_in_min);
                    case 0
                        stop_flag=1;
                end
            end
            if i==1 && j<=50
                stop_flag=0;
            end
            if stop_flag
                break;
            end
        end
    end
    if par.detail
        close(hbar);
    end
    %% stop outer loop
    if show_sparse
        X_show=show_sparse_X(W(X),par);
        X_all(1:length(X_show),i)=X_show;
    end
    cr_in(i)=cr_in1;
    cr_x1=norm(X-Xpout,'fro')/max(norm(Xpout,'fro'),par.eps_min);
    %cr_x1=norm(real(X-Xpout),'fro')/max(norm(real(Xpout),'fro'),par.eps_min);
    cr_x(i)=cr_x1;
    stop_flag=cr_x1<tol_out && i>=iter_out_min+1;
    if show_fval
        fval(i)=fun_fval(X,A,B,prob,type,lambda,reg,par,t_now);
        if i>=2
            cr_fval=abs(fval(i)-fval(i-1))/max(fval(i-1),par.eps_min);
        else
            cr_fval=1;
        end
        cr_f(i)=cr_fval;
        stop_flag=(cr_fval<tol_out || cr_x1<tol_out) && i>=iter_out_min+1;
    end
    iter_in(i)=j;
    if ~strcmp(regCS,'TV')
        con_Y(i)=norm(X-Y,'fro');
    else
        con_Y(i)=norm(W(X)-Y,'fro');
    end
    if strcmp(prob,'cs')
        switch SM
            case 0
                con_Z(i)=norm(Z-A*Y+B,'fro');
            case 1
                if ~strcmp(type,'con')
                    if ~strcmp(regCS,'TV')
                        Z_temp=fft2(Y);
                    else
                        Z_temp=fft2(X);
                    end
                    Z_temp=Z_temp(A)/sqrt(M*N);
                end
                con_Z(i)=norm(Z-Z_temp+B,'fro');
            case 2
                if ~strcmp(type,'con')
                    Z_temp=dct2(Y);Z_temp=Z_temp(A);
                end
                con_Z(i)=norm(Z-Z_temp+B,'fro');
        end
    end
    err(i)=norm(X-par.x_t,'fro')/norm(par.x_t,'fro');
    psn(i)=PSNR(real(X)*par.norm_X,par.x_t*par.norm_X);
    t_all(i)=t_now;
    if par.detail
        if i==1
            result_title='OutIt| InnIt|   PSNR|    ReErr|   OutCrX|   OutCrF|   InnCrX| Trun#';
            disp(result_title)
        end
        result_sprintf=sprintf('%5d| %5d| %6.2f| %8.2e| %8.2e| %8.2e| %8.2e| %5d',...
            i,iter_in(i),psn(i),err(i),cr_x(i),cr_f(i),cr_in(i),t_all(i));
        disp(result_sprintf)
        if par.show_images
            if any(size(X)==1)
                size_temp=floor(sqrt(numel(X)));
                imshow(reshape(real(X(1:(size_temp^2)))*par.norm_X,[size_temp size_temp]))
            else
                imshow(real(X)*par.norm_X)
            end
            set(gcf,'position',get(0,'screensize'));
            title({result_title result_sprintf})
            drawnow;
        end
    end
    if ~par.test_time%not record time
        if stop_flag
            break;
        end
    else%record time
        %if norm(X-Xpout,'fro')/norm(Xpout,'fro')<par.test_thre
        if norm(X-par.x_t,'fro')/norm(par.x_t,'fro')<par.test_thre
            break;
        end
    end
end
if strcmp(prob,'mc') && isfield(par,'tau') && par.tau==0
    X(A)=B(A);
end
X=real(X);
time_DCA=toc(time_DCA);
%% save data
if show_sparse
    X_show=show_sparse_X(par.x_t,par);
    X_all(1:length(X_show),i+2)=X_show;
end
switch prob
    case 'cs'
        spar_end=sum(sum(abs(W(X))>par.eps_min));
        switch SM
            case 0
                con_end=norm(B-A*X,'fro');
            case 1
                con_end_temp=fft2(X);con_end_temp=con_end_temp(A)/sqrt(M*N);
                con_end=norm(B-con_end_temp,'fro');
            case 2
                con_end_temp=dct2(X);con_end_temp=con_end_temp(A);
                con_end=norm(B-con_end_temp,'fro');
        end
        psn_ini=0;
    case 'mc'
        spar_end=sum(svd(X,'econ')>par.eps_min);
        con_end=norm(B(A)-X(A),'fro');
        psn_ini=PSNR(B*par.norm_X,par.x_t*par.norm_X);
end
t_end=t_now;
ReErr=norm(X-par.x_t,'fro')/norm(par.x_t,'fro');
psn_end=PSNR(real(X)*par.norm_X,par.x_t*par.norm_X);
mean_org=mean(par.x_t(:));mean_end=mean(X(:));
fval_true=fun_fval(par.x_t,A,B,prob,type,lambda,reg,par,t_now);
fval_end=fval(i);
[outc,outs]=showvars({[4 7 5] i},par,A,B,X,...
    i,time_DCA,con_end,spar_end,t_end,fval_true,fval_end,...
    ReErr,psn_ini,psn_end,mean_org,mean_end,...
    err,psn,cr_x,cr_f,iter_in,cr_in,t_all,con_Y,con_Z,fval);
end

function v=fun_fval(X,A,B,prob,type,lambda,reg,par,t_now)
switch prob
    case 'cs'
        switch reg
            case 'L1'
                v1=lambda*sum(sum(abs(par.W(X))));
            case 'L1-L2'
                v1=lambda*(sum(abs(X))-norm(X,'fro'));
            case 'TL1-L2'
                v1=lambda*(TL1L2norm(par.W(X),t_now,par.alpha));
        end
    case 'mc'
        switch reg
            case 'L1'
                v1=lambda*nuclearnorm(X);
            case 'L1-L2'
                v1=lambda*(nuclearnorm(X)-norm(X,'fro'));
            case 'TL1-L2'
                X_temp=svd(X,'econ');
                v1=lambda*(TL1L2norm(X_temp,t_now,par.alpha));
        end
end
switch type
    case 'con'
        v2=0;
    case 'uncon'
        switch prob
            case 'cs'
                switch par.sensingmatrix
                    case 0
                        v2=1/2*norm(A*X-B,2)^2;
                    case 1
                        v2_temp=fft2(X);v2_temp=v2_temp(A)/sqrt(numel(A));
                        v2=1/2*norm(v2_temp-B,'fro')^2;
                    case 2
                        v2_temp=dct2(X);v2_temp=v2_temp(A);
                        v2=1/2*norm(v2_temp-B,'fro')^2;
                end
            case 'mc'
                v2=1/2*norm(X(A)-B(A),'fro')^2;
        end
end
v=v1+v2;
end

function val=TL1L2norm(Z,t,alpha)
%compute truncated l_{1-2} metric
Z=sort(abs(Z),'descend');
val=sum(Z(t+1:end))-alpha*sqrt(sum(Z(t+1:end).^2));
end

function X_show=show_sparse_X(X,par)
switch par.prob
    case 'cs'
        supp_temp=find(abs(X)>par.eps_min);
        [X_abs_sorted,ind_temp]=sort(abs(X(supp_temp)),'descend');
        ind_temp=supp_temp(ind_temp);
        r_temp=length(supp_temp);
        t=find_t(X_abs_sorted,par.t_t(end));
        %t=sum((tril(ones(r_temp))*X_abs_sorted)<(par.t_t*sum(X_abs_sorted)));
        X_show=[[num2str(r_temp) ' ' num2str(t) ' ' num2str(sum(X_abs_sorted),'%.2f')];...
            num2cell([num2str(X(ind_temp),'%.2f') char(32*ones(r_temp,1)) num2str(ind_temp)],2)];
    case 'mc'
        X_abs_sorted=svd(X,'econ');
        X_abs_sorted=X_abs_sorted(X_abs_sorted>par.eps_min);
        r_temp=length(X_abs_sorted);
        t=sum((tril(ones(r_temp))*X_abs_sorted)<(par.t_t(end)*sum(X_abs_sorted)));
        X_show=[[num2str(r_temp) ' ' num2str(t) ' ' num2str(sum(X_abs_sorted),'%.2f')];...
            num2cell(num2str(X_abs_sorted,'%.2f'),2)];
end
end

function ux=Dxbackward(u,BoundaryCondition)
if nargin<2
    BoundaryCondition='circular';
end
switch BoundaryCondition
    case 'circular'
        ux=[u(:,1)-u(:,end) diff(u,1,2)];
    case 'symmetric'
        ux=[zeros(size(u,1),1) diff(u,1,2)];
    case 'zero'
        ux=[u(:,1) diff(u,1,2)];
end
end

function ux=Dxforward(u,BoundaryCondition)
if nargin<2
    BoundaryCondition='circular';
end
switch BoundaryCondition
    case 'circular'
        ux=[diff(u,1,2) u(:,1)-u(:,end)];
    case 'symmetric'
        ux=[diff(u,1,2) zeros(size(u,1),1)];
    case 'zero'
        ux=[diff(u,1,2) -u(:,end)];
end
end

function uy=Dybackward(u,BoundaryCondition)
if nargin<2
    BoundaryCondition='circular';
end
switch BoundaryCondition
    case 'circular'
        uy=[u(1,:)-u(end,:);diff(u,1,1)];
    case 'symmetric'
        uy=[zeros(1,size(u,2));diff(u,1,1)];
    case 'zero'
        uy=[u(1,:);diff(u,1,1)];
end
end

function uy=Dyforward(u,BoundaryCondition)
if nargin<2
    BoundaryCondition='circular';
end
switch BoundaryCondition
    case 'circular'
        uy=[diff(u,1,1);u(1,:)-u(end,:)];
    case 'symmetric'
        uy=[diff(u,1,1);zeros(1,size(u,2))];
    case 'zero'
        uy=[diff(u,1,1);-u(end,:)];
end
end

function t=find_t(X,a,type)
%find maximal t such that sum(X(1:t))<a (type=fixed) or
%sum(X(1:t))<a*sum(X(:)) (type=adaptive)
%(X,a) should be nonnegative
%X should be in descend order
X=sort(abs(X(:)),'descend');
if nargin==2
    type='adaptive';
end
if strcmp(type,'adaptive')
    a=a*sum(X);
end
if numel(X)==0 || X(1)>=a
    t=0;
    return;
end
n=length(X);
if sum(X)<a
    t=n;
    return;
end
tmin=1;
if sum(X(1:min(100,n)))>=a
    tmax=min(100,n);
else
    tmax=n;
end
while tmax-tmin>=2
    ttemp=floor((tmin+tmax)/2);
    if sum(X(1:ttemp))<a
        tmin=ttemp;
    else
        tmax=ttemp;
    end
    %disp([tmin tmax])
end
t=tmin;
end

function X=SVT(Z,tau,type)
%for computing SVT of a matrix
if nargin<3
    type='L1';
end
[m,n]=size(Z);
if m <= n
    AAT=Z*Z';
    [S,V]=eig(AAT);
    V=max(0,diag(V));
    %S=fliplr(S);
    %V=sort(diag(V),'descend');
    V=sqrt(V);
    tol=n*eps(max(V));
    mid=thresholdings(V,tau(end:-1:1),type);
    ind=mid>tol;
    if any(ind)
        mid=mid(ind)./V(ind);
        X=S(:,ind)*diag(mid)*S(:,ind)'*Z;
    else
        X=zeros(m,n);
    end
else
    X=SVT(Z',tau,type);
    X=X';
end
end

function x=thresholdings(y,tau,type)
%solve: min_x: 1/2(x-y)^2+tau*f(x)
%type:
% 'L1': f(x)=|x|
% 'L1/2': f(x)=sqrt(x)
if ~exist('type','var')
    type='L1';
end
switch type
    case 'L1'
        x=sign(y).*max(0,abs(y)-tau);
    case 'L1/2'
        ind=abs(y)>3/2*tau.^(2/3);
        x=zeros(size(y));
        y=y(ind);
        x(ind)=2/3*y.*(1+cos(2/3*(pi-acos(tau/4.*(abs(y)/3).^(-3/2)))));
end
end

function [A,As]=showvars(varargin)
%% Description
%for displaying results in cell and structure formats
%%
flag=varargin{1};
a=flag{1};
if numel(flag)>1
    num=flag{2};
else
    num=0;
end
var=varargin(2:end);
row=max(max(2*a),num+2);
name=cell(1,nargin-1);
for i=2:nargin
    name{i-1}=inputname(i);
end
%%
n=length(a);
num1=sum(a);
A1=cell(row,n);
name1=name(1:num1);
var1=var(1:num1);
k=0;
for i=1:n
    for j=1:a(i)
        k=k+1;
        A1{2*j-1,i}=name1{k};
        A1{2*j,i}=var1{k};
    end
end
%%
b=nargin-1-num1;
A2=cell(row,b);
name2=name(num1+1:num1+b);
var2=var(num1+1:num1+b);
for i=1:b
    A2(1:(num+2),i)=[name2{i} num2cell(var2{i}(1:num)) name2{i}];
end
%%
A=[A1 A2];
%%
for i=1:nargin-1
    As.(name{i})=var{i};
end
end
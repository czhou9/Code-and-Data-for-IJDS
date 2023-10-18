% In the second method of paramter selection by CV, we let the 5/10 fold
% data share the same projection matrix derived from the 400 training data
% which increases selection computation. The results is similar to
% 'CVproposed'.
%% cross-validation
% set the total number of training samples
Nt = 400; 
% We use the first 400 tensor samples as the training samples
X_4Dcv = X_4D(:,:,:,1:Nt); 
Ym_tcv = Ym_t(1:Nt); 

% Initilize full rank U1,U2,U3 from MPCA
X_4Dm = X_4Dcv;
X_4Dm(X_4Dm==0)=-1;
X_MPCA = X_4Dm;
Ym_MPCA = Ym_tcv;

TX = X_MPCA;
gndTX = Ym_MPCA;
testQ = 100;
maxK = 1;
[tUs, odrIdx, TXmean, Wgt]  = MPCA(TX,gndTX,testQ,maxK);

% set the number of folds in CV
n_fold = 5; 
% set the max low-rank number for 1st, 2nd, 3rd mode 
%(We can also set the min number to 3 for decreasing computation load)
LRcv1 = 4;  
LRcv2 = 4;
LRcv3 = 4;
% For the convenience of storing data and runing the cluster, we change
% alpha to integer in the code. Alpha in code = 10*(real alpha + 0.1). The
% range of alpha is set 0:0.1:1 which has 11 alternatives. If needed, we
% can set small division value such as 0:0.05:1, e.g. And 'parfor' can be
% used here to increase the computation speed.
N_alpha = 11;
I1 = size(X_4D,1);
I2 = size(X_4D,2);
I3 = size(X_4D,3);
% Here we use the median of the Error matrix as the criteria to select the
% best (p1, p2, p3) combination; We can also use the mean, variance, IQR, max/min
% as the criteria for different goals. And the results of different
% critera are similar.
Er = cell(LRcv1,LRcv2,LRcv3,N_alpha);
Er_median = cell(LRcv1,LRcv2,LRcv3,N_alpha);
%Er_mean = cell(LRcv1, LRcv2, LRcv3,11);
%Er_var = cell(LRcv1, LRcv2, LRcv3,11);
   
   for p1 = 1:LRcv1
       for p2 = 1:LRcv2
           for p3 = 1:LRcv3
               for alpha = 1:1:N_alpha
               % select the low-rank number from U1, U2, U3
                 U1 = tUs{1,1}(1:p1,:);
                 U2 = tUs{2,1}(1:p2,:);
                 U3 = tUs{3,1}(1:p3,:);
                 U32kror = kron(U3,U2);
                 U321kror = kron(U32kror,U1);
               % initialize S matrix 
                 S_t = double(tenmat(X_4Dm,4))* U321kror' * pinv(U321kror* U321kror');
                 N = size(X_4Dm,4);
                 S_t = reshape(S_t',[p1,p2,p3,N]);
               % Optimization and Iteration, get U1, U2, U3
                  %Set initial iteration criterion
                    ita = 100;
                    opt_0 = 1000000000;
                    iteration = 0;
       
                    X_4Ds = X_4Dm;
                    S = S_t;
                    Ym = Ym_tcv;
                    S_U4 = [ones(N,1) double(tenmat(S,4))];
                    Beta = pinv(S_U4' * S_U4) * S_U4' * log(Ym);
                    Beta1 = Beta(2:end,:);
                    Beta0 = Beta(1,:);


                  while  (ita > 10^(-4))  && (iteration < 500)
                        % Slove U1 by regression equation
                        S_U1 = double(tenmat(ttm(tensor(S),{U2' U3'},[2 3] ),1));
                        X_U1 = double(tenmat(X_4Ds,1));
                        indexU1 = find(sum(X_U1)==0);
                        S_U1(:,indexU1) = 0;
                        U1 = (X_U1 * S_U1' * pinv(S_U1 * S_U1'))';
                        % Slove U2 by regression equation
                        S_U2 = double(tenmat(ttm(tensor(S),{U1' U3'},[1 3] ),2));
                        X_U2 = double(tenmat(X_4Ds,2));
                        indexU2 = find(sum(X_U2)==0);
                        S_U2(:,indexU2) = 0;
                        U2 = (X_U2 * S_U2' * pinv(S_U2 * S_U2'))';
                        % Slove U3 by regression equation
                        [U3]=OptU3Miss(alpha,N,p1,p2,p3,I1,I2,I3,U1,U2,Beta1,Beta0,Ym, S,X_4Ds);
                        % Slove S by cvx
                        [S,S_un4]=OptSMiss(alpha,N,p1,p2,p3,I1,I2,I3,U1,U2,U3,Beta1,Beta0,Ym,X_4Ds);
                        % Slove Beta0 and Beta1
                        S_U4 = [ones(N,1) double(tenmat(S,4))];
                        Beta = pinv(S_U4' * S_U4) * S_U4' * log(Ym);
                        Beta1 = Beta(2:end,:);
                        Beta0 = Beta(1,:);
                        % Objective value
                        U32kro = kron(U3,U2);
                        U321kro = kron(U32kro,U1);
                        Dif_XS = (double(tenmat(X_4Ds,4)) - S_un4* U321kro).*logical(double(tenmat(X_4Ds,4)));
                        LSE_YS = log(Ym) - Beta0*ones(N,1) - S_un4 * Beta1;
                        %opt = (alpha)*(power(norm( Dif_XS,'fro'),2)) + (1-alpha)*(power(norm(LSE_YS,'fro'),2));
                        opt = (alpha/10-0.1)*(power(norm( Dif_XS,'fro'),2)) + (1.1-alpha/10)*(power(norm(LSE_YS,'fro'),2));
                        ita = opt_0 - opt;
                        opt_0 = opt;
                        iteration = iteration + 1;
                  end
                  
                  
               
                   
                   Er_fold = [];
                      for fd = 1:n_fold
          
                          [Er_fd]=CVproposed_sub_method2(Nt,n_fold,fd,p1,p2,p3,I1,I2,I3,U1,U2,U3,X_4Dcv,Ym_tcv,alpha);
                          Er_fold = [Er_fold  Er_fd];
                          
                      end
                   
                   Er_median{p1,p2,p3,alpha} = median(Er_fold,'all');
                   %Er_var{p1,p2,p3,alpha} = prctile(Er_fold,75,'all') + prctile(Er_fold,25,'all');
                   %Er{p1,p2,p3,alpha} = 1/3 * Er_median{p1,p2,p3,alpha} + 2/3 * Er_var{p1,p2,p3,alpha};
                   Er{p1,p2,p3,alpha} = Er_median{p1,p2,p3,alpha};
               end
           end
       end
   end
% Select parameter combination (p1, p2, p3, alpha) from 'Er' or 'I' index   
Er_double = cell2mat(Er);
[M,I] = min(Er_double,[],'all','linear');
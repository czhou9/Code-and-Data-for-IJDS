
%% input parater for Numerical Study
% Input the low rank value and trade-off parameter from the result of
% cross-validation. Specifically, for the sample data we generated in the
% attached file, we give the selected combination for convenience - those
% are - 
% Complete, (p1, p2, p3, alpha) = (1,2,3,0.8) or (1,3,2,0.4); 
% 10% missing - (p1, p2, p3, alpha) = (1,3,2,0.2);
% 50% missing - (p1, p2, p3, alpha) = (3,3,2,0) or (4,2,4,0.1);
% 90% missing - (p1, p2, p3, alpha) = (1,2,1,0.2)
p1 = 1;
p2 = 2;
p3 = 3;
% For the convenience of storing data and runing the cluster, we change
% alpha to integer in the code. Alpha in code = 10*(real alpha + 0.1)
alpha = 9;

% input the number of training sample and test sample, size of a
% high-dimensional tensor
N = 400;
N_test = 100;
I1 = size(X_4D,1);
I2 = size(X_4D,2);
I3 = size(X_4D,3);

% After randomizing the sequence, we select the 1-400 samples as training
% set, the 401-500 as testing set
X_4D_test = X_4D(:,:,:,(N+1):(1.25*N));
X_4D = X_4D(:,:,:,1:N);
Ym_test = Ym_t((N+1): (1.25*N));
Ym_t = Ym_t(1:N);

%% Initialize U1, U2, U3 from MPCA 
% Before initilization with MPCA, fill the missing value with -1, we can
% also use tensor completion to fill the empty values but it is a little
% time-consuming
X_4Dm = X_4D;
X_4Dm(X_4Dm==0)=-1;
X_MPCA = X_4Dm;
Ym_MPCA = Ym_t;

TX = X_MPCA;
gndTX = Ym_MPCA;
testQ = 100;
maxK = 1;
[tUs, odrIdx, TXmean, Wgt]  = MPCA(TX,gndTX,testQ,maxK);

% select the low-rank number from U1, U2, U3
U1 = tUs{1,1}(1:p1,:);
U2 = tUs{2,1}(1:p2,:);
U3 = tUs{3,1}(1:p3,:);
U32kror = kron(U3,U2);
U321kror = kron(U32kror,U1);
% initialize S matrix 
S_t = double(tenmat(X_4Dm(:,:,:,1:N),4))* U321kror' * pinv(U321kror* U321kror');
S_t = reshape(S_t',[p1,p2,p3,N]);

%% iteration process
     %Set initial iteration criterion
       ita = 100;
       opt_0 = 1000000000;
       iteration = 0;
       
       X_4Ds = X_4D(:,:,:,1:N);
       S = S_t(:,:,:,1:N);
       Ym = Ym_t(1:N); 
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
     opt = (alpha)*(power(norm( Dif_XS,'fro'),2)) + (1-alpha)*(power(norm(LSE_YS,'fro'),2));
     %opt = (alpha/10-0.1)*(power(norm( Dif_XS,'fro'),2)) + (1.1-alpha/10)*(power(norm(LSE_YS,'fro'),2));
     ita = opt_0 - opt;
     opt_0 = opt;
     iteration = iteration + 1;
     end
    % label the iterated U1, U2, U3 as U1_rv, U2_rv, U3_rv
    U1_rv = U1;
    U2_rv = U2;
    U3_rv = U3;

%% Generate Beta0 and Beta1 based on the iterated U1, U2, U3

Ym_train = Ym_t(1:N); 
    % Generate kronecker matrix 
    U32kro_rv = kron(U3_rv,U2_rv);
    U321kro_rv = kron(U32kro_rv,U1_rv);
    % Generate S matrix 
    X_S4_i = double(tenmat(X_4D(:,:,:,1:N),4));
    S_un4_i = [];
    for j = 1:N
        U32kro_rv = kron(U3_rv,U2_rv);
        U321kro_rv = kron(U32kro_rv,U1_rv);
        indexS_i = find(X_S4_i(j,:)==0);
        U321kro_rv(:,indexS_i) = 0;
        S_un4_i = [S_un4_i; X_S4_i(j,:)* U321kro_rv' * pinv(U321kro_rv* U321kro_rv')];
    end
    S_i = reshape(S_un4_i', [p1, p2, p3, N]);
    S_i_U4 = [ones(N,1) double(tenmat(S_i,4))];
    Beta = pinv(S_i_U4' * S_i_U4) * S_i_U4' * log(Ym_train);
    Beta1_rv = Beta(2:end,:);
    Beta0_rv = Beta(1,:);
    



%% Generate estimated response TTF and boxplot
% Select last 20% data as test set
%Ym_test = Ym_t((N+1): (1.25*N));
%Ym_test = Ym_t_test2; % edit #


% Generate kronecker matrix for test data
U32kro_test = kron(U3_rv,U2_rv);
U321kro_test = kron(U32kro_test,U1_rv);
% Generate S matrix of the test dataset 
%X_S4_test = double(tenmat(X_4D(:,:,:,(N+1):(1.25*N)),4)); % edit #
X_S4_test = double(tenmat(X_4D_test,4)); % edit #
S_un4_test = [];
for i = 1:N_test
    U32kro_test = kron(U3_rv,U2_rv);
    U321kro_test = kron(U32kro_test,U1_rv);
    indexS_test = find(X_S4_test(i,:)==0);
    U321kro_test(:,indexS_test) = 0;
    S_un4_test = [S_un4_test; X_S4_test(i,:)* U321kro_test' * pinv(U321kro_test* U321kro_test')];
    %S_un4_test = [S_un4_test; X_S4_test(i,:)* U321kro_test'];
end
S_test = reshape(S_un4_test', [p1, p2, p3, N_test]);

% Generate estimated Ym and make prediction error
Ym_est = exp(Beta0_rv * ones(N_test,1)  + double(tenmat(S_test,4)) * Beta1_rv);
PredEr = abs(Ym_test-Ym_est)./ Ym_test;

boxplot(PredEr)



















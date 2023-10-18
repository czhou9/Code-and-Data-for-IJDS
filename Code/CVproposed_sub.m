function [Er_fd]=CVproposed_sub(Nt,n_fold,fd,p1,p2,p3,I1,I2,I3,X_4Dcv,Ym_tcv,alpha)


%% select the training sample
    if (fd == 1)
        X_4D = zeros(I1,I2,I3,Nt-Nt/n_fold);
        for i = (fd*(Nt/n_fold)+1) : (n_fold*(Nt/n_fold))
            X_4D(:,:,:,i-Nt/n_fold) = X_4Dcv(:,:,:,i);
        end
        Ym = zeros(Nt-Nt/n_fold,1);
        for i = (fd*(Nt/n_fold)+1) : (n_fold*(Nt/n_fold))
            Ym(i-Nt/n_fold,1) = Ym_tcv(i,1);
        end
        
    else if (fd > 1) && (fd < n_fold)
        X_4D = zeros(I1,I2,I3,Nt-Nt/n_fold);
        for i = 1:((fd-1)*(Nt/n_fold))
            X_4D(:,:,:,i) = X_4Dcv(:,:,:,i);
        end
        for i = ((fd)*(Nt/n_fold)+1) : (n_fold *(Nt/n_fold))
            X_4D(:,:,:,i-Nt/n_fold) = X_4Dcv(:,:,:,i);
        end
        Ym = zeros(Nt-Nt/n_fold,1);
        for i = 1:((fd-1)*(Nt/n_fold))
            Ym(i,1) = Ym_tcv(i,1);
        end
        for i = ((fd)*(Nt/n_fold)+1) : (n_fold *(Nt/n_fold))
            Ym(i-Nt/n_fold,1) = Ym_tcv(i,1);
        end
        
    else
        X_4D = zeros(I1,I2,I3,Nt-Nt/n_fold);
        for i = 1 : ((n_fold-1) * (Nt/n_fold)) 
            X_4D(:,:,:,i) = X_4Dcv(:,:,:,i);
        end
        Ym = zeros(Nt-Nt/n_fold,1);
        for i = 1 : ((n_fold-1) * (Nt/n_fold))
            Ym(i,1) = Ym_tcv(i,1);
        end
        end
    end
    
%% Initilize U1,U2,U3 and S
X_4Dm = X_4D;
X_4Dm(X_4Dm==0)=-1;
X_MPCA = X_4Dm;
Ym_MPCA = Ym;

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
S_t = double(tenmat(X_4Dm,4))* U321kror' * pinv(U321kror* U321kror');
N = size(X_4Dm,4);
S_t = reshape(S_t',[p1,p2,p3,N]);

%% Optimization and Iteration, get U1, U2, U3


  
       
     %Set initial iteration criterion
       ita = 100;
       opt_0 = 1000000000;
       iteration = 0;
       
       X_4Ds = X_4D;
       S = S_t;
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
     % label the iterated U1, U2, U3 as U1_rv, U2_rv, U3_rv
     U1_rv = U1;
     U2_rv = U2;
     U3_rv = U3;
       
    
%% Generate Beta0 and Beta1 based on the iterated U1, U2, U3

Ym_train = Ym; 
    % Generate kronecker matrix 
    U32kro_rv = kron(U3_rv,U2_rv);
    U321kro_rv = kron(U32kro_rv,U1_rv);
    % Generate S matrix 
    X_S4_i = double(tenmat(X_4D,4));
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

%% Estimate the response
Ym_test = zeros(Nt/n_fold,1);
for i = (((fd - 1)*(Nt/n_fold))+1) : ((fd) *(Nt/n_fold))
    Ym_test(i-(fd-1)*Nt/n_fold,1) = Ym_tcv(i,1);
end

% Generate S matrix of the test dataset 
X_4Dtest = zeros(I1,I2,I3,Nt/n_fold);
for i = (((fd - 1)*(Nt/n_fold))+1) : ((fd) *(Nt/n_fold))
    X_4Dtest(:,:,:,i-(fd-1)*Nt/n_fold) = X_4Dcv(:,:,:,i);
end
% Generate kronecker matrix for test data
U32kro_test = kron(U3_rv,U2_rv);
U321kro_test = kron(U32kro_test,U1_rv);
% Generate S matrix of the test dataset 
%X_S4_test = double(tenmat(X_4D(:,:,:,(N+1):(1.25*N)),4)); % edit #
X_S4_test = double(tenmat(X_4Dtest,4)); % edit #
S_un4_test = [];
for i = 1:(Nt/n_fold)
    U32kro_test = kron(U3_rv,U2_rv);
    U321kro_test = kron(U32kro_test,U1_rv);
    indexS_test = find(X_S4_test(i,:)==0);
    U321kro_test(:,indexS_test) = 0;
    S_un4_test = [S_un4_test; X_S4_test(i,:)* U321kro_test' * pinv(U321kro_test* U321kro_test')];
    %S_un4_test = [S_un4_test; X_S4_test(i,:)* U321kro_test'];
end
S_test = reshape(S_un4_test', [p1, p2, p3, Nt/n_fold]);
Ym_est = exp(Beta0_rv * ones(Nt/n_fold,1)  + double(tenmat(S_test,4)) * Beta1_rv);
Er_fd = abs(Ym_test-Ym_est)./ Ym_test;
end

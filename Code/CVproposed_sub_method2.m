function [Er_fd]=CVproposed_sub_method2(Nt,n_fold,fd,p1,p2,p3,I1,I2,I3,U1,U2,U3,X_4Dcv,Ym_tcv,alpha)


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
    


%% label the iterated U1, U2, U3 as U1_rv, U2_rv, U3_rv
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
    for j = 1:(Nt-Nt/n_fold)
        U32kro_rv = kron(U3_rv,U2_rv);
        U321kro_rv = kron(U32kro_rv,U1_rv);
        indexS_i = find(X_S4_i(j,:)==0);
        U321kro_rv(:,indexS_i) = 0;
        S_un4_i = [S_un4_i; X_S4_i(j,:)* U321kro_rv' * pinv(U321kro_rv* U321kro_rv')];
    end
    S_i = reshape(S_un4_i', [p1, p2, p3, (Nt-Nt/n_fold)]);
    S_i_U4 = [ones((Nt-Nt/n_fold),1) double(tenmat(S_i,4))];
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
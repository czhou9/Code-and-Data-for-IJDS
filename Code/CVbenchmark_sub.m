function [Er_fd]=CVbenchmark_sub(Nt,n_fold,fd,p1,p2,p3,I1,I2,I3,X_4Dcv,Ym_tcv)


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
    
%% Determine U1, U2, U3
% Apply MPCA to generate U1, U2, U3 for 100% variance
TX = X_4D;
gndTX = Ym;
%gndTX = -1; % if no label for response
testQ = 100; 
maxK = 1;
[tUs, odrIdx, TXmean, Wgt]  = MPCA(TX,gndTX,testQ,maxK);

U1_MPCA = tUs{1,1}(1:p1,:);
U2_MPCA = tUs{2,1}(1:p2,:);
U3_MPCA = tUs{3,1}(1:p3,:);

%% Update S, Beta0, Beta1
X_4Ds = X_4D;
%Ym = Ym_t; 
S = double(ttm(tensor(X_4Ds), {U1_MPCA, U2_MPCA, U3_MPCA}, [1 2 3]));
S_U4 = [ones(Nt - Nt/n_fold,1) double(tenmat(S,4))];
Beta = pinv(S_U4' * S_U4) * S_U4' * log(Ym);
Beta1 = Beta(2:end,:);
Beta0 = Beta(1,:);


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

S_test = double(ttm(tensor(X_4Dtest), {U1_MPCA, U2_MPCA, U3_MPCA}, [1 2 3]));
Ym_est = exp(Beta0 * ones(Nt/n_fold,1)  + double(tenmat(S_test,4)) * Beta1);
Er_fd = abs(Ym_test-Ym_est)./ Ym_test;
end

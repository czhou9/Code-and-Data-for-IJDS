%% Set initial parameter 
N = 400;
N_test = 100;
X_MPCA = X_4D(:,:,:,1:N);
Ym_MPCA = Ym_t(1:N);

TX = X_MPCA;
gndTX = Ym_MPCA;
testQ =97;
maxK = 1;
[tUs, odrIdx, TXmean, Wgt]  = MPCA(TX,gndTX,testQ,maxK);

U1_MPCA = tUs{1,1};
U2_MPCA = tUs{2,1};
U3_MPCA = tUs{3,1};


%% Derive Beta0 and Beta1
S_MPCA = double(ttm(tensor(X_MPCA), {U1_MPCA, U2_MPCA, U3_MPCA}, [1 2 3]));
S_U4MPCA = [ones(N,1) double(tenmat(S_MPCA,4))];
Beta_MPCA = pinv(S_U4MPCA' * S_U4MPCA) * S_U4MPCA' * log(Ym_MPCA);
Beta1_MPCA = Beta_MPCA(2:end,:);
Beta0_MPCA = Beta_MPCA(1,:);

%% Estimate TTF 
% Select last 20% data as test set
Ym_testMPCA = Ym_t((N+1): (1.25*N));

% Generate S matrix of the test dataset 
S_testMPCA = double(ttm(tensor(X_4D(:,:,:,((N+1):(1.25*N)))), {U1_MPCA, U2_MPCA, U3_MPCA}, [1 2 3]));

% Generate estimated Ym and make prediction error
Ym_estMPCA = exp(Beta0_MPCA * ones(N_test,1)  + double(tenmat(S_testMPCA,4)) * Beta1_MPCA); %edit
PredEr_MPCA = abs(Ym_testMPCA-Ym_estMPCA)./ Ym_testMPCA;
boxplot(PredEr_MPCA)

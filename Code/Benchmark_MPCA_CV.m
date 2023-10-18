%% Initial Parameter
% After randomizing the sequence of data set, We set the first 400 as training set
% and 100 as test set.
N = 400;
N_test = 100;

% Input the low-rank of U1, U2 and U3 from the cross-validation. For the
% sample data in the attached file, we give the selected parater - those
% are
% complete - (p1, p2, p3) = (3,3,3)
% 10% missing - (p1, p2, p3) = (2,3,1)
% 50% missing - (p1, p2, p3) = (1,3,3)
% 90% missing - (p1, p2, p3) = (3,3,3)

p1 = 3;
p2 = 3;
p3 = 3;

%% Initialize U1, U2, U3 from MPCA
% Generate full-rank projection by seting the fraction of variation to 100
X_MPCA = X_4D(:,:,:,1:N);
Ym_MPCA = Ym_t(1:N);
TX = X_MPCA;
gndTX = Ym_MPCA;
testQ = 100;
maxK = 1;
[tUs, odrIdx, TXmean, Wgt]  = MPCA(TX,gndTX,testQ,maxK);

% Select the low-rank based on p1, p2 and p3
U1_MPCA = tUs{1,1}(1:p1,:);
U2_MPCA = tUs{2,1}(1:p2,:);
U3_MPCA = tUs{3,1}(1:p3,:);

%% Derive Beta0 and Beta1
S_MPCA = double(ttm(tensor(X_MPCA), {U1_MPCA, U2_MPCA, U3_MPCA}, [1 2 3]));
S_U4MPCA = [ones(N,1) double(tenmat(S_MPCA,4))];
Beta_MPCA = pinv(S_U4MPCA' * S_U4MPCA) * S_U4MPCA' * log(Ym_MPCA);
Beta1_MPCA = Beta_MPCA(2:end,:);
Beta0_MPCA = Beta_MPCA(1,:);

%% Estimate TTF 
% Select last 20% (here 401-500) data as test set
Ym_testMPCA = Ym_t((N+1): (1.25*N));

% Generate S matrix of the test dataset 
S_testMPCA = double(ttm(tensor(X_4D(:,:,:,((N+1):(1.25*N)))), {U1_MPCA, U2_MPCA, U3_MPCA}, [1 2 3]));

% Generate estimated Ym and make prediction error
Ym_estMPCA = exp(Beta0_MPCA * ones(N_test,1)  + double(tenmat(S_testMPCA,4)) * Beta1_MPCA);
PredEr_MPCA_CV = abs(Ym_testMPCA-Ym_estMPCA)./ Ym_testMPCA;
boxplot(PredEr_MPCA_CV)






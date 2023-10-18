%% Tensor Completion for training data
N = 400; 
I1 = size(X_4D,1);
I2 = size(X_4D,2);
I3 = size(X_4D,3);

data0 = X_4D(:,:,:,1:N);
N_TC = ndims(data0);
known = find(data0~=0);
data = data0(known);
Nway = size(data0);
coreNway = [2 2 2 10];
opts = [];
opts.maxit = 100; 
opts.tol = -1e-5; % run to maxit by using negative tolerance
opts.Mtr = data0; % pass the true tensor to calculate the fitting
opts.alpha_adj = 0;
opts.rank_adj = 0*ones(1,N_TC);
opts.rank_min = 5*ones(1,N_TC);
opts.rank_max = 20*ones(1,N_TC);

EstCoreNway = round(1.25*coreNway);
coNway = zeros(1,N_TC);
for n = 1:N_TC
    coNway(n) = prod(Nway)/Nway(n);
end

% use random generated starting point
for i = 1:N_TC
    X0{i} = randn(Nway(i),EstCoreNway(i));
    Y0{i} = randn(EstCoreNway(i),coNway(i));
end

opts.X0 = X0; opts.Y0 = Y0;
[X,Y,Out] = TMac(data,known,Nway,coreNway,opts);


TC_1 = reshape(X{1}*Y{1}, [I1 I2 I3 N]);
TC_2 = permute(reshape(X{2}*Y{2}, [I2 I1 I3 N]), [2 1 3 4]);
TC_3 = permute(reshape(X{3}*Y{3}, [I3 I1 I2 N]), [2 3 1 4]);
TC_4 = permute(reshape(X{4}*Y{4}, [N I1 I2 I3]), [2 3 4 1]);

TC = (1/N_TC)* (TC_1 + TC_2 + TC_3 + TC_4);

%% Tensor Completion for test data
X_4D_train = X_4D(:,:,:,1:N);
X_4D_test = X_4D(:,:,:,(N+1):(1.25*N));
I1 = size(X_4D,1);
I2 = size(X_4D,2);
I3 = size(X_4D,3);

N_TC_train = size(X_4D_train,4);
N_TC_test = size(X_4D_test,4);
X_total = zeros(I1,I2,I3,N_TC_train + 1);
X_total(:,:,:,1:N_TC_train) = X_4D_train;
TC_total = zeros(I1,I2,I3,N_TC_train + 1);
TC_test = zeros(I1,I2,I3,N_TC_test);

for k = 1:N_TC_test
    X_total(:,:,:,N_TC_train + 1) = X_4D_test(:,:,:,k);
    data0 = X_total;
    N_TC = ndims(data0);
    known = find(data0~=0);
    data = data0(known);
    Nway = size(data0);
    coreNway = [2 2 2 10];
    opts = [];
    opts.maxit = 100; 
    opts.tol = -1e-5; % run to maxit by using negative tolerance
    opts.Mtr = data0; % pass the true tensor to calculate the fitting
    opts.alpha_adj = 0;
    opts.rank_adj = 0*ones(1,N_TC);
    opts.rank_min = 5*ones(1,N_TC);
    opts.rank_max = 20*ones(1,N_TC);
    
    EstCoreNway = round(1.25*coreNway);
    coNway = zeros(1,N_TC);
    for n = 1:N_TC
        coNway(n) = prod(Nway)/Nway(n);
    end

    % use random generated starting point
    for i = 1:N_TC
        X0{i} = randn(Nway(i),EstCoreNway(i));
        Y0{i} = randn(EstCoreNway(i),coNway(i));
    end

    opts.X0 = X0; opts.Y0 = Y0;
    [X,Y,Out] = TMac(data,known,Nway,coreNway,opts);

    TC_1 = reshape(X{1}*Y{1}, [I1 I2 I3 size(data0,4)]);
    TC_2 = permute(reshape(X{2}*Y{2}, [I2 I1 I3 size(data0,4)]), [2 1 3 4]);
    TC_3 = permute(reshape(X{3}*Y{3}, [I3 I1 I2 size(data0,4)]), [2 3 1 4]);
    TC_4 = permute(reshape(X{4}*Y{4}, [size(data0,4) I1 I2 I3]), [2 3 4 1]);

    TC_total = (1/N_TC)* (TC_1 + TC_2 + TC_3 + TC_4);
    
    TC_test(:,:,:,k) = TC_total(:,:,:,N_TC_train + 1);
end

%% combine train and test data
X_4D = zeros(21,21,10,500);
X_4D(:,:,:,1:400) = TC;
X_4D(:,:,:,401:500) = TC_test;




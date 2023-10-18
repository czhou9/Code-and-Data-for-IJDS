close all
clear all

% 'X_4D' is a 21*21*100*500 tensor where 500 the sample size of image streams, (21, 21, 10) is the size of 
% a image stream. 'Ym_t' is a 500*1 vector which denotes the TTF corresponding to the 500 image streams.
% The code of randomizing the sequence and adding missing entries is hided at the last.
%% Input parameter
L = 0.2;%0.05
H = 0.2;%0.05
dx = 0.01;%0.0025
dy = 0.01;%0.0025
tmax = 200000;%50
dt = 0.25;%0.01
epsilon = 0.0001;%0.0001
N=400;
domainalp=1e-4:-1e-7:0;%to avoid randperm
domainalp_total=domainalp(1:(1.25*N));
nx = uint32(L/dx + 1);%21
ny = uint32(H/dy + 1);%21
ImageNumber=10;
rng('default')
AVET=[];
I1 = nx;
I2 = ny;
I3 = ImageNumber;

for iN=1:(1.25*N)
    r_x = domainalp_total(iN)*dt/dx^2;
    r_y = domainalp_total(iN)*dt/dy^2;
    fo = r_x + r_y;
    %if fo > 1/2
    if fo > 1/2
        %         warndlg({'Numerical stability requires Fo <= 1/2';
        %             sprintf('Current Fo = %g',fo)},'Numerically Unstable');
%         ER=1;
%         save(['./Simulated Data/SimulateData_' num2str(procnum) '.mat'],'ER')
        return;
    end
    % create the x, y meshgrid based on dx, dy
    nx = uint32(L/dx + 1);%21
    ny = uint32(H/dy + 1);%21
    [X,Y] = meshgrid(linspace(0,L,nx),linspace(0,H,ny));
    % take the center point of the domain
    ic = uint32((nx-1)/2+1);
    jc = uint32((ny-1)/2+1);
    % set initial and boundary conditions
    T = 0*ones(ny,nx);%initial T=10
    T(:,1) = 30;%left T=50
    T(:,end) = 30;%right T=50
    T(1,:) = 30;%bottom T=50
    T(end,:) = 30;%top T=50
    Tmin = min(min(T));
    Tmax = max(max(T));
    % iteration, march in time
    n = 0;
    nmax = uint32(tmax/dt);
    CenterT=[];
    AveT=[];SaveIndex=0;iSave=0;
    
    [rr, cc]=size(T);
    nSaved = 0;
    nIndex = 15:15:150;
    
    %% Add iid Noise and generate T matrix
    InitialNoise = zeros(rr,cc);
    %nSaved=0;
    while n < nmax
    n = n + 1;
        
        T_n = T;
        for j = 2:ny-1
            for i = 2:nx-1
                T(j,i) = T_n(j,i) + r_x*(T_n(j,i+1)-2*T_n(j,i)+T_n(j,i-1))...
                    + r_y*(T_n(j+1,i)-2*T_n(j,i)+T_n(j-1,i));
            end
        end
        %bSaveImage=0;
        NoiseT_exp = T + 0.1*randn(rr,cc);
        SimulateData_exp{n,iN} = NoiseT_exp;
        
         if sum(n==nIndex)==1 
         NoiseT=T+0.1*randn(rr,cc);
         nSaved = nSaved + 1;
         SimulateData{nSaved,iN}=NoiseT;
         end
            if nSaved==ImageNumber
                break;
            end
     end
end

for i = 1:(1.25*N)
    SimulateData_Noise_exp{i,1} = cat(3,SimulateData_exp{:,i});
end

for i = 1:(1.25*N)
    SimulateData_Noise{i,1} = cat(3,SimulateData{:,i});
end


%% Generate response variable based on heat transfer
X_4D = cat(4, SimulateData_Noise{1:1.25*N});
newx=zeros(1.25*N,150);
newy=zeros(1.25*N,1);
for i=1:(1.25*N)
    newx(i,:)=mean(mean(SimulateData_Noise_exp{i,1}));
    xx=newx(i,:);
    %the threshold is set as 22.95 (around 23), we want to gurantee there
    %is at least one image stream which is complete from time 1 to time 150
    index=find(xx>22.95,1); 
    SimulateData_Noise_exp{i,1}(:,:,index:150)=0;
    newy(i)=index;
end
Ym_t = newy;

%% Generate random sequence of 4D tensor and response
% randseq = randperm(500);
% X_4D = X_4D(:,:,:,randseq);
% Ym_t = Ym_t(randseq);

%% Generate X_4D matrix with missing values (missing entry is labelled as 0)
% % Miss percentage
% Mp = 0.1;
% 
% % Generate 4D tensor with missing values
% for i = 1:size(X_4D,4)
% 
%     index_last = find(sum(sum(X_4D(:,:,:,i))) ~= 0, 1,'last');
%     
%     % For the sample data we generated in the attached file, we use 'round'
%     % in 50% missing; 'ceil' in 10% missing; 'floor' in 90% missing
%     ImageMiss = randsample(index_last,round(Mp*index_last));
%     %ImageMiss = randsample(index_last,ceil(Mp*index_last));
%     %ImageMiss = randsample(index_last,floor(Mp*index_last));
%     
%     X_4D(:,:,ImageMiss,i) = 0;
% 
% end







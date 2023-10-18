%% Determine initial parameters
% set the total number of training samples
Nt = 400; 

% We use the first 400 tensor samples as the training samples
X_4Dcv = X_4D(:,:,:,1:Nt); 
Ym_tcv = Ym_t(1:Nt); 

% set the number of folds in CV
n_fold = 5; 

% set the max low-rank number for 1st, 2nd, 3rd mode 
%(We can also set the min number to 3 for decreasing computation load)
LRcv1 = 4;  
LRcv2 = 4;
LRcv3 = 4;
I1 = size(X_4D,1);
I2 = size(X_4D,2);
I3 = size(X_4D,3);


%% cross-validation
% Here we use the median of the Error matrix as the criteria to select the
% best (p1, p2, p3) combination; We can also use the mean, variance, IQR, max/min
% as the criteria for different goals. And the results of different
% critera are similar.

Er = cell(LRcv1,LRcv2,LRcv3);
Er_median = cell(LRcv1,LRcv2,LRcv3);
%Er_mean = cell(LRcv1, LRcv2, LRcv3);
%Er_var = cell(LRcv1, LRcv2, LRcv3);
   
   for p1 = 1:LRcv1
       for p2 = 1:LRcv2
           for p3 = 1:LRcv3
               Er_fold = [];
                      for fd = 1:n_fold
          
                          [Er_fd]=CVbenchmark_sub(Nt,n_fold,fd,p1,p2,p3,I1,I2,I3,X_4Dcv,Ym_tcv);
                           Er_fold = [Er_fold  Er_fd];
                      end

                   Er_median{p1,p2,p3} = median(Er_fold,'all');
                   %Er_fold = rmoutliers(Er_fold,'mean');
                   %Er_var{p1,p2,p3} = prctile(Er_fold,75,'all') + prctile(Er_fold,25,'all');
                   %Er_boundary{p1,p2,p3} = min(Er_fold,[],'all') + max(Er_fold,[],'all');
                   %Er{p1,p2,p3} = 1/5*(Er_median{p1,p2,p3} + Er_var{p1,p2,p3} + Er_boundary{p1,p2,p3});
                   Er{p1,p2,p3} = Er_median{p1,p2,p3};
           end
       end
   end
   
%% Check the index and value of the Error matrix 
% selecte the parameter combination (p1, p2, p3) from the 'Er' or 'I' index
Er_double = cell2mat(Er);
[M,I] = min(Er_double,[],'all','linear'); 

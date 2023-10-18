function [U3]=OptU3Miss(alpha,N,p1,p2,p3,I1,I2,I3,U1,U2,Beta1,Beta0,Ym, S,X_4Ds)

% create the non-zero index for each image
X_U3miss = double(tenmat(X_4Ds,4));
indexU3 = cell(I3,1);
for i = 1: I3
    indexU3{i,1} = find(sum(X_U3miss(:,((i-1)*I1*I2+1):(i*I1*I2)),2)~=0);
end

% create U3
U3 = [];
for i = 1: I3
    XS = zeros(1,p3);
    SS = zeros(p3,p3);
    for j = indexU3{i,1}
        SiSi = double(tenmat( ttm(tensor(S(:,:,:,j)),{U1' U2'},[1 2] ),3))*double(tenmat( ttm(tensor(S(:,:,:,j)),{U1' U2'},[1 2] ),3))';
        SS = SS + SiSi;
        Xi = double(tenmat(tensor(X_4Ds(:,:,:,j)),3));
        XiSi = Xi(1,:)*double(tenmat( ttm(tensor(S(:,:,:,j)),{U1' U2'},[1 2] ),3))';
        XS = XS + XiSi;
    end
    U3 = [U3 (XS * pinv(SS))'];
end

end
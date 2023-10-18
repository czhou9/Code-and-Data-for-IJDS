function [S,S_un4]=OptSMiss(alpha,N,p1,p2,p3,I1,I2,I3,U1,U2,U3,Beta1,Beta0,Ym,X_4Ds)

X_S4 = double(tenmat(X_4Ds,4));
U32kro = kron(U3,U2);
U321kro = kron(U32kro,U1);

S_un4 = [];
Ylogmiss = log(Ym);
for i = 1:N
    U32kro = kron(U3,U2);
    U321kro = kron(U32kro,U1);
    indexS = find(X_S4(i,:)==0);
    U321kro(:,indexS) = 0;
    %S_un4 = [S_un4; ((alpha)*X_S4(i,:)*U321kro' + (1-alpha)*(Ylogmiss(i,:)-Beta0)*Beta1') * pinv((alpha)*U321kro*U321kro' + (1-alpha)*Beta1*Beta1')];
    S_un4 = [S_un4; ((alpha/10-0.1)*X_S4(i,:)*U321kro' + (1.1-alpha/10)*(Ylogmiss(i,:)-Beta0)*Beta1') * pinv((alpha/10-0.1)*U321kro*U321kro' + (1.1-alpha/10)*Beta1*Beta1')];

end

S = reshape(S_un4', [p1, p2, p3, N]);
end
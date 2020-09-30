function [Wg,Wl] = PRL(X,Y,A,alpha,beta,gamma,maxItr)

[n,d] = size(X);
[~,k] = size(Y);

seed = 1;
rand('state',seed);
randn('state',seed);

Wg = randn(d,k);
Wl = randn(n*d,k);

Z = sparse(n,n*d);
for i = 1:n
    Z(i,1+d*(i-1):i*d) = X(i,:);
end


for ite = 1:maxItr
    
    % update D
    tmp = sqrt(sum(Wg.*Wg,2)+0.00001);
    tmpd = 0.5./tmp;
    D = diag(tmpd);
    clear tmp tmpd
    
    % update Wg
    Wg = -(X'*X + gamma*D)\X'*(Z*Wl-Y);
    
    % update C
    for i = 1:n
        for j = 1:n
            dis(i,j) = norm((Wl(1+d*(i-1):i*d,:)-Wl(1+d*(j-1):j*d,:)),'fro');
        end
    end
    tmp4 = dis.*A;
    tmp4 = sum(tmp4(:));
    
    tmp = (0.5./(dis+0.00001)).*A;
    td1 = diag(sum(tmp,1));
    td2 = diag(sum(tmp,2));
    C = td1 + td2 - 2*tmp;
    C = (C + C')/2 + eye(n)*0.00001;
    clear tmp
    
    CI = kron(sparse(C),speye(d));
    
    % update F
    for i = 1:n
        tp = Wl(1+d*(i-1):i*d,:);
        tmp(i) = sum(sqrt(sum(tp.*tp,2)));
    end
    tmp3 = sum(tmp.^2);
    index = 1:n*d;
    tmp = repmat(tmp,d,1);
    Fi = sqrt(sum(Wl.*Wl,2)+0.00001);
    F = sparse(index,index,tmp(:)./Fi);
    
    G = alpha*CI + beta*F; 
    
    % update W
    tmp1 = G\Z';
    tmp2 = (eye(n)+Z*tmp1)\(X*Wg-Y);
    Wl = tmp1*tmp2; 
    
    % obtain objective function value
    fval(ite) = (norm((Z*Wl+X*Wg-Y),'fro'))^2 + alpha*tmp4 + beta*tmp3 + gamma*trace(Wg'*D*Wg);
    
    
end

end
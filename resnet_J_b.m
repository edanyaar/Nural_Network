function [ J ] = resnet_J_b( W1,W2,b,X,v)
    [z,d] = size(W1);
    [d,z] = size(W2); 
    [d,m] = size(X); 
    [z,~] = size(b); 

    J = reshape(kron(eye(m),W2)*diag(reshape(d_sigma(W1*X+b),[],1))*repmat(v,m,1),d,m); 
end


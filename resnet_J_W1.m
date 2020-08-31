function [ J ] = resnet_J_W1( W1,W2,b,X,v)
    [z,d] = size(W1);
    [d,z] = size(W2); 
    [d,m] = size(X); 
    [z,~] = size(b); 
    J = reshape(kron(eye(m),W2)*(reshape(d_sigma(W1*X+b),[],1).*(kron(X',eye(z))*reshape(v,[],1))),d,m);

end


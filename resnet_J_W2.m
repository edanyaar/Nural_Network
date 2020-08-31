function [ J ] = resnet_J_W2( W1,W2,b,X,v)
    [z,d] = size(W1);
    [d,z] = size(W2); 
    [d,m] = size(X); 
    [z,~] = size(b); 
    
    J = v*sigma(W1*X+b); 


end


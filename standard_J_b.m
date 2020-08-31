function [ J ] = standard_J_b( W,b,X,v )
    [~,m] = size(X); 
    [z,~] = size(W); 
    J = reshape(diag(reshape(d_sigma(W*X+b),[],1))*repmat(v,m,1),z,m); 
end


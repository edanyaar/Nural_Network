function [ J ] = standard_J_W(  W,b,X,v  )
    [r,~] = size(W);
    [~,m] = size(X);
    J = reshape(reshape(d_sigma(W*X+b),[],1).*kron(X',eye(r))*reshape(v,[],1),r,m);
end


function [ J_W_t_v,J_b_t_v,J_X_t_v ] = standard_J_t_v(  W,b,X,v )
    [r,d] = size(W);
    [d,m] = size(X);
    
    J_W_t_v = reshape((reshape(d_sigma(W*X+b),[],1).*kron(X',eye(r)))'*reshape(v,[],1),r,d);
    d_sig_W_X = d_sigma(W*X+b); 
    J_b_t = zeros(r,r*m);
    for i = 0:m-1 
        J_b_t(:,i*r+1:(i+1)*r) = diag(d_sig_W_X(:,i+1));  
    end
    J_b_t_v = J_b_t*reshape(v,[],1); 
    J_X_t_v = W'*(d_sigma(W*X+b).*v);
end


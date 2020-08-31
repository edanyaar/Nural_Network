function [J_W1_t_times_v,J_W2_t_times_v,J_b_t_times_v,J_X_t_v ] = res_net_J(W1,W2,b,x,v )
    sigma =@(x) tanh(x) ; 
    %sigma =@(x) max(0,x); 
    d_sigma =@(x) 1-tanh(x).^2 ; 
    %d_sigma =@(x) max(0,x); 
    
    J_b_t_times_v = d_sigma(W1*x+b).*(W2'*v);
    J_W1_t_times_v = (d_sigma(W1*x+b).*(W2'*v))*x';
    J_W2_t_times_v = v*sigma(W1*x+b)'; 
    J_X_t_v = v + W1'*(d_sigma(W1*x+b).*W2')*v;

end


function [ J ] = standard_J_X(  W,b,X,v  )
    J = d_sigma(W*X+b).*(W*v);
end


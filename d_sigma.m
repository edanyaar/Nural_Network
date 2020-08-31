function [ res ] = d_sigma( x )
 func =@(x) 1-tanh(x).^2 ; 
 %func =@(x) max(0,x); 
 res = func(x); 

end


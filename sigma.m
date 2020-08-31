function [ res ] = sigma( x )
	func=@(x) tanh(x) ;
    res = func(x); 
end


function [obj,grad_W,grad_X] = my_soft_max( C,W,X )
%C - l label vectors of size m           (m x l)
%W - l weights vectors of size d         (d x l)
%X - m data points (arranged in a matrix)(d x m) 

m = size(X,2);
d = size(X,1);
l = size(W,2);

eta = max(X'*W,[],2);
v = sum(exp(X'*W-eta),2);
u = exp(X'*W-eta);
temp = u./v;
obj = -(1/m)*trace(C'*log(temp)); 
grad_W = (1/m)*X*(temp-C);
grad_X = (1/m)*W*(u'./v'-C');

end


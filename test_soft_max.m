load('SwissRollData.mat');

X = Yt;
C = Ct'; 

%{ 
seperable data: 
m = 100;
d = 2;
l = 2;
X = [1+1.1*rand(2,50),-1+1.1*rand(2,50)];
C = zeros(m,l);
C(1:50,1) = 1; 
C(51:100,2) = 1;
%}

max_iter = 10;
eps = 4;
W = 2*rand(d,l)-1; 
[ obj,grad ,~] = my_soft_max( C,W,X );

direction = rand(size(W));
direction = direction/norm(direction);

test_grad_linear = zeros(max_iter,1);
test_grad_quad = zeros(max_iter,1);


for i = 1:max_iter
   [obj,grad] = my_soft_max(C,W,X);
   [obj_eps,grad_eps] = my_soft_max(C,W+eps*direction,X);
   %%%% grad test:
   test_grad_linear(i) = abs(obj_eps - obj);
   test_grad_quad(i) = abs(obj_eps - obj - eps*reshape(direction,[],1)'*grad);
   eps = eps*0.5;
end

%{
[W,W_arr,obj_arr] = SGD_momentum(C,X);  

succ_precent = zeros(size(obj_arr));
[~,true_labels] = max(Ct',[],2); 
succ_precent_v = zeros(size(obj_arr));
[~,true_labels_v] = max(Cv',[],2); 


for i= 1: size(obj_arr,1)
    pred_labels = classify(Yt,W_arr(:,:,i));
    succ_precent(i) = 1 - size(find(pred_labels-true_labels),1)/size(Yt,2);
    pred_labels_v = classify(Yv,W_arr(:,:,i));
    succ_precent_v(i) = 1 - size(find(pred_labels_v-true_labels_v),1)/size(Yv,2);
end

figure
plot(obj_arr);
xlabel("iteration");
ylabel("softmax objective");
title("Softmax Objective for SwissRoll Data"); 

figure
plot(succ_precent);
hold on 
plot(succ_precent_v);
xlabel("iteration");
ylabel("success precent");
title("Success Precent for SwissRoll Data"); 
legend("train","test"); 

%}
function [labels] = classify(X,W)
    eta = max(X'*W,[],2);
    v = sum(exp(X'*W-eta),2);
    u = exp(X'*W-eta);
    probability_matrix = u./v;
    [~,labels] = max(probability_matrix,[],2); 
end






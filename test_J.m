eps = 1;
max_iter = 10;
d = 4; 
z = 3;
m = 100; 
X = rand(d,m);
X = X/norm(X);
W1 = 2*rand(z,d)-1;
W2 = 2*rand(d,z)-1;
b = randn(z,1);

d_W1 = rand(z,d);
d_W1 = d_W1/norm(d_W1);
d_W2 = rand(d,z);
d_W2 = d_W2/norm(d_W2);
d_b = rand(z,1);
d_b = d_b/norm(d_b);
d_X = rand(d,m);
d_X = d_X/norm(d_X);

test_J_W1_linear = zeros(max_iter,1);
test_J_W1_quad = zeros(max_iter,1);
test_J_b_linear = zeros(max_iter,1);
test_J_b_quad = zeros(max_iter,1);
test_J_X_linear = zeros(max_iter,1);
test_J_X_quad = zeros(max_iter,1);


res_test_J_W1_linear = zeros(max_iter,1);
res_test_J_W1_quad = zeros(max_iter,1);
res_test_J_W2_linear = zeros(max_iter,1);
res_test_J_W2_quad = zeros(max_iter,1);
res_test_J_b_linear = zeros(max_iter,1);
res_test_J_b_quad = zeros(max_iter,1);
res_test_J_X_linear = zeros(max_iter,1);
res_test_J_X_quad = zeros(max_iter,1);
%% test J 

for i = 1:max_iter
   %standard
   obj = sigma(W1*X+b);
   obj_eps = sigma(W1*X+b+eps*d_b);
   test_J_b_linear(i) = norm(obj_eps - obj);
   J_b_t_times_v =standard_J_b(W1,b,X,eps*d_b);
   test_J_b_quad(i) = norm(obj_eps - obj - J_b_t_times_v); 
   
   obj_eps = sigma((W1+eps*d_W1)*X+b);
   test_J_W1_linear(i) = norm(obj_eps - obj);
   J_W1_t_times_v =standard_J_W(W1,b,X,eps*d_W1);
   test_J_W1_quad(i) = norm(obj_eps - obj - J_W1_t_times_v); 
   
   obj_eps = sigma(W1*(X+eps*d_X)+b);
   test_J_X_linear(i) = norm(obj_eps - obj);
   J_X_t_times_v = standard_J_X(W1,b,X,eps*d_X);
   test_J_X_quad(i) = norm(obj_eps - obj - J_X_t_times_v); 
   
   %resnet
   obj = X + W2*sigma(W1*X+b);
   obj_eps = X + W2*sigma(W1*X+b+eps*d_b);
   res_test_J_b_linear(i) = norm(obj_eps - obj);
   J_b_t_times_v =resnet_J_b(W1,W2,b,X,eps*d_b);
   res_test_J_b_quad(i) = norm(obj_eps - obj - J_b_t_times_v); 
   
   obj_eps = X + W2*sigma((W1+eps*d_W1)*X+b);
   res_test_J_W1_linear(i) = norm(obj_eps - obj);
   J_W1_t_times_v =resnet_J_W1(W1,W2,b,X,eps*d_W1);
   res_test_J_W1_quad(i) = norm(obj_eps - obj - J_W1_t_times_v); 
   
   obj_eps = X + (W2+eps*d_W2)*sigma(W1*X+b);
   res_test_J_W2_linear(i) = norm(obj_eps - obj);
   J_W2_t_times_v =resnet_J_W2(W1,W2,b,X,eps*d_W2);
   res_test_J_W2_quad(i) = norm(obj_eps - J_W2_t_times_v - obj); 
   
   eps = eps*0.5;

end


%% test J transpose

u = randn(z,m);
v_W1 = rand(z,d);
v_W2 = rand(d,z);
v_b = rand(z,1);
v_X = rand(d,m);
[J_W_t_u,J_b_t_u,J_X_t_u] = standard_J_t_v(W1,b,X,u); 
test_b = abs(reshape(u,[],1)'*reshape(standard_J_b(W1,b,X,v_b),[],1)-reshape(v_b,[],1)'*reshape(J_b_t_u,[],1));
test_W1 = abs(reshape(u,[],1)'*reshape(standard_J_W(W1,b,X,v_W1),[],1)-reshape(v_W1,[],1)'*reshape(J_W_t_u,[],1));
test_X = abs(reshape(u,[],1)'*reshape(standard_J_X(W1,b,X,v_X),[],1)-reshape(v_X,[],1)'*reshape(J_X_t_u,[],1));




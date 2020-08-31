load('SwissRollData.mat');

X = Yt;
C = Ct';
[l,~] = size(Ct);
[d,~] = size(X);

layer_size = [d,4,5,l]; 
W_arr = init_W(layer_size);
max_iter = 10; 
eps = 0.5;
test_grad_linear = zeros(max_iter,1);
test_grad_quad = zeros(max_iter,1);
d = init_W(layer_size);
d = normalize_cell_arr(d);

for i = 1:max_iter
   [obj,input_arr] = forward_pass(C,W_arr,X,layer_size);
   d_eps = cellfun(@(x) x.*eps ,d,'UniformOutput',false);
   
   W_arr_eps = add_cell_arrays(W_arr,d_eps); 
   obj_eps = forward_pass(C,W_arr_eps,X,layer_size);
   
   test_grad_linear(i) = abs(obj_eps - obj);
   grad = backward_prop(C,W_arr,input_arr,layer_size );
   
   d_eps_t_times_grad = mul_cell_arrays(d_eps,grad);
   test_grad_quad(i) = abs(obj_eps - obj - d_eps_t_times_grad);
   eps = eps*0.5; 
end 


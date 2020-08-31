function [ labels ] = classify(X,W,layer_size)
    f_standard = @(x,W,b) sigma(W*x+b); 
    f_res_net = @(x,W1,W2,b) x + W2*sigma(W1*x+b);

    L = length(layer_size); 
    is_res_net = layer_size_to_is_resnet( layer_size );
    
    input_arr = cell(1,L-1);
    curr_x = X;
    input_arr{1,1} = curr_x;
    for i = 1:L-2
         
       W1 = W{1,i};
       W2 = W{2,i};
       b = W{3,i};
 
       if is_res_net(i) == 1
           curr_x = f_res_net(curr_x,W1,W2,b); 
       else
           curr_x = f_standard(curr_x,W1,b); 
       end
       input_arr{1,i+1} = curr_x;
    end 

    softmax_W = W{1,L-1};
    eta = max(curr_x'*softmax_W,[],2);
    v = sum(exp(curr_x'*softmax_W-eta),2);
    u = exp(curr_x'*softmax_W-eta);
    probability_matrix = u./v;
    [~,labels] = max(probability_matrix,[],2); 

end


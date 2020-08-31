function [obj,input_arr] = forward_pass(C,W_arr,X,layer_size)
    % X - the data points                  (d x m)
    % layer_size - the size of each layer  (1 x L)

    % standard step: 
    % W(i) - (layer_size(i+1) x layer_size(i))
    % b(i) - (layer_size(i+1) x 1)

    % res net step: 
    % W1(i) - (z x layer_size(i))
    % W2(i) - (layer_size(i) x z)
    % b(i) -  (z x 1)
    %(z is a dummy indx) 
    
    % W_arr - 3 x L-1 cell array --> {---W1---}
    %                                {---W2---} (relevant only for res net steps)
    %                                {---b--- }
    

    f_standard = @(x,W,b) sigma(W*x+b); 
    f_res_net = @(x,W1,W2,b) x + W2*sigma(W1*x+b);

    L = length(layer_size); 
    is_res_net = layer_size_to_is_resnet( layer_size );
    
    input_arr = cell(1,L-1);
    curr_x = X;
    input_arr{1,1} = curr_x;
    for i = 1:L-2
         
       W1 = W_arr{1,i};
       W2 = W_arr{2,i};
       b = W_arr{3,i};
 
       if is_res_net(i) == 1
           curr_x = f_res_net(curr_x,W1,W2,b); 
       else
           curr_x = f_standard(curr_x,W1,b); 
       end
       input_arr{1,i+1} = curr_x;
    end 
    [obj,~,~] = my_soft_max(C,W_arr{1,L-1},curr_x); 
    
end


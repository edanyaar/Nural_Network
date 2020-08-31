function [ grad_W ] = backward_prop(C,W_arr,input_arr,layer_size )
   % input_arr - 1 x L-1 cell array the arr of inputs for each layer generated during the
   % forward pass . 
   % input_arr{1,i} = input for layer i (layer_size(i) x m)

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
    grad_W = cell(size(W_arr));
    L = length(layer_size);
    is_res_net = layer_size_to_is_resnet( layer_size );

    [~,grad_W_softmax,grad_X_softmax] = my_soft_max(C,W_arr{1,L-1},input_arr{1,L-1});
    grad_W{1,L-1} = grad_W_softmax;  
    acc = grad_X_softmax; 
    
    for i = (L-2):-1:1
       W1 = W_arr{1,i};
       W2 = W_arr{2,i};
       b = W_arr{3,i};
       X = input_arr{1,i};
       
       if is_res_net(i)==1
          [J_W1_t_acc,J_W2_t_acc,J_b_t_acc,J_X_t_acc] = res_net_J(W1,W2,b,X,acc);
          grad_W{1,i} = J_W1_t_acc; 
          grad_W{2,i} = J_W2_t_acc;
          grad_W{3,i} = J_b_t_acc;
       else
          [J_W1_t_acc,J_b_t_acc,J_X_t_acc] = standard_J_t_v(W1,b,X,acc);
          grad_W{1,i} = J_W1_t_acc; 
          grad_W{3,i} = J_b_t_acc;
       end
       acc = J_X_t_acc; 
    end
    

end


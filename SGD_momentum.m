function [ W, W_arr,obj_arr] = SGD_momentum(C,X,layer_size,max_epochs,batch_size)
    m = size(X,2);
    W = init_W(layer_size);% initial value 
    alpha = 0.1;
    gamma = 0.7; 
    
    obj_arr = zeros(max_epochs,1);
    W_arr = cell(max_epochs,1);
    
    for epoch = 1:max_epochs
        M = cellfun(@(x) x.*0 ,W,'UniformOutput',false); % creat a zeros cell array with the same size as W 
        idxs = randperm(m);
        lr = alpha/sqrt(epoch); %learning rate 
        for k = 0:((m/batch_size)-1)
            idx_b = idxs(k*batch_size+1 : (k+1)*batch_size); 
            X_b = X(:,idx_b);
            C_b = C(idx_b,:);
            [obj,input_arr] = forward_pass(C_b,W,X_b,layer_size);
            grad = backward_prop(C_b,W,input_arr,layer_size); 
            M = add_cell_arrays(cellfun(@(x) x.*gamma,M,'UniformOutput',false),cellfun(@(x) x.*lr,grad,'UniformOutput',false));
            W = add_cell_arrays(W, cellfun(@(x)x.*(-1),M,'UniformOutput',false));
        end
        obj_arr(epoch) = obj; 
        W_arr{epoch,1} = W; 
    end
end


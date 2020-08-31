function W_arr = init_W(layer_size)
    L = length(layer_size); 
    W_arr = cell(3,L-1);
       
    is_res_net = layer_size_to_is_resnet( layer_size );
    
    for i = 1:L-1
       if is_res_net(i) == 1
           z = layer_size(i)+1;    
           W_arr{1,i} = randn(z,layer_size(i));
           W_arr{2,i} = randn(layer_size(i),z);
           W_arr{3,i} = randn(z,1);
       else
           W_arr{1,i} = randn(layer_size(i+1),layer_size(i));
           if i < L-1
                W_arr{3,i} = randn(layer_size(i+1),1);
           end
       end  
    end
    W_arr{1,L-1}= W_arr{1,L-1}'; 
end

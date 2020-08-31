function [ is_res_net ] = layer_size_to_is_resnet( layer_size )
    L = length(layer_size);
    is_res_net = zeros(1,L-1); 

    for i = 1:L-1
        if layer_size(i) == layer_size(i+1)
           is_res_net(i) = 0;  
        end
    end

end


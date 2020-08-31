function [ product ] = mul_cell_arrays (c1,c2 )
    
    product = 0; 
    [n_row,n_col] = size(c1);
    for i = 1:n_row
       for j = 1:n_col
           product = product + sum(sum(c1{i,j}.*c2{i,j})); 
       end
    end
end


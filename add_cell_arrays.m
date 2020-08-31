function [ res ] = add_cell_arrays (c1,c2 )
    
    res = cell(size(c1)); 
    [n_row,n_col] = size(c1);
    for i = 1:n_row
       for j = 1:n_col
           res{i,j} = c1{i,j}+c2{i,j}; 
       end
    end
end


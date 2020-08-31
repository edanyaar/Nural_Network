function [ normalized_cell_arr ] = normalize_cell_arr (cell_array )
    [n_row,n_col] = size(cell_array); 
    cell_array_norm_sqr = 0; 
    for i = 1:n_row
       for j = 1:n_col
          cell_array_norm_sqr = cell_array_norm_sqr + norm(cell_array{i,j},'fro')^2;   
       end
    end
    normalized_cell_arr = cellfun(@(x) x./(sqrt(cell_array_norm_sqr)), cell_array,'UniformOutput',false); 

end


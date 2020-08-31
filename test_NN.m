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

% W_arr - 3 x L cell array --> {---W1---}
%                              {---W2---} (relevant only for res net steps)
%                              {---b--- }



load('PeaksData.mat');
X = Yt;
C = Ct';
[l,~] = size(Ct);
[d,~] = size(X);
layer_size = [d,5,5,l]; 
max_epochs = 100; 
batch_size = 100; 



succ_precent_v = zeros(max_epochs,1);
[~,true_labels_v] = max(Cv',[],2);
succ_precent_t = zeros(max_epochs,1);
[~,true_labels_t] = max(Ct',[],2); 

[ final_W, W_arr,obj_arr] = SGD_momentum(C,X,layer_size,max_epochs,batch_size);
for i = 1:max_epochs
    pred_labels_v = classify(Yv,W_arr{i,1},layer_size); 
    succ_precent_v(i) = 1 - size(find(pred_labels_v-true_labels_v),1)/size(Yv,2);
    pred_labels_t = classify(Yt,W_arr{i,1},layer_size); 
    succ_precent_t(i) = 1 - size(find(pred_labels_t-true_labels_t),1)/size(Yt,2);
end
%%
figure
plot(obj_arr);
xlabel("iteration");
ylabel("NN objective");
title(sprintf("NN Objective for Peaks Data \n bs: %d, epochs:%d",batch_size,max_epochs)); 

figure
plot(succ_precent_t);
hold on 
plot(succ_precent_v);
xlabel("iteration");
ylabel("success precent");
title(sprintf("Success Precent for Peaks Data \n bs: %d, epochs:%d",batch_size,max_epochs)); 
legend("train","verification"); 




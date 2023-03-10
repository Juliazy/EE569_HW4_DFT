clc; clear all;
X = load('py_mnist.mat');
XTrain = X.image;
YTrain = X.label;

XTrain = reshape(XTrain, [60000, 784]);
% XTrain = XTrain';
XTrain = double(XTrain);
size(XTrain)
YTrain = double(YTrain)';
num_class = 10;

%%
size(XTrain)
size(YTrain)

featdim = 784;
XTrain_test = XTrain(:,1:featdim);
size(XTrain_test)

%%
dft_loss = get_all_loss(XTrain_test, double(YTrain), num_class);
size(dft_loss)
figure;plot(1:featdim, dft_loss);
hold;
plot(1:featdim, sort(dft_loss));

% classwise = zeros(1, num_class);
% for c = 1:num_class
%     classwise(1, c) = sum(YTrain(:) == c);
% end
% %%
% x = XTrain(:, 100);
% y = YTrain(:,1);
% feat_loss, all_loss = binning(x, y, classwise, 32, num_class);
%         

%%
function [entropy] = cal_entropy_from_y(y_array, num_cls)

    prob = zeros(1,num_cls);
    for c = 1:num_cls
        prob(c) = sum(y_array==c-1)./length(y_array);
    end 
    prob = prob./sum(prob);

    tmp = 0;
    for i = 1:length(prob)
        
        if prob(i) > 0
            tmp = tmp - prob(i) * log(prob(i));
        end
    end
    entropy = tmp./log(num_cls);

end

function [wH] = cal_weighted_H(X, y, bound,num_cls)
    if (sum(X<bound)==0 || sum(X>=bound)==0)
        wH = 1;
        return 
    else
        left_y = y(X<bound);
        right_y = y(X>=bound);
        left_num = length(left_y);
        right_num = length(right_y);

        left_entropy = cal_entropy_from_y(left_y, num_cls);
        right_entropy = cal_entropy_from_y(right_y,  num_cls);

        wH = left_num/(left_num + right_num) * left_entropy + right_num/(left_num + right_num) * right_entropy;
    end
    
end


function [best_loss] = binning(x, y,B, num_class)
    min_x = min(x);
    max_x = max(x);
    
    if max_x == min_x
        best_loss = 1;
        return 
    end
    
    % B bins (B-1) candicates of partioning point
    candidates = (min_x: (max_x - min_x) / B: max_x);
    candidates = candidates(2:length(candidates)-1);
    candidates = unique(candidates);
    loss_i = zeros(1,length(candidates));

    for idx=(1:length(candidates))

        loss_i(idx) = cal_weighted_H(x, y, candidates(idx), num_class);
       
    end
    %figure;
    %plot(1:length(candidates), loss_i);
        
    best_loss = min(loss_i);

end

function [feat_loss] = get_all_loss(X, Y, num_class)

    [N , D] = size(X);
    feat_loss = zeros(1, D);
    
    for k = 1:D
        x = X(:, k);
        y = Y;
        y = y(:, 1);
        feat_loss(k) = binning(x, y, 16, num_class);
    end
end



        
        
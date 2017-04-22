% 除非特别注明，以下X均指包含单位1的项，且训练集按行排列
% 多类别分类

%%%%%%%%使用%%%%%%%%%%%%
%%%训练参数
%X不包含单位1项%
[all_theta] = oneVsAll(X, y, num_labels, lambda);

% 用于预测
%X不包含单位1项%
pred = predictOneVsAll(all_theta, X);






%%%%%%% 被调用函数 %%%%%%%%%%
%%%计算代价函数和梯度
function [J, grad] = lrCostFunction(theta, X, y, lambda)







%%%%%%%%%  函数定义 %%%%%%%%%%%%
function [all_theta] = oneVsAll(X, y, num_labels, lambda)

m = size(X, 1);
n = size(X, 2);

all_theta = zeros(num_labels, n + 1);

X = [ones(m, 1) X];


	
for iter = 1:num_labels
	
	initial_theta = zeros(n + 1, 1);
	% 迭代参数需要按需调整
  options = optimset('GradObj', 'on', 'MaxIter', 50);
	temp = fmincg (@(t)(lrCostFunction(t, X, (y == iter), lambda)), initial_theta, options);
  all_theta(iter,:) = temp';

end
	
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p = predictOneVsAll(all_theta, X)

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];


pre = X*all_theta';
[temp,p] = max(pre,[],2);
end












%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

h = sigmoid(X*theta);
theta_lambda = theta(2:length(theta));
J = -1/m*(y'*log(h) + (1-y)'*log(1-h)) + lambda/(2*m)*theta_lambda'*theta_lambda;
grad = 1/m*X'*(h - y) + lambda/m*theta;
grad(1) = 1/m*X(:,1)'*(h - y);
grad = grad(:);

end


function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

list_C = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
list_sigma = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
nC = size(list_C,1);
nSigma = size(list_sigma,1);

tempC = 0;
tempSigma = 0;
Error = 1.0;
for i = 1:nC
	tempC = list_C(i);
	for j = 1:nSigma
		tempSigma = list_sigma(j);
		model = svmTrain(X, y, tempC, @(x1, x2) gaussianKernel(x1, x2, tempSigma));
		
		predictions = svmPredict(model, Xval);
		tempError = mean(double(predictions ~= yval));
		if tempError<Error
			Error = tempError;
			C = tempC;
			sigma = tempSigma;
		end
	end
end




% =========================================================================

end

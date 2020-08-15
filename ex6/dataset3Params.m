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
Clist = [0.01, 0,03, 0.1, 0.3, 1, 3, 10, 30];
sigmalist = [0.01, 0,03, 0.1, 0.3, 1, 3, 10, 30];
error = 999999;
Cbest= 0;
sigmabest = 0;
predictions = 0;

% looping over the suggested list of values, each time training a model and
% comparing the error. Returning the C, sigma from model with minimum error
% on CV set.
for ii = 1:length(Clist)
    for jj = 1:length(sigmalist)
        C = Clist(ii);
        sigma = sigmalist(jj);
        model = svmTrain(X, y, C, @(x1, x2)gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        if error >= mean(double(predictions ~= yval))
            error = mean(double(predictions ~= yval));
            Cbest = C;
            sigmabest = sigma;
        end
    end
end

C = Cbest;
sigma = sigmabest;
        
        
        








% =========================================================================

end

function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%


% initializing output vector num_labels x 1
op_y = zeros(num_labels,1); % 10 x 1 vector of 0s
% working with one training example i at a time - adding cost sequentially

delta3 = zeros(num_labels,1);
delta2 = zeros(size(Theta2, 2),1);
for i = 1:m
% calculate hidden layer
    a1 = [1 X(i,:)]'; %401x1
    z2 = Theta1*a1; %25x1
    a2 = [1;sigmoid(z2)]; % This is a 26x1 vector
    z3 = Theta2*a2; %10x1
    a3 = sigmoid(z3); % 10x1
    op_y = zeros(num_labels,1); % resetting this vector (10x1)
% Define y by checking y(i,:)
    op_y(y(i,1), 1) = 1; % 10x1 vector
% Now calculating J (cost function) using this output
    J = J + ((-op_y' * log(a3)) - ((1 - op_y)'* (log(1 - a3)))) + (lambda/(2*m))*(sum(Theta1(:,2:end) .^2 ,'all') + sum(Theta2(:,2:end) .^2, 'all'));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%


    delta3 = a3 - op_y; % 10x1 matrix
    meh = (Theta2(:,2:end)' * delta3); % This confused me the most
    delta2 = meh .* sigmoidGradient(z2);
    
    Theta1_grad = Theta1_grad + delta2*a1'; %25x401
    Theta2_grad = Theta2_grad + delta3*a2'; %10x26
    
end

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


J = (1/m) * J;
Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end



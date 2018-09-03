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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add the bias value in X
X = [ones(m, 1) X];
K = size(Theta2, 1); % number of classes


% Making a matrix for y
y_new = zeros(m, K);
for i = 1:m
    y_new(i, y(i)) = 1;
end

% Forward propogation
z_2 = X * transpose(Theta1);
a_2 = sigmoid(z_2);
a_2 = [ones(m, 1) a_2];
z_3 = a_2 * transpose(Theta2);
a_3 = sigmoid(z_3);
htheta_x = a_3;

% Computing Cost
J_matrix = (-1 .* y_new .* log(htheta_x)) + (-1 .* (1 - y_new) .* log(1 - htheta_x));
J = 1/m * sum(J_matrix(:));


Theta1_wout_bias = Theta1(:, 2:end);
Theta2_wout_bias = Theta2(:, 2:end);
regularizer_term = (lambda/ (2*m)) * sum(([Theta1_wout_bias(:); Theta2_wout_bias(:)].^2));

J = J + regularizer_term;

Theta_1_Del= zeros(size(Theta1));
Theta_2_Del= zeros(size(Theta2));

for t = 1:m
    % forward propogation
    x_t = transpose(X(t, :)); % 401 * 1
    a_1 = x_t;
    z_2 = Theta1 * a_1; % 25 * 1
    a_2_orig = sigmoid(z_2); % 25 * 1
    a_2 = [1; a_2_orig]; % 26 * 1
    z_3 = Theta2 * a_2; % 10 * 1
    a_3_orig = sigmoid(z_3); % 10 * 1
    htheta_x_t = a_3_orig; % 10 * 1
        
    % backward propogation
    del_3 = htheta_x_t - transpose(y_new(t, :)); % 10 * 1
    del_2 = (transpose(Theta2) * del_3) .* sigmoidGradient([1; z_2]); % 26 * 1
    del_2 = del_2(2:end); % 25 * 1
    Theta_1_Del = Theta_1_Del + (del_2 * transpose(a_1)); % 25 * 401
    Theta_2_Del = Theta_2_Del + (del_3 * transpose(a_2));
end
    
%{
for i = 1:m
    htheta_x_i = transpose(htheta_x(i, :));
    y_i = transpose(y_new(i, :));
    del_3 = htheta_x_i - y(i);
    size(a_2)
    del_2 = (transpose(Theta2) * del_3) .* transpose(a_2(i, :) .* (1 - a_2(i, :)));
    del_2 = del_2(2:end);
    Theta_1_Del = Theta_1_Del + (del_2 * (X(i, :)));
    Theta_2_Del = Theta_2_Del + (del_3 * a_2(i, :));
end
%}

Theta1_grad = (Theta_1_Del ./ m) + [zeros(size(Theta1, 1), 1) ((lambda/m) .* Theta1(:, 2:end))];
Theta2_grad = (Theta_2_Del ./ m) + [zeros(size(Theta2, 1), 1) ((lambda/m) .* Theta2(:, 2:end))];
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

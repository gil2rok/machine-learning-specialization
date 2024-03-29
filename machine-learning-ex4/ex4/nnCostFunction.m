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

%Forward Prop
a1 = [ones(m, 1), X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1), a2]; %adding column of bias units
z3 = a2 * Theta2';
a3 = sigmoid(z3);

%Putting y into a binary, vectorizable matrix
y_matrix = zeros(m, num_labels);  %5000 x 10
for i = 1:m
    y_matrix(i, y(i)) = 1;
end

%Cost Func Eqn
J = (-1/m) * sum(sum(y_matrix .* log(a3) + (1 - y_matrix) .* log(1 - a3))) % log(h) = 5000 x 10
Theta1_no_bias = Theta1(:, 2:end); %not including the bias unit
Theta2_no_bias = Theta2(:, 2:end); %not including the bias unit
J_reg = lambda/(2 * m) * (sum(sum((Theta1_no_bias.^2))) + sum(sum((Theta2_no_bias.^2)))); %Calculating the regularized term in the cost function J
J = J + J_reg;

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

d3 = a3 - y_matrix;
d2 = d3 * Theta2(:, 2:end) .* sigmoidGradient(z2); %removing bias unit from Theta2
D1 = d2' * a1;
D2 = d3' * a2;
Theta1_grad = D1 / m;
Theta2_grad = D2 / m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_reg = Theta1;
Theta2_reg = Theta2;
Theta1_reg(:, 1) = 0;
Theta2_reg(:, 1) = 0;
Theta1_reg = Theta1_reg * (lambda / m);
Theta2_reg = Theta2_reg * (lambda / m);

Theta1_grad = Theta1_grad + Theta1_reg;
Theta2_grad = Theta2_grad + Theta2_reg;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

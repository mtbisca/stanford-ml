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

% forward prop
X = [ones(m, 1) X];
output_1 = sigmoid(X * transpose(Theta1));
output_1 = [ones(m, 1) output_1];
output_final  = sigmoid(output_1 * transpose(Theta2));

% computing cost
for i = 1 : num_labels
  current_y = y == i;
  J = J + (sum((-current_y .* log(output_final(:,i))) - ((1 - current_y) .* log(1 - output_final(:,i)))) / m);
end

% regularization
temp1 = Theta1;
temp1(:,1) = 0;
temp2 = Theta2;
temp2(:,1) = 0;
reg = (lambda /( 2 * m)) * (sum(power(temp1, 2)(:)) +  sum(power(temp2, 2)(:)));
J = J + reg;

% backprop
error_3 = zeros(num_labels, 1);
error_2 = zeros(num_labels, 1);
delta_1 = 0;
delta_2 = 0;
for i = 1 : m
  a1 = X(i, :);
  z2 = Theta1 * transpose(a1);
  a2 = [1 ; sigmoid(z2)];
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);
  % computing error3
  for k = 1: num_labels
    current_y = y == k;
    error_3(k) = a3(k) - current_y(i);
  end

  % computing error2
  error_2 = (transpose(Theta2) * error_3) .* sigmoidGradient([1 ; z2]);

  delta_2 = delta_2 + error_3 * transpose(a2);
  delta_1 = delta_1 + error_2(2:end) * a1;
end
Theta2_grad = delta_2 / m + (lambda / m) * sum(temp2);
Theta1_grad = delta_1 / m + (lambda / m) * sum(temp1);
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

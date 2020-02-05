function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%Input Layer
X = [ones(m, 1) X];

for i = 1:m
    temp = X(i, :);
    temp = temp';
    
    %Hidden Layer #1
    z2 = sum(Theta1 * temp, 2);
    a2 = 1 ./ (1 + exp(-1 * z2)); %sigmoid hypothesis
    
    %Output Layer
    a2 = [1; a2]; %adding the bias unit
    z3 = sum(Theta2 * a2, 2);
    a3 = 1 ./ (1 + exp(-1 * z3)); %sigmoid hypothesis
    
    %Figuring out which number is the output
    [a, b] = max(a3, [], 1);
    p(i) = b;
end




% =========================================================================


end

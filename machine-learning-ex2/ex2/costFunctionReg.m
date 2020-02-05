function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%Calculating J
h = 1 ./ (1 + exp(-1 * (X * theta))); %sigmoid hypothesis
Jorig = (-1/m) * (y' * log(h) + (1 - y)' * log(1 - h));
Jreg = sum((lambda / (2*m)) * (theta(2:end)).^2);
J = Jorig + Jreg;

%Calculating gradient for theta0
X0 = X(:,1);
theta0 = theta(1);
h0 = 1 ./ (1 + exp(-1 * (X0 * theta0))); %sigmoid hypothesis for theta 0.
grad1 = (1/m) * X0' * (h - y); %h0

%Calculating gradient for theta >= 1 aka the rest of theta values
XRest = X(:, 2:end);
thetaRest = theta(2:end);
hRest = 1 ./ (1 + exp(-1 * (XRest * thetaRest))); %sigmoid hypothesis for theta >= 1.
grad2 = (1/m) * XRest' * (h - y) + ((lambda / m) * thetaRest); %hRest

grad = [grad1; grad2];


% =============================================================

end

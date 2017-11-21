function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

len = length(theta);
newTheta = zeros(len,1);

for iter = 1:num_iters
	% Setting new theta
	for i = 1:len
        newTheta(i) = theta(i) - ((alpha/m) * sum(((X * theta) - y) .* X(:,i)) );
	end
	
	% Simulltaneously update theta values (Attributing new theta)
    for i = 1:len	
		theta(i) = newTheta(i);
	end

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
end

end

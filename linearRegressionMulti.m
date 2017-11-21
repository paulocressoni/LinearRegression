%% Linear regression with multiple variables

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('dataset.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

%%% Using Gradient Descend %%%

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

%% Feature Normalization
[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.1;
%alpha = 0.01;
%% If your learning rate is too large, J(θ) can diverge and ‘blow up’, 
%% resulting in values which are too large for computer calculations.

% Choose the number of iterations
num_iters = 50;

%% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

%%% Estimate the price of a 1650 sq-ft, 3 br house %%%
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
house_size = (1650 - mu(1,1)) / sigma(1,1);
num_bedroom = (3 - mu(1,2)) / sigma(1,2);
price = [1, house_size, num_bedroom] * theta;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);
fprintf('\n');




%%% doing the same using Normal Equations %%%

%% Load Data
data = csvread('dataset.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);
%% Does not need Feature Normalization when using Normal Equation

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
price = [1, 1650, 3] * theta;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

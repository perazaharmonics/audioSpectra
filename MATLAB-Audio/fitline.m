% FITLINE: Estimates the Energy Decay Rate using Linear Regression
% This function fits a linear model to the given data points.
% It is designed to handle situations where the data may contain noise.
%
% Usage:
%   [slope, offset] = fitline(x, y);
%
% Inputs:
%   x     - Independent variable (column vector).
%   y     - Dependent variable (column vector).
%
% Outputs:
%   slope  - Slope of the fitted line, representing the energy decay rate.
%   offset - Y-intercept of the fitted line.
%
% The function implements the normal equations method for linear regression.

function [slope, offset] = fitline(x, y)
    % Ensure that x and y are column vectors.
    x = x(:);
    y = y(:);
    
    % Construct the design matrix 'phi'.
    % The first column contains values of 'x' and the second column contains all ones.
    % This design matrix allows us to model the linear relation: y = slope*x + offset.
    phi = [x, ones(length(x), 1)];
    
    % Compute the projection of 'y' onto the columns of 'phi'.
    % This step essentially computes the weighted sum of y-values for each of the basis functions in 'phi'.
    p = phi' * y;
    
    % Compute the auto-correlation matrix of 'phi'.
    % This matrix captures the overlap between the two basis functions in 'phi'.
    r = phi' * phi;
    
    % Solve the system of normal equations to get the [slope; offset].
    % This step determines the best fit line parameters.
    t = r \ p;
    
    % Extract the slope and offset from the solution.
    slope = t(1);
    offset = t(2);
end

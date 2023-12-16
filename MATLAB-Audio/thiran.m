% THIRAN Fractional Delay Filter Coefficients Calculation
% This function computes the coefficients of the Thiran all-pass fractional 
% delay filter of specified order and delay. The Thiran filter is 
% characterized by its maximally flat group delay response, making it 
% suitable for applications requiring low distortion of the delayed signal.
%
% Syntax:
%   [A, B] = thiran(D, N)
%
% Inputs:
%   D   - Desired fractional delay, specified as a real number. It represents 
%         the delay in terms of sample intervals. The value of D should be 
%         non-negative.
%   N   - The order of the Thiran filter, specified as a non-negative 
%         integer. Higher order filters yield a more accurate approximation 
%         of the desired fractional delay at the expense of increased 
%         computational complexity.
%
% Outputs:
%   A   - A row vector of length N+1 containing the feedforward filter 
%         coefficients.
%   B   - A row vector of length N+1 containing the feedback filter 
%         coefficients in reverse order.
%
% Example:
%   % Compute the coefficients of a Thiran filter with a fractional delay of 
%   % 0.45 and an order of 5
%   [A, B] = thiran(0.45, 5);
%
% Note: 
%   The Thiran filter is an all-pass filter, meaning it has a flat frequency 
%   response in magnitude. It is used to achieve a specified phase shift 
%   (or time delay) that is a function of frequency.

function [A, B] = thiran(D, N)
    A = zeros(1, N+1);
    for k=0:N
        Ak = 1;
        for n = 0:N
            Ak = Ak * (D - N + n) / (D - N + k + n);
        end
        A(k+1) = (-1)^k * nchoosek(N, k) * Ak;
    end
    B = A(N+1:-1:1);
end

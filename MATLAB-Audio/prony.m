function [B, A] = prony(h, N, M)
    % PRONYSPECTRALSUBTRACTION Uses Prony's method to compute the filter
    % coefficients for spectral subtraction based on a given impulse response.
    %
    % Input:
    %   h - Impulse response of the system
    %   N - Order of the numerator (zeros)
    %   M - Order of the denominator (poles)
    %
    % Output:
    %   B - Numerator coefficients of the filter
    %   A - Denominator coefficients of the filter
    
    % Step 1: Set up the Hankel matrix from the impulse response
    H = hankel(h(1:M), h(M:end));
    
    % Step 2: Compute the SVD (Singular Value Decomposition)
    [U, S, V] = svd(H, 'econ');
    
    % Step 3: Form the system of equations from the SVD and solve for poles
    S1 = S(1:M, 1:M);
    S2 = S(1:N, 1:N);
    U1 = U(:, 1:M);
    U2 = U(:, 1:N);
    V1 = V(:, 1:M);
    V2 = V(:, 1:N);
    
    A = [U1' * H * V2, -S2] \ (U1' * h(M+1:end));
    A = [1; A];
    
    % Step 4: Solve for the zeros (numerator coefficients) using the least squares method
    B = [eye(N+1), zeros(N+1, M); zeros(M, N+1), S1] \ [h(1:N+1); U1' * h(1:M)];
    B = B(1:N+1);
end

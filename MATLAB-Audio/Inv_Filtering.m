freq = 104.98;      %estimated peak frequency in Hz
bw = 10;            % Peak bandwidth estimate
bodyIR = 1;         % Tune according to guitar body

R = exp( - 01 * bw / fs);       % Pole radius
z = R * exp(j* 2 * freq / fs);  % Pole in-itself
B = [1, -(z + conj(z)), z * conj(z)]; % numerator
r = 0.9;        % zero / factor (notch isolation)
A = B.*(r.^[0 : length(B) - 1]);    % denominator

residual = filter(B, A, bodyIR);    % Apply inverse filter


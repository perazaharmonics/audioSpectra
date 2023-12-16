% Upsampling factor
L = 4;

% Original signal
Fs = 44100; % Sampling frequency
t = 0:1/Fs:1-1/Fs; % Time vector

% Gaussian envelope parameters
mu = 0.5; % Mean of the Gaussian envelope
sigma = 0.1; % Standard deviation of the Gaussian envelope

% Generate Gaussian envelope
envelope = exp(-0.5 * ((t - mu) / sigma).^2);

% Generate a sine wave signal modulated by a Gaussian envelope
x = envelope .* sin(2*pi*5*t); 


% Interpolated signal
xi = zeros(1, L*length(x));
xi(1:L:end) = x; % Insert original samples into upsampled signal

% Apply Lagrange interpolation filter for each missing sample
N = 5; % Degree of the Lagrange polynomial
for i = 1:L-1
    delay = i;
    h = lagrange(N, delay);
    filtered = conv(xi, h, 'same');
    xi(i+1:L:end) = filtered(i+1:L:end);
end

% Usage of the thiran function
D = 0.45;  % Fractional delay
N = 5;     % Filter order
[A, B] = thiran(D, N);  % Compute Thiran filter coefficients

% Compute frequency response
% Compute and plot frequency response with modified axes and increased resolution
[Hz, Freq] = freqz(B, A, 'half', 4096);  % Increase the number of points for a smoother response


% Plotting
figure;
subplot(2,1,1);
plot(t, x);
title('Original Signal');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(2,1,2);
ti = 0:1/(L*Fs):1-1/(L*Fs); % Time vector for interpolated signal
plot(ti, xi);
title('Interpolated Signal Using Lagrange Interpolation');
xlabel('Time (s)');
ylabel('Amplitude');

% Plot magnitude response
figure;

subplot(2, 1, 1);
plot(Freq/(2*pi)*4096, 20*log10(abs(Hz)));
title('Magnitude Response of the Thiran Filter');
xlabel('Frequency (normalized)');
ylabel('Magnitude (dB)');

% Focus on a specific frequency range and adjust y-axis limits
xlim([0 0.5]);  % Adjust the x-axis limits to focus on the frequency range of interest
ylim([-50 5]);  % Adjust the y-axis limits to better visualize the magnitude response

% Plot phase response
subplot(2, 1, 2);
plot(Freq/(2*pi)*4096, rad2deg(angle(Hz)));
title('Phase Response of the Thiran Filter');
xlabel('Frequency (normalized)');
ylabel('Phase (degrees)');
% Plot phase response
subplot(2,1,2);
plot(Freq/(2*pi)*(L*Fs), rad2deg(angle(Hz)));
title('Phase Response of the Lagrange Interpolation Filter');
xlabel('Frequency (Hz)');
ylabel('Phase (degrees)');
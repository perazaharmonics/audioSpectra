% This script models the excitation of a piano string using a digital
% waveguide model and calculates the frequency response of the system.
% The Kelly-Lochbaum Algorithm is used to simulate the reflection and
% transmission of waves along the string. An impulse is generated to
% represent the hammer striking the string, and the response of the system
% is calculated. The frequency response is then obtained using the Fast
% Fourier Transform (FFT) and plotted along with the spectrum and cepstrum.

% Define the given parameters for the hammer-string interaction
k = 1000; % N/m, stiffness constant for the hammer-string interaction
p = 2; % Exponent indicating the nonlinearity of the hammer-string interaction
y_h = 0.01; % m, initial displacement of the hammer before it strikes the string
y_s = 0; % m, initial displacement of the string, assuming it starts at rest

% Calculate the force applied by the hammer to the string using the equation F_h = k * (y_h - y_s)^p
F_h = k * (y_h - y_s)^p;

% Define the sampling frequency and create a time vector for the simulation
fs = 44100; % Sampling frequency in Hz, standard for audio applications
t = 0:1/fs:0.1; % Time vector of 0.1 seconds duration for the simulation

% Create an impulse representing the hammer strike at time zero
impulse = zeros(size(t));
impulse(1) = F_h; % The impulse magnitude is set to the calculated force F_h

% Define the digital waveguide model parameters
% Assume a single reflection point in the waveguide for simplicity
reflection_coefficient = -0.5; % Reflection coefficient at the boundary
delay_samples = round(fs * 0.002); % Number of samples for the delay based on a 2ms round trip

% Implement the Kelly-Lochbaum Algorithm to simulate wave propagation
output = zeros(size(t)); % Initialize the output vector
buffer = zeros(1, delay_samples); % Buffer to simulate the delay in the waveguide

for n = 1:length(t)
    % Calculate the current input as the sum of the direct impulse and the reflected wave
    current_input = impulse(n) + reflection_coefficient * buffer(end);
    
    % Update the buffer with the current input
    buffer = [current_input buffer(1:end-1)];
    
    % The output at each time step is a combination of the transmitted and reflected waves
    output(n) = current_input * (1 + reflection_coefficient);
end

% Compute the frequency response using FFT
N = length(output); % Number of points in FFT
frequency_response = fft(output, N); % Perform the FFT on the output signal
magnitude_response = abs(frequency_response/N); % Get the magnitude of the frequency response

% Frequency vector for plotting
f = (0:(N/2))*(fs/N);

% Plot the frequency response of the system
figure;
plot(f, magnitude_response(1:N/2+1));
title('Frequency Response of the System');
xlabel('Frequency (Hz)');
ylabel('|H(f)|');
grid on;

% Compute the two-sided spectrum
spectrum = fft(output);
magnitude_spectrum = abs(spectrum/N); % Normalize the magnitude of the FFT

% Compute the single-sided spectrum
single_sided_spectrum = magnitude_spectrum(1:floor(N/2)+1);
single_sided_spectrum(2:end-1) = 2*single_sided_spectrum(2:end-1);

% Plot the spectrum of the system
figure;
plot(f, single_sided_spectrum);
title('Spectrum of the System');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;

% Compute the log magnitude spectrum for the cepstrum
log_magnitude_spectrum = log(magnitude_spectrum + eps); % Add eps for numerical stability

% Compute the cepstrum
cepstrum = ifft(log_magnitude_spectrum);

% Plot the real cepstrum of the system
figure;
quefrency = (0:N-1) * (1/fs);
plot(quefrency, real(cepstrum));
title('Real Cepstrum of the System');
xlabel('Quefrency (s)');
ylabel('Amplitude');
grid on;

% Note: The above code assumes that the output is a real signal and thus plots the real part of the cepstrum.


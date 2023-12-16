% Sampling frequency
fs = 44100; % Standard CD quality sampling rate

% Define the resonant frequency and Q-factor for the Bi-Quad filter
f_resonant = 98.00; % Example resonant frequency in Hz
Q = 0.87; % Example quality factor

% Calculate normalized frequency
omega = 2 * pi * f_resonant / fs;
alpha = sin(omega) / (2 * Q);

% Coefficients for a Bi-Quad resonant filter (example for a band-pass filter)
b0 = alpha;
b1 = 0;
b2 = -alpha;
a0 = 1 + alpha;
a1 = -2 * cos(omega);
a2 = 1 - alpha;

% Normalize the filter coefficients
b = [b0, b1, b2] / a0;
a = [1, a1, a2] / a0;

% Create a transfer function model for the Bode plot
H = tf(b, a, 1/fs);

% Generate the Bode plot
figure;
[mag, phase, w] = bode(H, {10, fs/2}); % Frequency range from 10 Hz to Nyquist frequency
mag = reshape(mag, length(w), 1); % Reshape for plotting
phase = reshape(phase, length(w), 1); % Reshape for plotting

subplot(2, 1, 1);
semilogx(w/(2*pi), 20*log10(mag));
title('Magnitude Response of Bi-Quad Filter');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
grid on;

subplot(2, 1, 2);
semilogx(w/(2*pi), phase);
title('Phase Response of Bi-Quad Filter');
xlabel('Frequency (Hz)');
ylabel('Phase (Degrees)');
grid on;

% Generate a test signal (sine wave at the resonant frequency)
t = 0:1/fs:1; % 1 second duration
test_signal = sin(2 * pi * f_resonant * t);

% Apply the filter to the test signal
filtered_signal = filter(b, a, test_signal);

% Play the filtered signal
sound(filtered_signal, fs);
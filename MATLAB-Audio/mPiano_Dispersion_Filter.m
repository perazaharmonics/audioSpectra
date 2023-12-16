% Read data
[signal, fs] = audioread('D4.wav');
% If the signal is stereo, take one channel
if size(signal, 2) == 2
    signal = signal(:, 1); % Take the left channel or right channel
end

% Perform Fourier transform
N = length(signal);
fftSignal = fft(signal);

% Calculate the two-sided spectrum and then the one-sided spectrum
twoSidedSpectrum = abs(fftSignal/N);
oneSidedSpectrum = twoSidedSpectrum(1:N/2+1);
oneSidedSpectrum(2:end-1) = 2*oneSidedSpectrum(2:end-1);

% Define the frequency domain f
f = fs*(0:(N/2))/N;

% Plot the spectrum
figure;
plot(f, oneSidedSpectrum);
title('Frequency Spectrum of WAV file');
xlabel('Frequency (Hz)');
ylabel('|Magnitude|');
grid on;

% Given desired frequencies (in Hz) of the piano sample D4.Wav
f_desired = [294.455, 588.801, 883.905, 1180.31, 1778.4, 2079.78];

% Convert the desired frequencies to radian frequencies
omega_desired = 2 * pi * f_desired;

% Number of partials
K = length(f_desired);

% Initialize variables for M and N, assuming some values to start
M = 1; % Example, change according to your analysis
N = round(fs / min(f_desired)); % Example, change as needed

% Time period for one sample
T = 1 / fs;

% Calculate beta values
beta = -0.5 * (omega_desired + N * (2 * pi * fs / M));

% Construct matrix A and vector b for the linear equation system
A = sin(beta);
b = -sin(beta(1));

% Solve for the allpass filter coefficient a using least squares
a = (A' * A) \ (A' * b);

% Calculate the waveguide loop partial frequencies
f_loop = (1/(2*pi*T)) * (2 * atan(-a * sin(omega_desired*T)) + omega_desired*T);

% Calculate the error
e = sum((f_desired - f_loop).^2);

% Document the results
fprintf('Allpass filter coefficient a: %f\n', a);
fprintf('Error e: %f\n', e);

% Plot the frequency difference
freq_diff = f_desired - f_loop;
figure;
stem(freq_diff);
title('Frequency Difference');
xlabel('Partial Number');
ylabel('Frequency Difference (Hz)');

% Plot the frequency differences on the spectrum plot
hold on;
plot(f_desired, zeros(size(f_desired)), 'ro', 'MarkerFaceColor', 'r');
hold off;
legend('|Magnitude|', 'Desired Frequencies');

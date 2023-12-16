% Read data
[signal, fs] = audioread('D4.wav');
% If the signal is stereo, take one channel
if size(signal, 2) == 2
    signal = signal(:, 1); % Take the left channel or right channel
end

% Compute the inverse Fourier Transform of the log magnitude
[cepstrum, quefrency] = fCepstrum(signal, fs, 4096);
% Define resonant frequencies and Q-factor for the filter bank
f_resonances = [98.00, 123.47, 196.00];
nfft = 2048; % Number of FFT points
Q = 1; % Example quality factor

% Initialize the filters
B_total = []; % Numerator coefficients
A_total = []; % Denominator coefficients
H_total = []; % Transfer function models for Bode plot

% Loop over each resonant frequency to create the filter bank
for f_resonant = f_resonances
    % Calculate normalized frequency
    omega = 2 * pi * f_resonant / fs;
    alpha = sin(omega) / (2 * Q);

    % Coefficients for a Bi-Quad resonant filter (band-pass)
    b0 = alpha;
    b1 = 0;
    b2 = -alpha;
    a0 = 1 + alpha;
    a1 = -2 * cos(omega);
    a2 = 1 - alpha;

    % Normalize the filter coefficients
    b = [b0, b1, b2] / a0;
    a = [1, a1, a2] / a0;
    
    % Store the filter coefficients for filtering
    B_total = [B_total; b];
    A_total = [A_total; a];
    
    % Create a transfer function model for Bode plot
    H = tf(b, a, 1/fs);
    H_total = [H_total; H];
end

% Create a single figure for all Bode plots
figure;
hold on; % Hold on to plot multiple graphs in the same figure

% Define a set of colors for the plots
colors = ['b', 'r', 'g']; % blue, red, green

% Plot the Bode plot for each filter in the bank on the same figure
for i = 1:length(H_total)
    bodeplot(H_total(i), colors(i)); % Plot the Bode plot with specified color
    % Note: You might need to adjust the color selection for more filters
end
title('Overlayed Bode Plots for Filter Bank');
grid on; % Add a grid for better visibility
hold off; % Release the figure

% Apply the filter bank to the audio signal
filtered_signal = zeros(size(signal)); % Initialize to zeros
for i = 1:size(B_total, 1)
    filtered_signal = filtered_signal + filter(B_total(i, :), A_total(i, :), signal);
end

% Plot the real part of the cepstrum
time_axis = (0:length(cepstrum)-1) / fs; % Time axis in seconds
figure;
plot(time_axis, real(cepstrum));
title('Real Cepstrum of the Input Signal');
xlabel('Time (s)');
ylabel('Amplitude');

% Plot the original and filtered signal for comparison
figure;
subplot(2, 1, 1);
plot(signal);
title('Original Signal');
xlabel('Sample Number');
ylabel('Amplitude');

subplot(2, 1, 2);
plot(filtered_signal);
title('Filtered Signal');
xlabel('Sample Number');
ylabel('Amplitude');

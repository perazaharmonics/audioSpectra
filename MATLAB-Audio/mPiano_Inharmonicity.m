% This script calculates the harmonic frequencies of a piano string taking into
% account the inharmonicity. It then simulates the phase delay introduced by
% an allpass biquad filter for two notes, A0 and C7, given their fundamental
% frequencies and inharmonicity coefficients. The loss filter is assumed to be
% a scalar and thus does not affect the phase. The phase delay for each harmonic
% is plotted for both notes, and the frequency response of the allpass filter is
% also plotted using freqz.

% Parameters
f0_A0 = 27.5; % Fundamental frequency for note A0
f0_C7 = 2093.0; % Fundamental frequency for note C7
B_A0 = 1e-4; % Inharmonicity coefficient for A0
B_C7 = 1e-3; % Inharmonicity coefficient for C7
fs = 44.1e3; % Sampling frequency

% Calculate harmonics frequencies using the inharmonicity formula
k = 1:25; % Harmonic numbers
f_k_A0 = f0_A0 * sqrt(1 + B_A0 * k.^2);
% Print the inharmonic frequencies for A0 and C7
fprintf('Inharmonic frequencies for A0:\n');
disp(f_k_A0);

f_k_C7 = f0_C7 * sqrt(1 + B_C7 * k.^2);
fprintf('\nInharmonic frequencies for C7:\n');
disp(f_k_C7);

% Convert frequencies to angular frequencies
omega_k_A0 = 2 * pi * f_k_A0;
omega_k_C7 = 2 * pi * f_k_C7;

% Define the allpass filter coefficients (make sure these are defined correctly)
b0 = 0.9; 
b1 = -0.44;  
b2 = 0.6;   
a0 = 1;   
a1 = -b2; % a1 is always the negative of b2
a2 = -b1; % a2 is always the negative of b1  

b = [b0, b1, b2]; % Numerator coefficients
a = [a0, a1, a2]; % Denominator coefficients

% Calculate the frequency response using freqz
[H, freq] = freqz(b, a, 2048, fs); % freqz returns frequency vector and response

% Convert the response to magnitude and phase
magResp = 20 * log10(abs(H)); % Convert magnitude to dB
phaseResp = unwrap(angle(H)); % Unwrap the phase response

% Plot the magnitude response on a logarithmic scale
figure;
subplot(2, 1, 1);
semilogx(freq, magResp, 'LineWidth', 2);
grid on;
title('Magnitude Response of Allpass Biquad Filter');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');

% Plot the phase response on a logarithmic scale
subplot(2, 1, 2);
semilogx(freq, phaseResp, 'LineWidth', 2);
grid on;
title('Phase Response of Allpass Biquad Filter');
xlabel('Frequency (Hz)');
ylabel('Phase (radians)');

% Plotting the inharmonic frequencies for A0
figure;
subplot(1, 2, 1);
stem(k, f_k_A0, 'b', 'filled');
title('Inharmonic Frequencies for A0');
xlabel('Harmonic number');
ylabel('Frequency (Hz)');

% Plotting the inharmonic frequencies for C7
subplot(1, 2, 2);
stem(k, f_k_C7, 'b', 'filled');
title('Inharmonic Frequencies for C7');
xlabel('Harmonic number');
ylabel('Frequency (Hz)');

% Show grid
grid on;

% Set the figure properties
set(gcf, 'Color', 'w');
set(findall(gcf,'-property','FontSize'),'FontSize', 10);

% Ensure the same length for vectors before plotting
assert(length(freq) == length(phaseResp), 'Frequency and phase vectors must be the same length.');


% The first set of plots shows the magnitude and phase response of the allpass
% filter. The second set of plots shows the phase delay across the harmonics for
% notes A0 and C7. These plots can be used to analyze the effect of the allpass
% filter on the phase of each harmonic component of the piano string's vibration.

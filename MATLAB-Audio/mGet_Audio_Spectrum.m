% Read data
[signal, fs] = audioread("D4.wav");
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
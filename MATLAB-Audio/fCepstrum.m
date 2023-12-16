function [cepstrum, quefrency] = fCepstrum(signal, fs, nfft)
    % If nfft is not provided, use the next power of two greater than the signal length
    if nargin < 3
        nfft = 2^nextpow2(length(signal));
    end
    
    % Compute the Fourier Transform of the signal
    fftSignal = fft(signal, nfft);
    
    % Compute the logarithm of the magnitude
    logMagnitude = log(abs(fftSignal) + eps); % Adding eps to avoid log(0)
    
    % Compute the cepstrum by taking the IFT of the log magnitude
    cepstrum = ifft(logMagnitude, nfft);
    
    % Generate a quefrency vector corresponding to the cepstrum values
    quefrency = (0:nfft-1) * (1/fs);
end

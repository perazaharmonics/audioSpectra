% =========================================================================
% MATLAB script for Digital Filter Design
%
% This script designs a digital filter that approximates a specified
% frequency response within the audio frequency band. The design process
% incorporates extrapolation, interpolation, and cepstral analysis to
% achieve an optimal response. The resulting filter is displayed in various
% graphical formats for analysis.
%
% =========================================================================

% -----------------------
% User-defined parameters
% -----------------------
% These values are initial settings for the filter's design specifications.

NZ = 1;         % Number of zeros for the filter.
NP = 4;         % Number of poles for the filter.
NG = 10;        % Count of gain measurements.
fmin = 100;     % Lowest frequency for gain measurement.
fmax = 3000;    % Highest frequency for gain measurement.
fs = 10000;     % Discrete-time filter's sampling rate.
Nfft = 512;     % FFT size for frequency analysis.

% -----------------------
% Derived parameters
% -----------------------
% These parameters are derived from the user-defined values.

df = (fmax / fmin)^(1 / (NG - 1));   % Log-frequency spacing calculation.
f = fmin * df.^(0:NG - 1);           % Frequency axis based on measurements.

% Triangular amplitude response for synthetic gain:
Gdb = 10*[1 : NG/2, NG/2 : -1 : 1] / (NG / 2); % triangular amplitude response

% DC Gain Extrapolation:
dc_amp = Gdb(1) - f(1)*(Gdb(2) - Gdb(1)) / (f(2) - f(1));

% Nyquist Frequency Gain Extrapolation:
Gdb_last_slope = (Gdb(NG) - Gdb(NG - 1)) / (f(NG) - f(NG - 1));
nyq_amp = Gdb(NG) + Gdb_last_slope * (fs / 2 - f(NG));

% Combine extrapolated and measured gains:
Gdbe = [dc_amp, Gdb, nyq_amp];
fe = [0, f, fs / 2];
NGe = NG + 2;

% Cubic spline interpolation for a uniform frequency grid:
Gdbei = spline(fe, Gdbe);
fk = fs*[0:Nfft / 2] / Nfft;       
Gdbfk = ppval(Gdbei, fk);

% Plot the measured vs. interpolated frequency response:
figure(1);
semilogx(fk(2:end-1), Gdbfk(2:end-1), 'k'); grid on;
hold on; semilogx(f, Gdb, 'ok');
xlabel('Frequency (Hz)');   ylabel('Magnitude (dB)');
title('Measured vs. Interpolated Amplitude Response');

% Time aliasing check for the impulse response:
Ns = length(Gdbfk); 
if Ns ~= Nfft / 2+1, error("Confusion in array sizes"); end
Sdb = [Gdbfk, Gdbfk(Ns - 1:-1:2)];    % Consider negative frequencies

S = 10.^(Sdb / 20);                   % Convert to linear magnitude
s = real(ifft(S));                    % Desired impulse response (real)
tlerr = 100*norm(s(round(0.9*Ns:1.1*Ns))) / norm(s);  % Time-limitedness check

% Alert if the outer 20% of impulse response is above 1% of total RMS:
if tlerr > 1.0, error('Increase Nfft and/or smooth Sdb'); end

% Plot the impulse response:
figure(2);
plot(s, '-k'); grid on;
xlabel('Time (samples)');   ylabel('Amplitude');
title('Impulse Response');

% Cepstral analysis for minimum phase:
c = ifft(Sdb);
caliaserr = 100 * norm(c(round(Ns*0.9:Ns*1.1))) / norm(c);

% Alert if aliasing in the cepstrum:
if caliaserr > 1.0, error('Increase Nfft or smooth Sdb'); end

% Fold the cepstrum:
cf = [c(1), c(2:Ns-1) + c(Nfft:-1:Ns+1), c(Ns), zeros(1, Nfft - Ns)];
Cf = fft(cf);
Smp = 10.^(Cf / 20);                   % Minimum phase spectrum

Smpp = Smp(1:Ns);                      % Non-negative frequency portion
wt = 1./(fk+1);                        % typical weigth function for audio
wk = 2*pi*fk / fs;                     % twiddle factor

% Inverse Z-transform:
[B, A] = invfreqz(Smp(1:Ns), wk, NZ, NP, wt);
Hh = freqz(B, A, Ns);   

% Plot magnitude response:
figure(3);
plot(fk, db(Smpp(:)), fk, db(Hh(:)));
grid on;
xlabel('Frequency (Hz)');   
ylabel('Magnitude (dB)');
title('Magnitude Frequency Response');
legend('Desired', 'Filter');


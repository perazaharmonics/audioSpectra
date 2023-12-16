Nx = 1024; %length of input signal
Nh = 128; % FIR filter length
A 1; B = ones(1, nH);   % FIR "running-sum" filter
n = 0: Nx - 1;
x = sin(n*2*pi*(Nx / Nh - 1) / Nx); %input singla- zero-pad it:
zp = zeros(1 , Nx / 2); xzp = [zp, x, zp]; nzp=[0:length(xzp) -1];
y = filter(B, A, xzp);      % filter output signal
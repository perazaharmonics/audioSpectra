Nx = 1024; %length of input signal
Nh = 300; % IIR filter length
A = [ 1 -0.99]; B = 1; % One-pole recursive filter

n = 0: Nx - 1;
x = sin(n*2*pi*(Nx / Nh - 1) / Nx); %input singla- zero-pad it:
zp = zeros(1 , Nx / 2); xzp = [zp, x, zp]; nzp=[0:length(xzp) -1];
y = filter(B, A, xzp);      % filter output signal
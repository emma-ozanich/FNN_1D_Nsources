fpath = 'data/incoh/';
fname = [num2str(snr) 'dB/'];
Nsource = 3;

% % array parameters % %
theta = 0:2:359;%10:0.1:30;
lambda = 1500/79;
Nsensors = 27;
xq = textread('data/positions_hlanorth.txt');
xq = [xq(:,3), xq(:,2)];
disp(size(xq))
u = sind(theta);
v = cosd(theta);
beam = exp(1j*2*pi/lambda * (xq(:,1)*u + xq(:,2)*v) )/ sqrt(Nsensors); %2D BF

% % SBL parameters % %
options.convergence.error   = 10^(-4);
options.convergence.delay   = 200;

% maximum number of iterations
% if flag == 1, code issues a warning if not convereged to
% error specified by report.convergence
options.convergence.maxiter = 1000;

% solution only accepted after this iteration
options.convergence.min_iteration = 15; 

% status report every xx iterations
options.status_report = 150;

% noise power initialization guess
options.noisepower.guess = 0.1;

% fixed point acceleration [0.5 2]
options.fixedpoint = 2;

% number of sources (required for noise variance estimate)
options.Nsource = Nsource;

% flag for issuing warnings
options.flag = 0;
options.tic = 1;

fpath = 'data/incoh/';
fname = [num2str(snr) 'dB/'];
Nsource = 5;

% % array parameters % %
theta = -90:1:89;%10:0.1:30;
lambda = 1500/200;
Nsensors = 20;
d = 2.5;%3.75;
q = 0:(Nsensors-1);
xq = (q - (Nsensors-1)/2).*d;
if size(xq,1)<size(xq,2)
    xq = xq.';
end
beam = exp(1i*2*pi/lambda*xq*sind(theta))/ sqrt(Nsensors);


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

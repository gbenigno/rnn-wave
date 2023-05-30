
% A RECURRENT NEURAL NETWORK INTERPRETATION OF THE SCALAR WAVE EQUATION
% 
% GABRIEL B. BENIGNO, 2023
% 
% This code draws heavily from the tutorial by Ingo Berg
% (https://beltoforion.de/en/recreational_mathematics/2d-wave-equation.php),
% and was also influenced by the following paper:
% 
% Hughes, Williamson, Monkov, Fan. (2019) Wave physics as an analog
% recurrent neural network. Science Advances. 

%% clear workspace and command window
clearvars;clc

%% init
dx = 1; % space step
dt = 1; % time step
Nr = 300; % number of rows in network
Nc = 300; % number of columns in network
N = Nr*Nc; % totral number of neurons
c = 0.5; % wave speed
a_t = zeros(N,1); % a at time t
a_tm1 = a_t; % a at time t-1
T = 300; % number of time steps
f = zeros(Nr,Nc,T); % external input
f(48:52,48:52,1) = 120;
f = reshape(f,N,T);
A = nan(Nr,Nc,T); % stack of images

%% Laplacian matrix
v = sparse(1,Nr);
v([2 end]) = 1;
offdi = toeplitz([v(1) fliplr(v(2:end))], v);
I = speye(Nr);
II = speye(N);
L = kron(offdi,I) + kron(I,offdi) - 4*II;
M = 2*II + (dt*c/dx)^2 * L;

%% simulation
for tt = 1 : T
    tmp = a_t;
    a_t = M*a_t - a_tm1 + f(:,tt);
    a_tm1 = tmp;
    A(:,:,tt) = reshape(a_t,Nr,Nc);
end

%% visualization
for ii = 1 : 6
    tt = 40*(ii-1)+1;
    subplot(1,6,ii)
    imagesc(A(:,:,tt))
    clim([0 60])
    pbaspect([1 1 1])
    % colorbar
    title(sprintf('A[t=%u]',tt))
    xlabel('x')
    ylabel('y')
end

% figure
% sliceViewer(A); % requires matlab's image processing toolbox
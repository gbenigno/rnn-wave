# A RECURRENT NEURAL NETWORK INTERPRETATION OF THE SCALAR WAVE EQUATION
# 
# GABRIEL B. BENIGNO, 2023
# 
# This code draws heavily from the tutorial by Ingo Berg
# (https://beltoforion.de/en/recreational_mathematics/2d-wave-equation.php),
# and was also influenced by the following paper:
# 
# Hughes, Williamson, Monkov, Fan. (2019) Wave physics as an analog
# recurrent neural network. Science Advances. 

#%%
import numpy as np
import scipy.linalg as la
from scipy import sparse
import matplotlib.pyplot as plt

#%% init
dx = 1 # space step
dt = 1 # time step
Nr = 300 # number of rows in network
Nc = 300 # number of columns in network
N = Nr*Nc # totral number of neurons
c = 0.5 # wave speed
a_t = np.zeros(N) # a at time t
a_tm1 = a_t # a at time t-1
T = 300 # number of time steps
f = np.zeros((Nr,Nc,T)); # external input
f[48:52,48:52,0] = 120;
f = np.resize(f,(N,T));
A = np.empty((Nr,Nc,T)); # stack of images # PS sorry for matlab indexing

#%% Laplacian matrix
v = np.zeros((1,Nr))
v[0, 1] = 1
v[0, -1] = 1
offdi = la.circulant(v)
offdi = sparse.lil_array(offdi)
I = sparse.eye(Nr)
II = sparse.eye(N)
L = sparse.kron(offdi,I) + sparse.kron(I,offdi) - 4*II
M = 2*II + (dt*c/dx)**2*L

#%% simulation
for tt in np.arange(T):
    tmp = a_t
    a_t = M*a_t - a_tm1 + f[:,tt]
    a_tm1 = tmp
    tmp2 = np.resize(a_t,(Nr,Nc))
    A[:,:,tt] = tmp2
    
#%% visualize
for ii in np.arange(6):
    tt = 40*ii + 1
    plt.subplot(1,6,ii+1)
    plt.imshow(A[:,:,tt])
    plt.title(f"A[t={tt}]")
    plt.xlabel('x')
    plt.ylabel('y')
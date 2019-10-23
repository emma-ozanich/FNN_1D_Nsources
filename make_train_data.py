import numpy as np
import os
from config import params
import scipy.io as sio
from math import pi
import numpy.random as r

# # define parameters
snap = 1
s = np.arange(1000,1001000)# set random seed
snr = 100
maxs = 10
fname = 'train'#_swellex96'

nsamples = len(s) # number of samples
Nfeat = params.Nsensor*(params.Nsensor+1) # number of features
    
    
def form_features(csdm):
    
    N = csdm.shape[1]
    iu1 = np.triu_indices(N)
    csdm_vec = csdm[iu1]
    feats = np.concatenate( (np.real(csdm_vec), np.imag(csdm_vec)),axis=-1 )
    
    return feats

def generate_signal(theta, Amp, A, snr, snap, params):
    Xsource = np.zeros((snap,params.nclasses))
    Q = np.zeros((params.Nsources,), dtype=int)
        
    # Sparse indices for N sources
    for n in np.arange(params.Nsources):
        Q[n] = np.argmin(np.abs(params.theta - theta[n]))# index for source n
    Xsource[:,Q] = np.matlib.repmat(Amp, snap, 1)
    x = Xsource
    
    # random source phase
    if not params.Coh: 
        phi = np.exp(-1j*2*pi*r.rand(snap, params.nclasses)-pi)
        Xsource = Xsource*phi/np.abs(phi)
        
    y = np.matmul(Xsource, A.T).T
    y = y/np.linalg.norm(y, axis=0)
    if not (snr==100):
        rnl = 10**(-snr/20)/np.sqrt(2)/np.sqrt(Nsensor) #note: sqrt(0.5) ensures the random variance is 1
        y = y + rnl*np.squeeze(r.randn(Nsensor, snap, 2).view(np.complex128)) 
        
    return y, x


# preallocate
x_train = np.zeros((nsamples, Nfeat))
y_train = np.zeros((nsamples, params.nclasses))

## Make A matrix (plane wave weights)
Nsensor = len(params.xq)
u = np.sin(np.deg2rad(params.theta))
v = np.cos(np.deg2rad(params.theta))
A = np.exp(1j*2*pi/params.l*(np.outer(params.xq[:,0],u) + np.outer(params.xq[:,1],v)))/np.sqrt(Nsensor)#))/np.sqrt(Nsensor)
A = A/np.linalg.norm(A, axis=0)

# # loop over random samples # #
rs = 0 # counter if using multiple seeds
for si in s: # for all random sets
    r.seed( si )
    
    # choose random num. of sources 
    source_ind = np.arange(maxs)+1
    params.Nsources = source_ind[np.random.randint(maxs, size=1)][0] # pick random num. source between 1&10
        
    # choose random angles 
    I = np.random.randint(len(params.theta), size=params.Nsources) # draw from theta UNIFORM RANDOM
    theta = params.theta[I]
    Amp = np.ones(theta.shape) # amplitudes
    y, Xsource = generate_signal(theta, Amp, A, snr, snap, params)
    csdm = np.inner(y[:,0:snap], y[:,0:snap].conj())/snap

    x_train[rs,:] = form_features(csdm)
    y_train[rs,:] = np.squeeze(Xsource[0,:])
    rs = rs+1
       
# save
if not os.path.isdir(params.fpath + 'train/'):
    if not os.path.isdir('data/'):
        os.mkdir('data/')
    if not os.path.isdir(params.fpath):
        os.mkdir(params.fpath)
    os.mkdir(params.fpath + 'train/')
np.save(params.fpath + 'train/x_' + fname, x_train)#
np.save(params.fpath + 'train/y_' + fname, y_train)#

print(x_train.shape, y_train.shape)
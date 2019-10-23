import numpy as np
from numpy.random import seed as randseed
import os
from config import params
import scipy.io as sio
from math import pi
  
s = 20001 # random seed
snr = 100
params.fname = str(snr) + 'dB/' # redo fname for SNR
snap = 10
nsamples = len(params.theta)
Nfeat = params.Nsensor*(params.Nsensor+1)#2*params.Nsensor**2#
seed = np.arange((nsamples))
filename = '_5track_' + str(snap) + 'snaps'

# Define source angles
params.Nsources = 5
Amp = [1,1,1,1,1]
theta = np.zeros((params.Nsources, len(params.theta)))
theta[0,:] = params.theta[np.arange(0, len(params.theta))]
theta[1,:] = params.theta[np.arange(len(params.theta)-1,-1, -1)]
theta[2,:] = params.theta[22]*np.ones((len(params.theta),))
theta[3,:] = params.theta[60]*np.ones((len(params.theta),))
theta[4,:] = params.theta[170]*np.ones((len(params.theta),))

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
        phi = np.exp(-1j*2*pi*np.random.rand(snap, params.nclasses)-pi)
        Xsource = Xsource*phi/np.abs(phi)
    y = np.matmul(Xsource, A.T).T
    y = y/np.linalg.norm(y, axis=0)
    if not (snr==100):
        rnl = 10**(-snr/20)/np.sqrt(2)/np.sqrt(Nsensor) #note: sqrt(0.5) ensures the random variance is 1
        y = y + rnl*np.squeeze(np.random.randn(Nsensor, snap, 2).view(np.complex128)) 
        
    return y, x

def form_features(csdm):
    
    N = csdm.shape[1]
    iu1 = np.triu_indices(N)
    csdm_vec = csdm[iu1]
    feats = np.concatenate( (np.real(csdm_vec), np.imag(csdm_vec)),axis=-1 )
    
    return feats


## Make A matrix (plane wave weights)
Nsensor = len(params.xq)
u = np.sin(np.deg2rad(params.theta))
v = np.cos(np.deg2rad(params.theta))
A = np.exp(1j*2*pi/params.l*(np.outer(params.xq[:,0],u) + np.outer(params.xq[:,1],v)))/np.sqrt(Nsensor)#))/np.sqrt(Nsensor)
A = A/np.linalg.norm(A, axis=0)

print('Starting angles: ' ,theta[0,0], theta[1,0], theta[2,0], theta[3,0], theta[4,0])

## make data
TEST = np.zeros((params.Nsensor, snap, nsamples), dtype=complex)
csdm = np.zeros((nsamples, params.Nsensor, params.Nsensor),dtype=complex)
x_test = np.zeros((nsamples, Nfeat))
y_test = np.zeros((nsamples, params.nclasses))

for i_theta in np.arange(nsamples):
        randseed( seed[i_theta] )
            
        y, Xsource = generate_signal(theta[:,i_theta], Amp, A, snr, snap, params)
        TEST[:,:,i_theta] = y
        csdm[i_theta,:,:] = np.inner(y[:,0:snap], y[:,0:snap].conj())/snap
        x_test[i_theta,:] = form_features(csdm[i_theta,:,:])
        y_test[i_theta,:] = np.squeeze(Xsource[0,:])
labels = theta.T

    
## Save
if not os.path.isdir(params.fpath + 'test/' + params.fname):
    if not os.path.isdir('data/'):
        os.mkdir('data/')
    if not os.path.isdir(params.fpath):
        os.mkdir(params.fpath)
    if not os.path.isdir(params.fpath + 'test/'):
        os.mkdir(params.fpath + 'test/')
    os.mkdir(params.fpath + 'test/' + params.fname)
    
## FOR FNN # #
np.save(params.fpath + 'test/' + params.fname + 'xtest' + filename, x_test)
np.save(params.fpath + 'test/' + params.fname + 'ytest' + filename, y_test)
print(params.fpath + 'test/' + params.fname + 'xtest' + filename)

## FOR CBF # #
np.save(params.fpath + 'test/' + params.fname + 'raw' + filename, TEST)
np.save(params.fpath + 'test/' + params.fname + 'labels' + filename, labels)

## FOR SBL # #
data = dict();
data['TEST'] = TEST
data['labels'] = labels
sio.savemat(params.fpath +  'test/' + params.fname + filename[1:], mdict={'data':data})

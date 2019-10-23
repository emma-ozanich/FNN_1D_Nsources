
# coding: utf-8

# In[5]:

# # configuration file # #
import numpy as np
import numpy.matlib
class structtype():
    pass

params = structtype()

# # for SwellEx96 # #
params.Nsensor = 27
hla_pos = np.loadtxt('data/positions_hlanorth.txt')
params.xq = np.stack((hla_pos[:,2], hla_pos[:,1]),axis=1) # x and y reversed
params.theta = np.arange(0, 360,1)
params.nclasses = 360
f = 79
c = 1500
params.l = c/f # wavelength (m)

# processor parameters #
SNR = 100
params.nsnaps = 5
params.Coh = False
params.theta = np.arange(0,360,2) # #!

# signal parameters #
params.Nsources = 2

# FNN parameters #
params.GPU_frac = 0.7
params.nhidden = 512
params.nclasses = 180
params.batch_size = int(180*89.5)
params.nepochs = 100
params.output = 'softmax'
params.loss = 'categorical_crossentropy'

# set filepaths #
if params.Coh:
    params.fpath = 'data/coh/'
else:
    params.fpath = 'data/incoh/'
params.fname = str(SNR) + 'dB/'




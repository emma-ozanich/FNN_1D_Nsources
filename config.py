
# coding: utf-8

# In[5]:

# # configuration file # #
import numpy as np
import numpy.matlib
class structtype():
    pass

params = structtype()
# array parameters #
c = 1500
f = 200
d = 2.5#3.75
params.Nsensor = 20
q = np.arange(params.Nsensor)
params.xq = np.stack( ( (q - (params.Nsensor-1)/2)*d, np.zeros((params.Nsensor,)) ), axis=1)#(q - (params.Nsensor-1)/2)*d#
params.l = c/f # wavelength (m)

# processor parameters #
SNR = 100
params.nsnaps = 5
params.Coh = False
params.theta = np.arange(-90,90,1)#0.18)

# signal parameters #
params.Nsources = 5

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



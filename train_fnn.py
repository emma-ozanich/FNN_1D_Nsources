'''FNN-based CBF
   Haiqiang Niu 08/08/2018
   ( modified 03/13/2019 E. Ozanich)
'''
from numpy.random import seed, randn
seed(1)
import tensorflow as tf
tf.set_random_seed(1)
import time
import numpy as np
import math
import os

from keras.layers import Dense, GaussianNoise, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam,SGD,RMSprop
import keras.backend as K
from config import params
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.regularizers import l1, l2
# instantiate regularizer
reg = l2(0.001)

# # set GPU # #
#os.environ["CUDA_DEVICE_ORDER"] = "00000000:03:00.0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=params.GPU_frac) # fix memory usage
tf.GPUOptions.allow_growth = True

# don't use early stopping
#callback = EarlyStopping(monitor='val_loss',
#                  min_delta=0.01,
#                  patience=20, # default: 2, number of increasing steps before stopping
#                  verbose=0, mode='min')

# learning rate scheduler
def step_decay(epoch):
    init=1e-4
    drop = 0.5
    epoch_drop = 30
    lrate = init*math.pow(drop, math.floor((1+epoch)/epoch_drop))
    return lrate

lrate = LearningRateScheduler(step_decay)

# # Set Training file ---------------------------------
fname = 'train'

nlayers = 5
params.nhidden = 512

if params.Coh:
    coh = 'coh'
else:
    coh = 'incoh'
    
modelname = 'FNN_models/%02ddeep_%04dhid_%s_%s_%03dbatch_%s'% (nlayers, params.nhidden, params.output,                                                          params.loss, params.batch_size, coh)
print(modelname)

#time.sleep(4600) # to delay, if multiple tasks

#-----------------------------------------------------------------
# # load data # #
#  training data
# two sources
x_train = np.load(params.fpath + 'train/x_' + fname + '.npy')
y_train = np.load(params.fpath + 'train/y_' + fname + '.npy')


print(x_train.shape)

#  validation data
params.fname = str(0) + 'dB/'
x_val = x_train
y_val = y_train

#-----------------------------------------------------------------
# # start TF session # #
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) # start a Tensorflow session using GPU
tf.set_random_seed(1)
input_size = x_train.shape[1]

# # create model # #
model = Sequential()
#model.add(GaussianNoise(0,input_shape=(input_size,)))
model.add(Dense(params.nhidden, activation='relu', input_shape=(input_size,)))
#model.add(GaussianNoise(0.1))
#model.add(Activation(activation='relu'))#, kernel_regularizer=l2(0.02)))

model.add(Dense(params.nhidden, activation='relu'))
model.add(Dense(params.nhidden, activation='relu'))
model.add(Dense(params.nhidden, activation='relu'))
layer7 = model.add(Dense(params.nhidden, activation='relu')) # visualize this layer

#model.add(Dropout(0.5))

model.add(Dense(params.nclasses, activation=params.output))#, activity_regularizer=l2(0.001)))
model.compile(loss=params.loss,
              optimizer=Adam(0.001))#0.1)) 
model.summary()

#-----------------------------------------------------------------
# # train model # #
t0 = time.time()
print(modelname)
history = model.fit(x_train, y_train,
                    batch_size=params.batch_size,
                    epochs=params.nepochs,
                   # validation_data=(x_val,y_val),
                    verbose=1, shuffle=False)#, callbacks = [callback]) 
print('%1d epochs trained in %.2f seconds.' % (params.nepochs, (time.time() - t0)) )

#-----------------------------------------------------------------
# # save model and training loss # #
history = history.history
model.save(modelname + '.h5') 
np.savez(modelname + '_history', loss=history['loss'])#, val_loss = history['val_loss'])


# Training linear oscillator NN model
# Alessandro Antonucci @AlexRookie
# University of Trento

#=====================================================================================================================#
#
# Initialization
#
#=====================================================================================================================#

# Parameters

data_folder = './data/oscillator-linear/'
n_files = 100  # number of files (per sets)
#max_len = 1000 # maximum length per file

train_p = 70   # training samples percentage (test_p = 100-train_p)
observe = 30   # observation window length
predict = 1    # prediction window length
window  = 70   # window length

epochs = 200   # number of epochs
batch  = 128   # batch size
l_rate = 0.001 # learning rate
#k_reg  = 1e-3  #kernel_regularizer

options_weights = True # show weights
options_save    = True # export trained model

model_folder = './models/'
model_name  = 'model-linear-oscillator-rnn'

#=====================================================================================================================#

import os.path
import numpy as np
from numpy import array
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import Tensorflow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Dense, Add, Lambda, RNN
#from tensorflow.keras.initializers import Constant
#from tensorflow.keras.constraints import Constraint, NonNeg

# Clear session
K.clear_session()

#=====================================================================================================================#
#
# Prepare data
#
#=====================================================================================================================#

print('Import data....')

# Read param file
paramsfile = open(data_folder + 'params.txt', 'r')
params = paramsfile.readlines()
paramsfile.close()

# Randomize data files
files = np.arange(n_files)
np.random.shuffle(files)

#num_samples = []
#for file in files:
#    splitline = params[file].split(";")
#    num_samples.append(int(splitline[11]))

pbar = tqdm(total=n_files)
xIn, vIn, xOut, vOut, u = [], [], [], [], []
#m, k, c, x0, v0 = [], [], [], [], []

for file in files:
    # Read data file
    t_file, x_file, v_file, u_file = [], [], [], []
    all_lines = open(data_folder + 'data/' + str(file+1) + '.txt', 'r')
    lines = all_lines.readlines()#[1:] # skip first line to avoid NaNs
    for l in range(0, len(lines)):
        splitline = lines[l].rstrip("\n").split(";")
        t_file.append(float(splitline[0]))
        x_file.append(float(splitline[1]))
        v_file.append(float(splitline[2]))
        u_file.append(float(splitline[3]))
    # Read param file
    #splitline_p = params[file].rstrip("\n").split(";")

    # Normalize x and y
    #x_file = [x - x_file[0] for x in x_file]
    #y_file = [x - y_file[0] for x in y_file]

    # Collect data samples
    if len(t_file) < (window+observe+predict-1):
        continue
    for i in range(0, len(t_file)-(window+observe+predict-1)):
        #if i >= max_len:
        #    break
        j_xIn, j_vIn, j_xOut, j_vOut, j_u = [], [], [], [], []
        #j_m, j_k, j_c = [], [], []
        for j in range(i, i+window):
            j_xIn.append(x_file[j:j+observe])
            j_vIn.append(v_file[j:j+observe])
            j_xOut.append(x_file[j+observe])
            j_vOut.append(v_file[j+observe])
            j_u.append(u_file[j+observe-1])
            #j_m.append(float(splitline_p[2]))
            #j_k.append(float(splitline_p[3]))
            #j_c.append(float(splitline_p[4]))
        xIn.append(j_xIn)
        vIn.append(j_vIn)
        xOut.append(j_xOut)
        vOut.append(j_vOut)
        u.append(j_u)
        #m.append(j_m)
        #k.append(j_k)
        #c.append(j_c)

    pbar.update(1)

pbar.close()

print('Convert data....')
xIn  = np.asarray(xIn)
vIn  = np.asarray(vIn)
xOut = np.asarray(xOut)
vOut = np.asarray(vOut)
u    = np.asarray(u)

# Divide train and valid samples
num_of_samples = len(xIn)
train = round(train_p*num_of_samples/100)
valid = num_of_samples-train

if train < batch or valid < batch:
    batch = 1
else:
    # Samples must be multiplier of batch
    train = int(train/batch) * batch
    valid = num_of_samples-train
    valid = int(valid/batch) * batch

xIn_train  = xIn[0:train,:]
vIn_train  = vIn[0:train,:]
xOut_train = xOut[0:train]
vOut_train = vOut[0:train]
u_train    = u[0:train]

xIn_tvalid  = xIn[train:train+valid,:]
vIn_valid  = vIn[train:train+valid,:]
xOut_valid = xOut[train:train+valid]
vOut_valid = vOut[train:train+valid]
u_valid    = u[train:train+valid]

del xIn, vIn, xOut, vOut, u #, m, k, c, x0, v0
del t_file, x_file, v_file, u_file
del j_xIn, j_vIn, j_xOut, j_vOut, j_u #, j_m, j_k, j_c

print('Samples: {:d}/{:d} ({:d} training + {:d} validation)'.format(train+valid, num_of_samples, train, valid))
print('Batch size: {:d}'.format(batch))

#=====================================================================================================================#
#
# Training RNN model
#
#=====================================================================================================================#

# Custom methods

def rmse(y_true, y_pred):
    # Root mean squared error (rmse) for regression
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# Custom classes

class RNNCell(Layer):
    def __init__(self, output_size, input_size, observe, **kwargs):
        super(RNNCell, self).__init__(**kwargs)
        self.output_size = output_size
        self.input_size = input_size
        self.state_size = [tf.TensorShape([output_size]), tf.TensorShape([observe])]
        self.observe = observe

    def build(self, input_shape):
        # Define constants and layer weights
        self.W_x = self.add_weight(name='dense_x', shape=(self.observe, 1), trainable=True) # activation: none #regularizer=tf.keras.regularizers.l2(k_reg)
        self.W_f = self.add_weight(name='dense_u', shape=(1, 1), trainable=True) # activation: none
        #self.dt = 0.05

        #self.kernel = self.add_weight(name='input_matrix',
        #                              shape=(self.input_size, self.output_size),
        #                              initializer=Constant([[0, 1]]),
        #                              trainable=False)
        #self.recurrent_kernel = self.add_weight(name='state_matrix',
        #                                        shape=(self.output_size, self.output_size),
        #                                        initializer=Constant([[1,      1],# -self.k*self.dt/self.m],
        #                                                              [self.dt,1]]),# 1-self.c*self.dt/self.m]]),
        #                                        trainable=False)
        self.built = True

    def call(self, inputs, states):
        # Input: [input_at_t, states_at_t]
        prev_output, x_set = states
        #print('x_set '+str(K.int_shape(x_set)))

        #shift_x = x_set - x_set[:,:1]
        dense_x = K.dot(x_set, self.W_x)
        dense_u = K.dot(inputs, self.W_f)

        output = dense_x + dense_u
        
        # Dynamical model
        #h = K.dot(inputs, self.kernel)
        #output = h + K.dot(prev_output, self.recurrent_kernel) # [x_, v_,]
 
        x_new_set = tf.concat([x_set[:,1:self.observe], output], axis=1) 

        # Output: [output_at_t, states_at_t_plus_1]
        return output, (output, x_new_set)

    def get_config(self):
        config = {'output_size': self.output_size,
                  'input_size': self.input_size,
                  'state_size': self.state_size,
                  'observe': self.observe}
        base_config = super(RNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#=====================================================================================================================#

# Define model

input_x = Input(shape=(window, observe), batch_size=None, name='input_x')
input_u = Input(shape=(window, 1), batch_size=None, name='input_u')

# initial state: [x_n, [x_set]]; shape: (batch, state_size)
init_state_rnn = Lambda(lambda x: [x[:,0,-1:], x[:,0,:]], name='init_state_rnn')(input_x)

out_x = RNN(RNNCell(output_size=1, input_size=1, observe=observe), return_sequences=True, stateful=False, unroll=True, name='rnn')(input_u, initial_state=init_state_rnn)

#[out_x, out_v] = Lambda(lambda tensors: tf.split(tensors, num_or_size_splits=2, axis=2), name='out')(out_rnn)

model_rnn = Model(inputs=[input_x, input_u], outputs=[out_x])

# Print model
print(model_rnn.summary())

#=====================================================================================================================#

# Check the trainable status of the individual layers
#for layer in model_noise.layers:
#    print(layer, layer.trainable)

# Configure model for training
opt = optimizers.Adam(learning_rate=l_rate) #optimizers.Adam(learning_rate=l_rate) #optimizers.RMSprop(learning_rate=lrate, rho=0.4)
model_rnn.compile(optimizer=opt,
                  loss='mean_squared_error',
                  metrics=[rmse])

# Train model
print('[Fitting rnn]')
fit = model_rnn.fit([xIn_train, u_train],
                    [xOut_train],
                    epochs=epochs, batch_size=batch, verbose=1)

# Get learned weights
if options_weights:
    print('[Weights:]')
    weights = model_rnn.get_weights()
    names = [weight.name for layer in model_rnn.layers for weight in layer.weights]
    for name, weight in zip(names, weights):
        print(name, weight, weight.shape)

#=====================================================================================================================#

# Prediction on validation samples

print('[Prediction]')
out = model_rnn.predict([xIn_valid, u_valid]) #, callbacks=[NoiseCallback()])

xOut_hat = np.squeeze(out)

# Scores
RMSE = np.sqrt(np.mean(np.square(xOut_hat.flatten() - xOut_valid.flatten())))
print(RMSE)

# Plot
fig = plt.figure(1)
#plt.subplot(2,1,1)
plt.plot(xOut_valid[1000:5000].flatten(), c='b', label='xOut valid')
plt.plot(xOut_hat[1000:5000].flatten(), c='r', label='xOut predicted')
plt.legend()
#plt.subplot(2,1,2)
#plt.plot(vOut_test[1000:5000].flatten(), c='b', label='vOut test')
#plt.plot(vOut_hat[1000:5000].flatten(), c='r', label='vOut predicted')
#plt.legend()
fig.tight_layout()

#plt.draw() # non-blocking plot
#plt.pause(0.1)

plt.show() # blocking plot

#=====================================================================================================================#
#
# Create single-shot model
#
#=====================================================================================================================#

# Define model

input_x = Input(shape=(observe, ), batch_size=None, name='input_x')
input_u = Input(shape=(1, ), batch_size=None, name='input_u')

dense_x = Dense(units=1, activation=None, use_bias=None, name='dense_x')(input_x) # weights=[np.ones([window,1])], regularizer=tf.keras.regularizers.l2(k_reg)

dense_u = Dense(units=1, activation=None, use_bias=None, name='dense_u')(input_u)

out_x = Add(name='add')([dense_x, dense_u])

model = Model(inputs=[input_x, input_u], outputs=[out_x])

# Print model
print(model.summary())

#=====================================================================================================================#

# Copy weights
model.set_weights(model_rnn.get_weights())

# Get weights
if options_weights:
    print('[Weights:]')
    weights = model.get_weights()
    names = [weight.name for layer in model.layers for weight in layer.weights]
    for name, weight in zip(names, weights):
        print(name, weight, weight.shape)

# Export model
if options_save:
    model.save(model_folder + model_name + '.h5')
    #model_rnn_check.save(model_folder + model_name + '_rnn_check.h5')

#=====================================================================================================================#

"""
# Prediction open loop

print('[Prediction open loop]')

n_test = 30s
random_file = np.random.choice(test, n_test, replace=False)

# Ground truth
x_gt, v_gt = np.empty(shape=(n_test,window)), np.empty(shape=(n_test,window))
for j in range(0, n_test):
    for i in range(0, window):
        x_gt[j,i] = xOut_test[random_file[j],i]
        v_gt[j,i] = vOut_test[random_file[j],i]

# Predict
x_hat, v_hat = np.empty(shape=(n_test,window)), np.empty(shape=(n_test,window))
for j in range(0, n_test):
    # Clear stuff
    K.clear_session()

    x, u = [], []
    x_ = np.expand_dims(xIn_test[random_file[j],0,:], axis=0)
    y_ = np.expand_dims(yIn_test[random_file[j],0,:], axis=0)
    print(j, random_file[j])

    for i in range(0, window):
        # Predict
        [x, y, vx, vy] = model([x_, y_], training=False)
        # New input
        x_ = np.expand_dims(np.append(x_[:,1:], x), axis=0)
        y_ = np.expand_dims(np.append(y_[:,1:], y), axis=0)
        # Predictions
        x_hat[j,i] = x
        y_hat[j,i] = y
        vx_hat[j,i] = vx
        vy_hat[j,i] = vy
    out1 = model.predict([xIn_test, yIn_test]) #, callbacks=[NoiseCallback()])
    out2 = model_check.predict([xIn_test, yIn_test]) #, callbacks=[NoiseCallback()])
"""

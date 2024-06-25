import logging
import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *

x = Input('x', dimensions=3)
y = Input('y', dimensions=3)
w = Input('w')
k = Input('k')

lin = Linear(output_dimension=1)
fir = Fir(output_dimension=1)
out = Output('out', Fir(lin(x.tw(0.05) + y.tw(0.05))))
out2 = Output('out2', fir(w.last() + k.last()))

test = Neu4mes()
test.minimizeError('out', out, out2)
test.neuralizeModel(0.01)

## Custom dataset
data_x = np.zeros(shape=(100,3), dtype=np.float32)
data_y = np.ones(shape=(100,3), dtype=np.float32)
data_w = np.zeros(shape=(100,1), dtype=np.float32)
data_k = np.ones(shape=(100,1), dtype=np.float32)
dataset = {'x':data_x, 'y':data_y, 'w':data_w, 'k':data_k}

test.loadData(source=dataset)
print('x.shape: ',test.data['x'].shape)
print('x first sample: ',test.data['x'][0])
print('y.shape: ',test.data['y'].shape)
print('x first sample: ',test.data['y'][0])
print('w.shape: ',test.data['w'].shape)
print('x first sample: ',test.data['w'][0])
print('k.shape: ',test.data['k'].shape)
print('x first sample: ',test.data['k'][0])

## Load from file
data_folder = os.path.join(os.path.dirname(__file__), 'data/')
data_struct = ['x','y','w','k']
test.loadData(source=data_folder, format=data_struct, skiplines=0, delimiter=',')
print('x.shape: ',test.data['x'].shape)
print('x first sample: ',test.data['x'][0])
print('y.shape: ',test.data['y'].shape)
print('x first sample: ',test.data['y'][0])
print('w.shape: ',test.data['w'].shape)
print('x first sample: ',test.data['w'][0])
print('k.shape: ',test.data['k'].shape)
print('x first sample: ',test.data['k'][0])

## Try to train the model
test.trainModel(test_percentage=20, training_params={'num_of_epochs':100, 'train_batch_size':4, 'test_batch_size':4})
import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
from neu4mes.visualizer import MPLVisulizer
import torch

# Input of the function
x = Input('x')
# Output the function
target_y = Input('target_y')

# Create the target functions
data_x = np.random.rand(250)*10-5
data_a = 2
data_b = -3
dataset = {'x': data_x, 'target_y': (data_a*data_x) + data_b}


print('EXAMPLE 1')
## In this example we create a triangular function with 11 centers 
fuz = Fuzzify(11,centers=[-5,-4,-3,-2,-1,0,1,2,3,4,5],functions='Triangular')
out = Output('out', fuz(x.last()))

test = Neu4mes(visualizer=None)
test.addModel(out)
test.minimizeError('error', target_y.last(), out, 'mse')
test.neuralizeModel()
test.loadData(name='fuzzy_dataset', source=dataset)

random_sample = test.get_random_samples(dataset='fuzzy_dataset',window=1)
print('X: ',random_sample['x'])
results = test(random_sample, sampled=True)
print('activation function: ', results['out'])


print('\nEXAMPLE 2')
## In this example we create a rectangular function with 11 centers 
fuz = Fuzzify(11,centers=[-5,-4,-3,-2,-1,0,1,2,3,4,5],functions='Rectangular')
out = Output('out', fuz(x.last()))

test = Neu4mes(visualizer=None)
test.minimizeError('x', target_y.last(), out, 'mse')
test.neuralizeModel()
test.loadData(name='fuzzy_dataset', source=dataset)

random_sample = test.get_random_samples(dataset='fuzzy_dataset', window=1)
print('X: ',random_sample['x'])
results = test(random_sample, sampled=True)
print('activation function: ', results['out'])

print('\nEXAMPLE 3')
## In this example we define a custom function to use as activation function with 3 centers 
def fun(x):
    return torch.cos(x)
fuz = Fuzzify(3,centers=[-4,0,4],functions=fun)
out = Output('out', fuz(x.last()))

test = Neu4mes(visualizer=None)
test.minimizeError('x', target_y.last(), out, 'mse')
test.neuralizeModel()
test.loadData(name='fuzzy_dataset', source=dataset)

random_sample = test.get_random_samples(dataset='fuzzy_dataset', window=2)
print('X: ',random_sample['x'])
results = test(random_sample, sampled=True)
print('activation function: ', results['out'])

print('\nEXAMPLE 4')
## In this example we create two custom functions with 4 centers , the first and third center will use the first activation function
## while the second and forth center will use the second activation function
def fun1(x):
    return torch.cos(x)
def fun2(x):
    return torch.sin(x)
fuz = Fuzzify(4,centers=[-9.0,-3.0,3.0,9.0],functions=[fun1,fun2])
out = Output('out', fuz(x.last()))

test = Neu4mes(visualizer=None)
test.minimizeError('x', target_y.last(), out, 'mse')
test.neuralizeModel()
test.loadData(name='fuzzy_dataset', source=dataset)

random_sample = test.get_random_samples(dataset='fuzzy_dataset', window=2)
print('X: ',random_sample['x'])
results = test(random_sample, sampled=True)
print('activation function: ', results['out'])
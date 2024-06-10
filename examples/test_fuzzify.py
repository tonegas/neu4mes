import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
from neu4mes.visualizer import MPLVisulizer

# Input of the function
x = Input('x')
# Output the function
target_y = Input('target_y')

# Create the target functions
data_x = np.random.rand(250)*10-5
data_a = 2
data_b = -3
dataset = {'x': data_x, 'target_y': (data_a*data_x) + data_b}

# Create the neu4mes object
#opt_fun = Neu4mes(visualizer=MPLVisulizer())
print('EXAMPLE 1')
fuz = Fuzzify(11,centers=[-5,-4,-3,-2,-1,0,1,2,3,4,5],functions='Triangular')
out = Output('out', fuz(x))

opt_fun = Neu4mes()
opt_fun.minimizeError('x', target_y, out, 'mse')
opt_fun.neuralizeModel()
opt_fun.loadData(dataset)

random_sample = opt_fun.get_random_samples(window=2)
print('X: ',random_sample['x'])
results = opt_fun(random_sample, sampled=True)
print('activation function: ', results['out'])


print('\nEXAMPLE 2')
fuz = Fuzzify(11,centers=[-5,-4,-3,-2,-1,0,1,2,3,4,5],functions='Rectangular')
out = Output('out', fuz(x))

opt_fun = Neu4mes()
opt_fun.minimizeError('x', target_y, out, 'mse')
opt_fun.neuralizeModel()
opt_fun.loadData(dataset)

random_sample = opt_fun.get_random_samples(window=2)
print('X: ',random_sample['x'])
results = opt_fun(random_sample, sampled=True)
print('activation function: ', results['out'])
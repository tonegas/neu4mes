import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
from neu4mes.visualizer import MPLVisulizer

# Quadratic function
#def parametric_fun(x,a,b,c):
#    return x**2*a+x*b+c

# Linear function
def parametric_fun(x,a,b):
    return x*a+b

# Create the dataset 
data_x = np.asarray(range(1000), dtype=np.float32)
data_a = 2
data_b = -3
#data_c = 2
dataset = {'x': data_x, 'target_y': parametric_fun(data_x,data_a,data_b)}
print('x: ', dataset['x'][:10])
print('target_y: ', dataset['target_y'][:10])


### EXAMPLE 1
print('#### EXAMPLE 1 - TEST FIR ####')

# Input of the function
x = Input('x')
target_y = Input('target_y')

## create a parameter
fir_mono_out_mono_window = Fir(output_dimension=1)
fir_mono_out_multi_window = Fir(output_dimension=1)
fir_multi_out_mono_window = Fir(output_dimension=3)
fir_multi_out_multi_window = Fir(output_dimension=3)

# Output of the function
y1 = Output('fir_mono_out_mono_window',fir_mono_out_mono_window(x.tw([-1,0])))
y2 = Output('fir_mono_out_multi_window',fir_mono_out_multi_window(x.tw([-5,0])))
y3 = Output('fir_multi_out_mono_window',fir_multi_out_mono_window(x.tw([-1,0])))
y4 = Output('fir_multi_out_multi_window',fir_multi_out_multi_window(x.tw([-5,0])))
test = Neu4mes()
test.minimizeError('out', target_y.sw(1), y1, 'rmse')
test.minimizeError('out', target_y.sw(1), y2, 'rmse')
test.minimizeError('out', target_y.sw(3), y3, 'rmse')
test.minimizeError('out', target_y.sw(3), y4, 'rmse')
# Neuralize the models
test.neuralizeModel()
# Load the dataset create with the target function
test.loadData(dataset)

print('BEFORE TRAINING')
#sample = {'in1':[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]} ## 2 samples
sample = test.get_random_samples(2) 
print('random sample: ', sample)
results = test(sample, sampled=True)
print('results: ', results)

# Train the models
test.trainModel(test_percentage = 20, training_params={'num_of_epochs':50, 'train_batch_size':4, 'test_batch_size':4})

print('random sample: ', sample)
results = test(sample, sampled=True)
print('results: ', results)
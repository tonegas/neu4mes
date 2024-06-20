import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
from neu4mes.visualizer import MPLVisulizer
from pprint import pprint

# Quadratic function
def parametric_fun(x,a,b,c):
    return x**2*a+x*b+c

# Linear function
def parametric_fun(x,a,b):
    return x*a+b

### EXAMPLE 1
print('#### EXAMPLE 1 - TEST FIR ####')

# Create the dataset 
data_x = np.asarray(range(1000), dtype=np.float32)
data_a = 2
data_b = -3
#data_c = 2
dataset = {'x': data_x,
           'x_multi': np.repeat(np.array(data_x)[:, np.newaxis], 3, axis=1), 
           'target_y': parametric_fun(data_x,data_a,data_b), 
           'target_y_multi': np.repeat(np.array(parametric_fun(data_x,data_a,data_b))[:, np.newaxis], 3, axis=1)}
print('### EXAMPLE DATASET ###')
print('x: ', dataset['x'][:3])
print('x_multi: ', dataset['x_multi'][:3])
print('target_y: ', dataset['target_y'][:3])
print('target_y_multi: ', dataset['target_y_multi'][:3])
print('x shape: ', dataset['x'].shape)
print('x_multi shape: ', dataset['x_multi'].shape)
print('target_y shape: ', dataset['target_y'].shape)
print('target_y_multi shape: ', dataset['target_y_multi'].shape)

# Input of the function
x = Input('x')
target_y = Input('target_y')
target_y_multi = Input('target_y_multi', dimensions=3)

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
test.minimizeError('out1', target_y.sw(1), y2, 'rmse')
test.minimizeError('out2', target_y_multi.sw(1), y3, 'rmse')
test.minimizeError('out3', target_y_multi.sw(1), y4, 'rmse')

# Neuralize the models
test.neuralizeModel()
# Load the dataset create with the target function
test.loadData(dataset)

print('BEFORE TRAINING')
#sample = {'in1':[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]} ## 2 samples
sample = test.get_random_samples(1) 
print('random sample: ', sample)
results = test(sample, sampled=True)
print('results')
pprint(results)

# Train the models
test.trainModel(test_percentage = 20, training_params={'num_of_epochs':50, 'train_batch_size':4, 'test_batch_size':4})

print('AFTER TRAINING')
print('random sample: ', sample)
results = test(sample, sampled=True)
print('results')
pprint(results)


### EXAMPLE 2
print('#### EXAMPLE 2 - TEST FUZZY ####')

def fun1(x):
    import torch
    return torch.sin(x)

def fun2(x):
    import torch
    return torch.cos(x)

# Input of the function
x = Input('x')
x_multi = Input('x_multi')

## create a parameter
fuzzy1 = Fuzzify(output_dimension=4, range=[1,4], functions='Triangular')
fuzzy2 = Fuzzify(output_dimension=4, range=[1,4], functions='Rectangular')
fuzzy3 = Fuzzify(output_dimension=4, range=[1,4], functions=fun1)
fuzzy4 = Fuzzify(output_dimension=4, range=[1,4], functions=fun2)

# Output of the function
y1 = Output('fuzzy1',fuzzy1(x.last()))
y2 = Output('fuzzy2',fuzzy2(x.last()))
y3 = Output('fuzzy3',fuzzy3(x.last()))
y4 = Output('fuzzy4',fuzzy4(x.last()))
y1_multi = Output('fuzzy1_multi',fuzzy1(x_multi.tw([-2,2])))
y2_multi = Output('fuzzy2_multi',fuzzy2(x_multi.tw([-2,2])))
y3_multi = Output('fuzzy3_multi',fuzzy3(x_multi.tw([-2,2])))
y4_multi = Output('fuzzy4_multi',fuzzy4(x_multi.tw([-2,2])))

test = Neu4mes()
test.addModel(y1)
test.addModel(y2)
test.addModel(y3)
test.addModel(y4)
test.addModel(y1_multi)
test.addModel(y2_multi)
test.addModel(y3_multi)
test.addModel(y4_multi)
test.neuralizeModel()

sample = {'x':[1], 'x_multi':[1,2,3,4]}
print('example: ', sample)
pprint(test(sample))


### EXAMPLE 3
print('#### EXAMPLE 3 - TEST PART ####')

# Input of the function
x_multi = Input('x_multi', dimensions=3)

# Output of the function
y1 = Output('part1', Part(x_multi.last(), 0, 1))
y2 = Output('part2', Part(x_multi.last(), 0, 2))
y3 = Output('Part3', Part(x_multi.tw([-2,2]), 0, 1))
y4 = Output('Part4', Part(x_multi.tw([-2,2]), 0, 2))

test = Neu4mes()
test.addModel(y1)
test.addModel(y2)
test.addModel(y3)
test.addModel(y4)
test.neuralizeModel()

sample = {'x_multi':[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]}
print('example: ', sample)
pprint(test(sample))


### EXAMPLE 4
print('#### EXAMPLE 4 - TEST SELECT ####')

# Input of the function
x_multi = Input('x_multi', dimensions=3)

# Output of the function
y1 = Output('part1', Select(x_multi.last(), 0))
y2 = Output('part2', Select(x_multi.last(), 1))
y3 = Output('part3', Select(x_multi.last(), 2))

test = Neu4mes()
test.addModel(y1)
test.addModel(y2)
test.addModel(y3)
test.neuralizeModel()

sample = {'x_multi':[[1,2,3]]}
print('example: ', sample)
print(test(sample))


### EXAMPLE 5
print('#### EXAMPLE 5 - TEST SAMPLE SELECT ####')

# Input of the function
x = Input('x')
x_multi = Input('x_multi', dimensions=3)

# Output of the function
y1 = Output('sample_select1', SampleSelect(x.sw([-2,2]), 0))
y2 = Output('sample_select2', SampleSelect(x.sw([-2,2]), 1))
y3 = Output('sample_select3', SampleSelect(x.sw([-2,2]), 2))
y4 = Output('sample_select4', SampleSelect(x.sw([-2,2]), 3))
y1_multi = Output('sample_select1_multi', SampleSelect(x_multi.sw([-2,2]), 0))
y2_multi = Output('sample_select2_multi', SampleSelect(x_multi.sw([-2,2]), 1))
y3_multi = Output('sample_select3_multi', SampleSelect(x_multi.sw([-2,2]), 2))
y4_multi = Output('sample_select4_multi', SampleSelect(x_multi.sw([-2,2]), 3))

test = Neu4mes()
test.addModel(y1)
test.addModel(y2)
test.addModel(y3)
test.addModel(y4)
test.addModel(y1_multi)
test.addModel(y2_multi)
test.addModel(y3_multi)
test.addModel(y4_multi)
test.neuralizeModel()

sample = {'x':[1,2,3,4], 'x_multi':[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]}
print('example: ', sample)
pprint(test(sample))

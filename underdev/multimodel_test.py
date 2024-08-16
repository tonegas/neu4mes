import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
from neu4mes.visualizer import MPLVisulizer
from pprint import pprint

def linear_function(x, k1, k2):
    return x*k1 + k2

data_a = np.arange(1,1001, dtype=np.float32)
bias_a = np.arange(1,1001, dtype=np.float32)
data_b = linear_function(data_a, 2, bias_a)

data_c = np.arange(1,1001, dtype=np.float32)
bias_c = np.arange(1,1001, dtype=np.float32)
data_d = linear_function(data_c, 5, bias_c)

dataset = {'a': data_a, 'bias_a':bias_a, 'b_t': data_b, 'c':data_c, 'bias_c':bias_c, 'd_t':data_d}

a = Input('a')
a_bias = Input('bias_a')
b_t = Input('b_t')
b = Output('b',Linear(W='condiviso')(a.last())+Linear(W='A')(a_bias.last()))

model = Neu4mes(seed=42)
model.addModel('b_model',b)
model.addMinimize('b_min',b,b_t.last())

c = Input('c')
c_bias = Input('bias_c')
d_t = Input('d_t')
d = Output('d',Linear(W='condiviso')(c.last())+Linear(W='C')(c.last())+Linear(W='D')(c_bias.last()))

model.addModel('d_model', d)
model.addMinimize('d_min',d,d_t.last())

model.neuralizeModel(0.1)

model.loadData('dataset', dataset)

params = {'num_of_epochs': 100, 
          'train_batch_size': 32, 
          'val_batch_size':32, 
          'test_batch_size':1, 
          'learning_rate':0.01}

print('### BEFORE TRAIN ###')
print(model.model.relation_forward['Linear18'].weights)
print(model.model.relation_forward['Linear8'].weights)
print(model.model.relation_forward['Linear5'].weights)
print(model.model.relation_forward['Linear25'].weights)
print(model.model.relation_forward['Linear21'].weights)
print('# PARAMETERS #')
print(model.model.all_parameters['A'])
print(model.model.all_parameters['C'])
print(model.model.all_parameters['D'])
print(model.model.all_parameters['condiviso'])
#model.trainModel(training_params=params, splits=[100,0,0], connect={'b_in':'b'}, lr_gain={'condiviso':1, 'A':1, 'B':1, 'C':1, 'D':1})
model.trainModel(splits=[100,0,0], training_params=params)
#model.trainModel(connect={'b_in':'b'}, lr_gain = {'condiviso':0,'A':0,'B':0,'C':1,'D':1})
print('### AFTER TRAIN ###')
print(model.model.relation_forward['Linear18'].weights)
print(model.model.relation_forward['Linear8'].weights)
print(model.model.relation_forward['Linear5'].weights)
print(model.model.relation_forward['Linear25'].weights)
print(model.model.relation_forward['Linear21'].weights)
print('# PARAMETERS #')
print(model.model.all_parameters['A'])
print(model.model.all_parameters['C'])
print(model.model.all_parameters['D'])
print(model.model.all_parameters['condiviso'])
print('### MODEL PARAMETERS ###')
for param in model.model.parameters():
    print(type(param), param.size())
print('### NUMBER OF PARAMETERS ###')
print(sum(p.numel() for p in model.model.parameters()))
'''
a = Input('a')
a_bias = Input('bias_a')
b_t = Input('b_t')
b = Output('b',Fir(parameter='condiviso')(a.last())+Fir(parameter='A')(a_bias.last()))

model = Neu4mes(seed=42)
model.addModel('b_model',b)
model.addMinimize('b_min',b,b_t.last())

c = Input('c')
c_bias = Input('bias_c')
d_t = Input('d_t')
d = Output('d',Fir(parameter='condiviso')(c.last())+Fir(parameter='C')(c.last())+Fir(parameter='D')(c_bias.last()))

model.addModel('d_model', d)
model.addMinimize('d_min',d,d_t.last())

model.neuralizeModel(0.1)

model.loadData('dataset', dataset)

params = {'num_of_epochs': 100, 
          'train_batch_size': 32, 
          'val_batch_size':32, 
          'test_batch_size':1, 
          'learning_rate':0.01}

print('### BEFORE TRAIN ###')
print(model.model.relation_forward['Fir18'].lin.weight)
print(model.model.relation_forward['Fir8'].lin.weight)
print(model.model.relation_forward['Fir5'].lin.weight)
print(model.model.relation_forward['Fir25'].lin.weight)
print(model.model.relation_forward['Fir21'].lin.weight)
print('# PARAMETERS #')
print(model.model.all_parameters['A'])
print(model.model.all_parameters['C'])
print(model.model.all_parameters['D'])
print(model.model.all_parameters['condiviso'])
#model.trainModel(training_params=params, splits=[100,0,0], connect={'b_in':'b'}, lr_gain={'condiviso':1, 'A':1, 'B':1, 'C':1, 'D':1})
model.trainModel(splits=[100,0,0], training_params=params)
#model.trainModel(connect={'b_in':'b'}, lr_gain = {'condiviso':0,'A':0,'B':0,'C':1,'D':1})
print('### AFTER TRAIN ###')
print(model.model.relation_forward['Fir18'].lin.weight)
print(model.model.relation_forward['Fir8'].lin.weight)
print(model.model.relation_forward['Fir5'].lin.weight)
print(model.model.relation_forward['Fir25'].lin.weight)
print(model.model.relation_forward['Fir21'].lin.weight)
print('# PARAMETERS #')
print(model.model.all_parameters['A'])
print(model.model.all_parameters['C'])
print(model.model.all_parameters['D'])
print(model.model.all_parameters['condiviso'])
print('### MODEL PARAMETERS ###')
for param in model.model.parameters():
    print(type(param), param.size())
print('### NUMBER OF PARAMETERS ###')
print(sum(p.numel() for p in model.model.parameters()))
'''
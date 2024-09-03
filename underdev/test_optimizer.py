import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())
from neu4mes import *

def funIn(x,w):
    return x*w

def funOut(x,w):
    return x/w

## Model1
input1 = Input('in1')
a = Parameter('a', dimensions=1, tw=0.05, values=[[1], [1], [1], [1], [1]])
shared_w = Parameter('w', values=[[1]])
output1 = Output('out1', Fir(parameter=a)(input1.tw(0.05))+ParamFun(funIn,n_input=1,parameters={'w':shared_w})(input1.last()))

test = Neu4mes()
test.addModel('model1', output1)
test.addMinimize('error1', input1.next(), output1)

## Model2
input2 = Input('in2')
b = Parameter('b', dimensions=1, tw=0.05, values=[[1], [1], [1], [1], [1]])
output2 = Output('out2', Fir(parameter=b)(input2.tw(0.05))+ParamFun(funOut,n_input=1,parameters={'w':shared_w})(input2.last()))

test.addModel('model2', output2)
test.addMinimize('error2', input2.next(), output2)
test.neuralizeModel(0.01)

# Dataset for train
data_in1 = np.linspace(0, 5, 60)
data_in2 = np.linspace(10, 15, 60)
data_out1 = 2
data_out2 = -3
dataset = {'in1': data_in1, 'in2': data_in2, 'out1': data_in1 * data_out1, 'out2': data_in2 * data_out2}
test.loadData(name='dataset1', source=dataset)

data_in1 = np.linspace(0, 5, 60)
data_in2 = np.linspace(10, 15, 60)
data_out1 = 2
data_out2 = -3
dataset = {'in1': data_in1, 'in2': data_in2, 'out1': data_in1 * data_out1, 'out2': data_in2 * data_out2}
test.loadData(name='dataset2', source=dataset)

# Optimizer
# Basic usage
# Standard optimizer with standard configuration
# We train all the models with split [70,20,10], lr =0.01 and epochs = 100
# TODO if more than one dataset is loaded I use all the dataset
test.trainModel()

# We train only model1 with split [100,0,0]
# TODO Learning rate automoatically optimized based on the mean and variance of the output
# TODO num_of_epochs automatically defined
# now is 0.001 for learning rate and 100 for the epochs and optimizer Adam
test.trainModel(models='model1', splits=[100,0,0])

# Set number of epoch and learning rate via parameters it works only for standard parameters
test.trainModel(models='model1', splits=[100, 0, 0], lr=0.5, num_of_epochs=5)

# Set number of epoch and learning rate via parameters it works only for standard parameters and use two different dataset one for train and one for validation
test.trainModel(models='model1', train_dataset='dataset1', validation_dataset='dataset2', lr=0.5, num_of_epochs=10)

# Use dictionary for set number of epoch, learning rate, etc.. This configuration works only standard parameters (all the parameters that are input of the trainModel).
training_params = {
    'models':['model1'],
    'splits': [55, 40, 5],
    'num_of_epochs': 20,
    'lr': 0.7
}
test.trainModel(training_params = training_params)
# If I add a function parameter it has the priority
# In this case apply train parameter but on a different model
test.trainModel(models='model2', training_params = training_params)

# Modify additional parameters in the optimizer that are not present in the standard parameter
# In this case I modify the learning rate and the betas of the Adam optimizer
optimizer_defaults =  {
    'lr': 0.1,
    'betas': (0.5, 0.99)
}
test.trainModel(training_params = training_params, optimizer_defaults = optimizer_defaults, lr = 0.2)
test.trainModel(training_params = training_params, optimizer_defaults = optimizer_defaults)
test.trainModel(training_params = training_params)
# For the optimizer parameter the priority is the following
# max priority to the function parameter ('lr' : 0.2)
# then the standard_optimizer_parameters ('lr' : 0.1)
# finally the standard_train_parameters  ('lr' : 0.5)

# Modify the optimizer non standard args of a stardard optimizer
# In this case use the SGD with 0.2 of momentum
optimizer_defaults = {
    'momentum':0.2
}
test.trainModel(optimizer='SGD', training_params = training_params, optimizer_defaults = optimizer_defaults, lr = 0.2)

# Modify standard optimizer parameter for each training parameter
training_params = {
    'models':['model1'],
    'splits': [100, 0, 0],
    'num_of_epochs': 30,
    'lr': 0.5,
    'lr_param': {'a': 0.1}
}
test.trainModel(training_params = training_params)

# Modify standard optimizer parameter for each training parameter using optimizer_params
training_params = {
    'models':['model1'],
    'splits': [100, 0, 0],
    'num_of_epochs': 40,
    'lr': 0.5,
    'lr_param': {'a': 0.1},
    'optimizer_params' : [{'params':['a'],'lr':0.7}],
    'optimizer_defaults' : {'lr': 0.12}
}
optimizer_params = [
    {'params':['a'],'lr':0.6}
]
optimizer_defaults = {
    'lr': 0.2
}
test.trainModel(training_params = training_params, optimizer_params = optimizer_params, optimizer_defaults=optimizer_defaults, lr_param={'a': 0.4})
test.trainModel(training_params = training_params, optimizer_params = optimizer_params, optimizer_defaults=optimizer_defaults)
test.trainModel(training_params = training_params, optimizer_params = optimizer_params)
test.trainModel(training_params = training_params)
training_params = {
    'models':['model1'],
    'splits': [100, 0, 0],
    'num_of_epochs': 40,
    'lr': 0.5,
    'lr_param': {'a': 0.1},
    'optimizer_defaults' : {'lr': 0.2}
}
test.trainModel(training_params = training_params)
# The priority is the following
# max priority to the function parameter ( 'lr_param'={'a': 0.4})
# then the optimizer_params ( {'params':'a','lr':0.6} )
# then the optimizer_params inside the train_parameters ( {'params':['a'],'lr':0.7} )
# finally the train_parameters  ( 'lr_param'={'a': 0.1})
# The value applied is 0.4 on the weight 'a'

# Maximum level of configuration I define a custom optimizer with defaults
class RMSprop(Optimizer):

    def __init__(self, optimizer_defaults = {}, optimizer_params = []):
        super(RMSprop, self).__init__('RMSprop', optimizer_defaults, optimizer_params)

    def get_torch_optimizer(self):
        import torch
        return torch.optim.RMSprop(self.replace_key_with_params(), **self.optimizer_defaults)

optimizer_defaults = {
    'alpha' : 0.8
}
optimizer = RMSprop(optimizer_defaults)
test.trainModel(optimizer=optimizer, training_params = training_params,optimizer_defaults={'lr':0.1}, lr = 0.4)
test.trainModel(optimizer=optimizer, training_params = training_params,optimizer_defaults={'lr':0.1})
test.trainModel(optimizer=optimizer, training_params = training_params)
test.trainModel(optimizer=optimizer)
# For the optimizer default the priority is the following
# max priority to the function parameter ('lr'= 0.4)
# then the optimizer_defaults ('lr':0.1)
# then the optimizer_defaults inside the train_parameters ('lr'= 0.2)
# finally the train_parameters  ('lr'= 0.5)
# The value applied is 0.4 on the weight 'a'

# Maximum level of configuration I define a custom optimizer with custom value for each params
optimizer_defaults = {
    'alpha' : 0.8
}
optimizer_params = [
    {'params':['a'],'lr':0.6}
]
optimizer = RMSprop(optimizer_defaults, optimizer_params)
test.trainModel(optimizer=optimizer, training_params = training_params, optimizer_defaults = {'lr':0.3}, optimizer_params = [{'params':['a'],'lr':1.0}], lr_param = {'a':0.2} )
test.trainModel(optimizer=optimizer, training_params = training_params, optimizer_defaults = {'lr':0.3}, optimizer_params = [{'params':['a'],'lr':1.0}] )
test.trainModel(optimizer=optimizer, training_params = training_params, optimizer_defaults = {'lr':0.3} )
test.trainModel(optimizer=optimizer, optimizer_defaults = {'lr':0.3} )
# The priority is the following
# max priority to the function parameter ( 'lr_param'={'a': 0.2})
# then the optimizer_params ( [{'params':['a'],'lr':1.0}] )
# then the optimizer_params inside the train_parameters (  [{'params':['a'],'lr':0.7}] )
# then the train_parameters  ( 'lr_param'={'a': 0.1} )
# finnaly the optimizer_paramsat the time of the optimizer initialization [{'params':['a'],'lr':0.6}]
# The value applied is 0.2 on the weight 'a'
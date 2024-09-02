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
output1 = Output('out1', Fir(parameter=a)(input1.tw(0.05))+ParamFun(funIn,parameters={'w':shared_w})(input1.last()))

test = Neu4mes(visualizer=None, seed=42)
test.addModel('model1', output1)
test.addMinimize('error1', input1.next(), output1)

## Model2
input2 = Input('in2')
b = Parameter('b', dimensions=1, tw=0.05, values=[[1], [1], [1], [1], [1]])
output2 = Output('out2', Fir(parameter=b)(input2.tw(0.05))+ParamFun(funOut,parameters={'w':shared_w})(input2.last()))

test.addModel('model2', output2)
test.addMinimize('error2', input2.next(), output2)
test.neuralizeModel(0.01)

# Dataset for train
data_in1 = np.linspace(0, 5, 6)
data_in2 = np.linspace(10, 15, 6)
data_out1 = 2
data_out2 = -3
dataset = {'in1': data_in1, 'in2': data_in2, 'out1': data_in1 * data_out1, 'out2': data_in2 * data_out2}
test.loadData(name='dataset1', source=dataset)

data_in1 = np.linspace(0, 5, 6)
data_in2 = np.linspace(10, 15, 6)
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
test.trainModel(models='model1', splits=[100, 0, 0], lr=0.5, num_of_epochs=200)

# Set number of epoch and learning rate via parameters it works only for standard parameters and use two different dataset one for train and one for validation
test.trainModel(models='model1', train_dataset='dataset1', validation_dataset='dataset2', splits=[100, 0, 0], lr=0.5, num_of_epochs=200)

# Use dictionary for set number of epoch, learning rate, etc.. This configuration works only standard parameters (all the parameters that are input of the trainModel).
train_parameters = {
    'models':['model1'],
    'splits': [100, 0, 0],
    'num_of_epochs': 200,
    'lr': 0.5
}
test.trainModel(train_parameters = train_parameters)
# If I add a function parameter it has the priority
# In this case apply train parameter but on a different model
test.trainModel(models='model2', train_parameters = train_parameters)

# Modify additional parameters in the optimizer that are not present in the standard parameter
# In this case I modify the learning rate and the betas of the Adam optimizer
optimizer_defaults =  {
    'lr': 0.1,
    'betas': (0.5, 0.99)
}
# For the optimizer parameter the priority is the following
# max priority to the function parameter ('lr' : 0.2)
# then the standard_optimizer_parameters ('lr' : 0.1)
# finally the standard_train_parameters  ('lr' : 0.5)
# The value applied is 0.2
test.trainModel(train_parameters = train_parameters, optimizer_defaults = optimizer_defaults, lr = 0.2)

# Modify the optimizer non standard args of a stardard optimizer
# In this case use the SGD with 0.2 of momentum
optimizer_defaults = {
    'momentum':0.2
}
test.trainModel(optimizer='SGD', train_parameters = train_parameters, optimizer_defaults = optimizer_defaults, lr = 0.2)

# Modify standard optimizer parameter for each training parameter
train_parameters = {
    'models':['model1'],
    'splits': [100, 0, 0],
    'num_of_epochs': 200,
    'lr': 0.5,
    'lr_params': {'a': 0.1}
}
test.trainModel(train_parameters = train_parameters)

# Modify standard optimizer parameter for each training parameter using optimizer_params
# The priority is the following
# max priority to the function parameter ( 'lr_param'={'a': 0.4})
# then the optimizer_params ( {'params':['a'],'lr':0.6} )
# then the optimizer_params inside the train_parameters ( {'params':['a'],'lr':0.7} )
# finally the standard_train_parameters  ( 'lr_param'={'a': 0.1})
# The value applied is 0.4 on the weight 'a'
train_parameters = {
    'models':['model1'],
    'splits': [100, 0, 0],
    'num_of_epochs': 200,
    'lr': 0.5,
    'lr_params': {'a': 0.1},
    'optimizer_params' : [{'params':['a'],'lr':0.7}],
    'optimizer_defaults' : {'lr': 0.2}
}
optimizer_params = [
    {'params':['a'],'lr':0.6}
]
optimizer_defaults = {
    'lr': 0.2
}
test.trainModel(train_parameters = train_parameters, optimizer_params=optimizer_params, optimizer_defaults=optimizer_defaults, lr_param={'a': 0.4})

# Maximum level of configuration I define a custom optimizer
class RMSprop(Optimizer):
    def __init__(self, optimizer_defaults, optimizer_params = []):
        super(RMSprop, self).__init__('RMSprop', optimizer_defaults, optimizer_params)

    def get_torch_optimizer(self):
        import torch
        return torch.optim.RMSprop(self.replace_key_with_params(), **self.optimizer_defaults)

optimizer_defaults = {
    'alpha' : 0.8
}
optimizer = RMSprop(optimizer_defaults)
test.trainModel(optimizer=optimizer, train_parameters = train_parameters, optimizer_defaults = optimizer_defaults, lr = 0.2)
# For the optimizer parameter the priority is the following
# max priority to the function parameter ('lr' : 0.2)
# then the standard_optimizer_parameters ('lr' : 0.1)
# then the standard_train_parameters  ('lr' : 0.5)
# then the inizitlization parameter of the MyOptimizer ('lr' : 0.05)
# The value applied is 0.2
import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
from neu4mes.visualizer import MPLVisulizer
from neu4mes.output import log

# This example shows how to fit a simple linear model.
# The model chosen is a mass spring damper.
# The data was created previously and loaded from file.
# The data are the position/velocity of the mass and the force applied.
# The neural model mirrors the structure of the physical model.
# The network build estimate the future position of the mass and the velocity.

# Create neural model
# List the input of the model
x = Input('x') # Position of the mass
dx = Input('dx') # Velocity of the mass
F = Input('F') # Force

# List the relations of the model
displacement_x = Fir(parameter_init=init_negexp)
force_x = Fir(parameter_init=init_constant,parameter_init_params={'value':1})
displacement_dx = Fir(parameter_init=init_negexp)
force_dx = Fir(parameter_init=init_constant,parameter_init_params={'value':1})

# List the outputs of the model
next_x = Output('next_x', displacement_x(x.tw(0.2))+force_x(F.last()))
next_dx = Output('next_dx', displacement_dx(x.tw(0.2))+force_dx(F.last()))

# Add the neural models to the neu4mes structure
mass_spring_damper = Neu4mes(visualizer=MPLVisulizer())
mass_spring_damper.addModel('position_model',next_x)
mass_spring_damper.addModel('velocity_model',next_dx)

# These functions are used to impose the minimization objectives.
# Here it is minimized the error between the future position of x get from the dataset x.z(-1)
# and the estimator designed useing the neural network. The miniminzation is imposed via MSE error.
mass_spring_damper.addMinimize('position', x.next(), next_x, 'mse')
# The second minimization is between the velocity get from the dataset and the velocity estimator.
mass_spring_damper.addMinimize('velocity', dx.next(), next_dx, 'mse')

# Nauralize the model and gatting the neural network. The sampling time depends on the datasets.
mass_spring_damper.neuralizeModel(sample_time = 0.05) # The sampling time depends on the dataset

# Data load
data_struct = ['time','x','dx','F']
data_folder = './tutorials/datasets/mass-spring-damper/data/'
mass_spring_damper.loadData(name='mass_spring_dataset', source=data_folder, format=data_struct, delimiter=';')

#Neural network train not reccurent training
params = {'num_of_epochs': 100,
          'train_batch_size': 128, 
          'val_batch_size':128, 
          'test_batch_size':1, 
          'learning_rate':0.001}

'''
def early_stop(train_losses, val_losses):
    for loss_name, loss_value in train_losses.items():
        if sum(loss_value[-10:]) > 0.08:
            return False
    return True
'''

mass_spring_damper.trainModel(splits=[70,20,10], training_params = params)

# Add visualizer and show the results on the loaded dataset
#vis = MPLVisulizer()
#vis.set_n4m(mass_spring_damper)
#vis.showOneResult("validation")

#Neural network train not reccurent training
#mass_spring_damper.trainModel(splits=[70,20,10], training_params =  {'learning_rate':0.0001}, close_loop={'x':'x[k+1]'}, prediction_samples=10)

# Add visualizer and show the results on the loaded dataset
#vis.showOneResult("validation")


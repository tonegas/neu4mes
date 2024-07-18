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

##TODO: input name must be different from output name

# List the output of the model
xk1 = Output('x[k+1]', Fir(x.tw(0.2))+Fir(F.last()))
dxk1 = Output('dx[k+1]', Fir(Fir(x.tw(0.2))+Fir(F.last())))

# Add the neural models to the neu4mes structure
mass_spring_damper = Neu4mes(visualizer=MPLVisulizer())
mass_spring_damper.addModel(xk1)
mass_spring_damper.addModel(dxk1)

# These functions are used to impose the minimization objectives.
# Here it is minimized the error between the future position of x get from the dataset x.z(-1)
# and the estimator designed useing the neural network. The miniminzation is imposed via MSE error.
mass_spring_damper.minimizeError('next-pos', x.next(), xk1, 'mse')
# The second minimization is between the velocity get from the dataset and the velocity estimator.
mass_spring_damper.minimizeError('next-vel', dx.next(), dxk1, 'mse')

# Nauralize the model and gatting the neural network. The sampling time depends on the datasets.
mass_spring_damper.neuralizeModel(sample_time = 0.05) # The sampling time depends on the dataset

# Data load
data_struct = ['time',('x', 'x_state'),'dx','F']
data_folder = './tutorials/datasets/mass-spring-damper/data/'
mass_spring_damper.loadData(name='mass_spring_dataset', source=data_folder, format=data_struct, delimiter=';')

#Neural network train
params = {'num_of_epochs': 100, 
          'train_batch_size': 4, 
          'val_batch_size':4, 
          'test_batch_size':1, 
          'learning_rate':0.001}
mass_spring_damper.trainModel(splits=[70,20,10], prediction_horizon=0.2, training_params = params)



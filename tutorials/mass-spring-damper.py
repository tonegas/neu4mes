import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
from neu4mes.visualizer import MPLVisulizer

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

# List the output of the model
xk1 = Output('x[k+1]', Fir(x.tw(0.2))+Fir(F))
dxk1 = Output('dx[k+1]', Fir(Fir(x.tw(0.2))+Fir(F)))

# Add the neural models to the neu4mes structure
mass_spring_damper = Neu4mes(visualizer=MPLVisulizer())
mass_spring_damper.addModel(xk1)
mass_spring_damper.addModel(dxk1)

# These functions are used to impose the minimization objectives.
# Here it is minimized the error between the future position of x get from the dataset x.z(-1)
# and the estimator designed useing the neural network. The miniminzation is imposed via MSE error.
mass_spring_damper.minimizeError('next-pos', x.z(-1), xk1, 'mse')
# The second minimization is between the velocity get from the dataset and the velocity estimator.
mass_spring_damper.minimizeError('next-vel', dx.z(-1), dxk1, 'mse')

# Nauralize the model and gatting the neural network. The sampling time depends on the datasets.
mass_spring_damper.neuralizeModel(sample_time = 0.05) # The sampling time depends on the dataset

# Data load
data_struct = ['time','x','dx','F']
data_folder = './tutorials/datasets/mass-spring-damper/data/'
mass_spring_damper.loadData(data_folder, data_struct)

#Neural network train
mass_spring_damper.trainModel(test_percentage = 10, training_params = {'num_of_epochs': 150})

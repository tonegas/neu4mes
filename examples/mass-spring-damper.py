import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
from neu4mes.visualizer import MPLVisulizer

# This example shows how to fit a simple linear model. The model chosen is a mass spring damper.
# The data was created previously and loaded from file.
# The neural model mirrors the structure of the physical model.

# Create neural model
# List the input of the model
x = Input('x') # Position
F = Input('F') # Force

# List the output of the model
x_z = Output('x_z', Fir(x.tw(0.2))+Fir(F))

# Add the neural model to the neu4mes structure and neuralization of the model
mass_spring_damper = Neu4mes()
mass_spring_damper.addModel(x_z)
mass_spring_damper.minimizeError('next-pos', x.z(-1), x_z, 'mse')

# Create the neural network
mass_spring_damper.neuralizeModel(sample_time = 0.05) # The sampling time depends to the dataset

# Data load
data_struct = ['time','x','x_s','F']
data_folder = './examples/datasets/mass-spring-damper/data/'
mass_spring_damper.loadData(data_folder, data_struct)

#Neural network train
mass_spring_damper.trainModel(test_percentage = 10, training_params = {'num_of_epochs': 100})
mass_spring_damper.trainRecurrentModel(close_loop={'x':'x_z'},test_percentage = 10, training_params = {'num_of_epochs': 100})
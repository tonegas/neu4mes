import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

import time
from neu4mes import *
from neu4mes.visualizer import StandardVisualizer

# if __name__ == '__main__':
# Create neural model
# List the input of the model
x = Input('x')
F = Input('F')

# List the output of the model
x_z = Output('x_k', Fir(x.tw(0.2))+Fir(F))

# Add the neural model to the neu4mes structure and neuralization of the model
mass_spring_damper = Neu4mes(verbose = 1, visualizer=StandardVisualizer)
mass_spring_damper.addModel(x_z)
mass_spring_damper.minimizeError('next-pos', x.z(-1), x_z, 'mse')

# Create the neural network
mass_spring_damper.neuralizeModel(0.05)

# Data load
data_struct = ['time','x','x_s','F']
data_folder = './examples/datasets/mass-spring-damper/data/'
mass_spring_damper.loadData(data_folder, data_struct)

#Neural network train
start = time.time()
mass_spring_damper.trainModel(test_percentage = 10, training_params={'num_of_epochs': 2000})
end = time.time()
print(end - start)

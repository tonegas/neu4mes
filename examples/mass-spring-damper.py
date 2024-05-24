import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

import time
from neu4mes import *
from neu4mes.visualizer import StandardVisualizer

# Create neural model
x = Input('x')
F = Input('F')
x_z = Output('x_k', Fir(x.tw(2))+Fir(F))


# Add the neural model to the neu4mes structure and neuralization of the model
mass_spring_damper = Neu4mes(verbose = True,  visualizer = StandardVisualizer())
mass_spring_damper.addModel(x_z)
mass_spring_damper.minimizeError('out',x.z(-1), x_z, 'mse')
mass_spring_damper.neuralizeModel(0.05)

# Data load
data_struct = ['time','x','x_s','F']
data_folder = './examples/datasets/mass-spring-damper/data/'
mass_spring_damper.loadData(data_struct)

# Neural network train
start = time.time()
mass_spring_damper.trainModel(test_percentage = 10, show_results = True)
end = time.time()
print(end - start)

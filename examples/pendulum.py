import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
from neu4mes.visualizer import StandardVisualizer

# Create neu4mes structure
pendolum = Neu4mes(verbose = True, visualizer = StandardVisualizer())

# Create neural model
theta = Input('theta')
T     = Input('torque')
lin_theta = Fir(theta.tw(1.5))
sin_theta = Fir(Sin(theta))
torque = Fir(T)
out = Output('theta_s', lin_theta+sin_theta+torque)

# Add the neural model to the neu4mes structure and neuralization of the model
#pendolum.minimizeError(theta.z(-1), lin_theta+sin_theta+torque)
pendolum.addModel(out)
pendolum.neuralizeModel(0.05)

# Data load
data_struct = ['time','theta','theta_s','','','torque']
data_folder = './examples/datasets/pendulum/data/'
pendolum.loadData(folder = data_folder, format=data_struct)

# Neural network train
pendolum.trainModel(test_percentage = 30, show_results = True)
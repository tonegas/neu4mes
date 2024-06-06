
import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
from neu4mes.visualizer import MPLVisulizer

# Create neu4mes structure
pendolum = Neu4mes(visualizer = MPLVisulizer())

# Create neural model
theta = Input('theta')
T     = Input('torque')
# TODO the training stuck when add the parameter c
c = Parameter('c')
lin_theta = Fir(theta.tw(1.5))
sin_theta = Fir(Sin(theta.tw(1)))#+c
torque = Fir(T)
out = Output('theta_s', sin_theta+torque+c)

# Add the neural model to the neu4mes structure and neuralization of the model
pendolum.minimizeError('next-theta', theta.z(-1), sin_theta+torque)
pendolum.addModel(out)
pendolum.neuralizeModel(0.05)

# Data load
data_struct = ['time','theta','theta_s','','','torque']
data_folder = './examples/datasets/pendulum/data/'
pendolum.loadData(source=data_folder, format=data_struct)

# Neural network train
pendolum.trainModel(test_percentage = 30)
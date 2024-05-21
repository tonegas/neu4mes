import sys
import os
import numpy as np
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
from neu4mes.visualizer import StandardVisualizer

# Create neu4mes structure
pendolum = Neu4mes(verbose = True, visualizer = StandardVisualizer())

# Create neural model
theta = Input('theta')
T     = Input('torque')

p1 = Parameter('p1', dimensions =  1)
p2 = Parameter('p2', dimensions =  1)
def myFun(K1,K2,p1,p2):
    return p1*K1+p2*K2

parfun = ParamFun(myFun, output_dimension = 1) # definisco una funzione scalare basata su myFun
lin_theta = Fir(theta)
sin_theta = Fir(Sin(theta))
torque = Fir(T)
momentum = parfun(lin_theta, sin_theta)
out = Output('theta_s', lin_theta+sin_theta+torque+momentum)

# Add the neural model to the neu4mes structure and neuralization of the model
#pendolum.minimizeError(theta.z(-1), lin_theta+sin_theta+torque)
pendolum.addModel(out)
pendolum.neuralizeModel(0.05)

# Data load
data_struct = ['time','theta','theta_s','','','torque']
data_folder = './examples/datasets/pendulum/data/'
pendolum.loadData(data_struct, folder = data_folder)

# Neural network train
pendolum.trainModel(test_percentage = 30, show_results = True)
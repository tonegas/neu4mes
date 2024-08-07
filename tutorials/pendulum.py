
import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
from neu4mes.visualizer import MPLVisulizer

# Create neu4mes structure
pendolum = Neu4mes(visualizer=MPLVisulizer())

# Create neural model
# Input of the neural model
theta = Input('theta')
T     = Input('torque')
omega = Input('omega_target')

# Relations of the neural model
gravity_force = Fir(Sin(theta.tw(0.5)))
friction = Fir(theta.tw(0.5))
torque = Fir(T.last())
out = Output('omega', gravity_force+friction+torque)

# Add the neural model to the neu4mes structure and neuralization of the model
pendolum.addMinimize('omega error', omega.next(), out)
pendolum.addModel('omega',out)
pendolum.neuralizeModel(0.05)

# Data load
data_struct = ['time','theta','omega_target','cos(theta)','sin(theta)','torque']
data_folder = './tutorials/datasets/pendulum/data/'
pendolum.loadData(name='pendulum_dataset', source=data_folder, format=data_struct, delimiter=';')

# Neural network train
params = {'learning_rate':0.001, 'train_batch_size':32, 'val_batch_size':32, 'num_of_epochs':50}
pendolum.trainModel(splits=[70,20,10], training_params=params)

## Neural network Predict
sample = pendolum.get_random_samples(dataset='pendulum_dataset', window=1)
print('Predicted omega: ', pendolum(sample, sampled=True))
print('True omega: ', sample['omega'])
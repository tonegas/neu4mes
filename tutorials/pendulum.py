
import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
from neu4mes.visualizer import MPLVisulizer

# Create neu4mes structure
pendolum = Neu4mes()

# Create neural model
theta = Input('theta')
T     = Input('torque')
theta_s = Input('theta_s')

# TODO the training stuck when add the parameter c
c = Parameter('c', dimensions=1, sw=1)
#lin_theta = Fir(theta.tw(1))
sin_theta = Fir(Sin(theta.tw(1)))+c
torque = Fir(T.tw(1))
out = Output('theta_pred', sin_theta+torque+c)

# Add the neural model to the neu4mes structure and neuralization of the model
pendolum.minimizeError('next-theta', theta_s.next(), out)
pendolum.addModel(out)
pendolum.neuralizeModel(0.05)

# Data load
data_struct = ['time','theta','theta_s','','','torque']
data_folder = './tutorials/datasets/pendulum/data/'
pendolum.loadData(name='pendulum_dataset', source=data_folder, format=data_struct, delimiter=';')

# Neural network train
params = {'learning_rate':0.001, 'train_batch_size':32, 'val_batch_size':16, 'test_batch_size':1}
pendolum.trainModel(splits=[70,20,10], training_params=params)

## Neural network Predict
sample = pendolum.get_random_samples(dataset='pendulum_dataset', window=1)
print('Prediction: ', pendolum(sample, sampled=True))
print('True theta: ', sample['theta_s'])
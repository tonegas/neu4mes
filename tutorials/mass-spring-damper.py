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

# List the relations of the model
displacement_x = Fir(parameter_init=init_negexp)
force_x = Fir(parameter_init=init_constant,parameter_init_params={'value':1})
displacement_dx = Fir(parameter_init=init_negexp)
force_dx = Fir(parameter_init=init_constant,parameter_init_params={'value':1})

# List the outputs of the model
next_x = Output('next_x', displacement_x(x.tw(0.2))+force_x(F.last()))
next_dx = Output('next_dx', displacement_dx(x.tw(0.2))+force_dx(F.last()))

# Add the neural models to the neu4mes structure
result_path = os.path.join(os.getcwd(), "results")
mass_spring_damper = Neu4mes(visualizer=MPLVisulizer(), workspace=result_path)
mass_spring_damper.addModel('position_model',next_x)
mass_spring_damper.addModel('velocity_model',next_dx)

# These functions are used to impose the minimization objectives.
# Here it is minimized the error between the future position of x get from the dataset x.z(-1)
# and the estimator designed useing the neural network. The miniminzation is imposed via MSE error.
mass_spring_damper.addMinimize('position', x.next(), next_x, 'mse')
# The second minimization is between the velocity get from the dataset and the velocity estimator.
mass_spring_damper.addMinimize('velocity', dx.next(), next_dx, 'mse')

# Nauralize the model and gatting the neural network. The sampling time depends on the datasets.
mass_spring_damper.neuralizeModel(sample_time = 0.05) # The sampling time depends on the dataset

# Data load
data_struct = ['time','x','dx','F']
data_folder = './tutorials/datasets/mass-spring-damper/data/'
mass_spring_damper.loadData(name='mass_spring_dataset', source=data_folder, format=data_struct, delimiter=';')

#Neural network train not reccurent training
params = {'num_of_epochs': 100,
          'train_batch_size': 128, 
          'val_batch_size':128, 
          'test_batch_size':1, 
          'learning_rate':0.001}
mass_spring_damper.trainModel(splits=[70,20,10], training_params = params)

## Make predictions
sample = mass_spring_damper.get_random_samples(dataset='mass_spring_dataset', window=200)
result = mass_spring_damper(sample, sampled=True)
true_position = [pos[-1].item() for pos in sample['x']]
true_velocity = [vel.squeeze().item() for vel in sample['dx']]
pred_position = result['next_x']
pred_velocity = result['next_dx']

## Plot predictions
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
axes[0].plot(true_position, label='True Position', marker='o')
axes[0].plot(pred_position, label='Predicted Position', marker='x')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Position')
axes[0].set_title('Position Inference')
axes[1].plot(true_velocity, label='True Velocity', marker='o')
axes[1].plot(pred_velocity, label='Predicted Velocity', marker='x')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Velocity')
axes[1].set_title('Velocity Inference')
plt.tight_layout()
plt.show()

## Recurrent training
params = {'num_of_epochs': 20,
          'train_batch_size': 128, 
          'val_batch_size':128, 
          'test_batch_size':1, 
          'learning_rate':0.0001}
mass_spring_damper.trainModel(splits=[70,20,10], training_params = params, close_loop={'x':'next_x'}, prediction_samples=10)


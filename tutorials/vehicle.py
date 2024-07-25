import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *

# Create neu4mes structure
vehicle = Neu4mes(visualizer=MPLVisulizer())

#
n  = 25
na = 21

# Create neural model
velocity = Input('vel')
brake = Input('brk')
gear = Input('gear')
torque = Input('trq')
altitude = Input('alt',dimensions=na)
acc = Input('acc')

air_drag_force = Fir(velocity.last()**2)
breaking_force = -Relu(Fir(brake.sw(n)))
gravity_force = Linear(altitude.last())
fuzzi_gear = Fuzzify(6, range=[2,7], functions='Rectangular')(gear.last())
local_model = LocalModel(input_function=Fir)
engine_force = local_model(torque.sw(n), fuzzi_gear)

out = Output('accelleration', air_drag_force+breaking_force+gravity_force+engine_force)

# Add the neural model to the neu4mes structure and neuralization of the model
vehicle.addModel(out)
vehicle.minimizeError('acc_error', acc.next(), out)
vehicle.neuralizeModel()

# Data load
data_struct = ['vel','trq','brk','gear','alt','acc']
data_folder = './tutorials/datasets/vehicle_data/trainingset'
vehicle.loadData(name='trainingset', source=data_folder, format=data_struct, skiplines=25)
data_folder = './tutorials/datasets/vehicle_data/validationset'
vehicle.loadData(name='validationset', source=data_folder, format=data_struct, skiplines=25)

# Neural network train
vehicle.trainModel(train_dataset='trainingset',validation_dataset='validationset', training_params={'num_of_epochs':500, 'val_batch_size':1000, 'train_batch_size':1000, 'learning_rate':0.001})
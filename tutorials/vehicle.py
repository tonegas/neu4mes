import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *

# Create neu4mes structure
vehicle = Neu4mes()

# Create neural model
velocity = Input('vel')
brake = Input('brk')
gear = Input('gear')
torque = Input('trq')
altitude = Input('h')
acc = Input('acc')

air_drag = Fir(velocity.last()**2)
breaking = Relu(Fir(brake.tw([-2,0])))
slope = Fir(altitude.tw([-2,2], offset=0))
fuzzi_gear = Fuzzify(8, range=[1,8], functions='Triangular')(gear.last())
local_model = LocalModel(input_function=Fir(output_dimension=8), output_function=Fir(output_dimension=1))
engine_torque = local_model(torque.tw([-2,0]), activations=fuzzi_gear)

out = Output('accelleration', air_drag+breaking+slope+engine_torque)

# Add the neural model to the neu4mes structure and neuralization of the model
vehicle.minimizeError('acc_error', acc.last(), out)
vehicle.addModel(out)
vehicle.neuralizeModel(0.1)

# Data load
data_struct = ['vel','trq','brk','gear','','h','acc','','']
data_folder = './examples/datasets/vehicle_data'
vehicle.loadData(name='vehicle_dataset', source=data_folder, format=data_struct)

# Neural network train
vehicle.trainModel(splits=[70,20,10], training_params={'num_of_epochs':50, 'train_batch_size':128, 'val_batch_size':128, 'test_batch_size':128})
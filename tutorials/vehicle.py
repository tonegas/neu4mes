import sys
import os
# append a new directory to sys.path
print(os.getcwd())
sys.path.append(os.getcwd())

from neu4mes import *

# Create neu4mes structure
workspace = os.path.join(os.getcwd(), "results")
vehicle = Neu4mes(visualizer=MPLVisulizer(), seed=0, workspace=workspace)

# Dimensions of the layers
n  = 25
na = 21

#Create neural model inputs
velocity = Input('vel')
brake = Input('brk')
gear = Input('gear')
torque = Input('trq')
altitude = Input('alt',dimensions=na)
acc = Input('acc')

# Create neural network relations
air_drag_force = Linear(b=True)(velocity.last()**2)
breaking_force = -Relu(Fir(parameter_init = init_negexp, parameter_init_params={'size_index':0, 'first_value':0.005, 'lambda':3})(brake.sw(n)))
gravity_force = Linear(W_init=init_lin, W_init_params={'size_index':1, 'first_value':-1, 'last_value':1})(altitude.last())
fuzzi_gear = Fuzzify(6, range=[2,7], functions='Rectangular')(gear.last())
local_model = LocalModel(input_function=lambda: Fir(parameter_init = init_negexp))
engine_force = local_model(torque.sw(n), fuzzi_gear)

# Create neural network output
out = Output('accelleration', air_drag_force+breaking_force+gravity_force+engine_force)

# Add the neural model to the neu4mes structure and neuralization of the model
vehicle.addModel('acc',[out])
vehicle.addMinimize('acc_error', acc.last(), out, loss_function='rmse')
vehicle.neuralizeModel(0.05)
vehicle.exportJSON()

# Load the training and the validation dataset
data_struct = ['vel','trq','brk','gear','alt','acc']
data_folder = './tutorials/datasets/vehicle_data/trainingset'
vehicle.loadData(name='trainingset', source=data_folder, format=data_struct, skiplines=1)
data_folder = './tutorials/datasets/vehicle_data/validationset'
vehicle.loadData(name='validationset', source=data_folder, format=data_struct, skiplines=1)

# Filter the data
def filter_function(sample):
    return np.all(sample['vel'] >= 1.).tolist()
vehicle.filterData(filter_function = filter_function, dataset_name = 'trainingset')

# Neural network train
params = {'num_of_epochs':300, 
          'val_batch_size':128, 
          'train_batch_size':128, 
          'learning_rate':0.00003}
vehicle.trainModel(train_dataset='trainingset', validation_dataset='validationset', training_params=params)

## Neural network Predict
sample = vehicle.get_random_samples(dataset='validationset', window=1)
result = vehicle(sample, sampled=True)
print('Predicted accelleration: ', result['accelleration'])
print('True accelleration: ', sample['acc'])

file_name = vehicle.exportTracer()
#vehicle.exportONNX(file_name)

## Import the tracer model
#vehicle.importTracer(file_name=file_name)
#result = vehicle(sample, sampled=True)
#print('Predicted accelleration: ', result['accelleration'])
#print('True accelleration: ', sample['acc'])

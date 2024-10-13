import sys
import os
# append a new directory to sys.path
print(os.getcwd())
sys.path.append(os.getcwd())

from neu4mes import *

# Create neu4mes structure
vehicle = Neu4mes(visualizer=MPLVisulizer(),seed=0)

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
breaking_force = -Relu(Fir(parameter_init = init_negexp, parameter_init_params={'size_index':0, 'first_value':0.002, 'lambda':3})(brake.sw(n)))
gravity_force = Linear(W_init=init_constant, W_init_params={'value':0}, dropout=0.1, W='gravity')(altitude.last())
fuzzi_gear = Fuzzify(6, range=[2,7], functions='Rectangular')(gear.last())
local_model = LocalModel(input_function=lambda: Fir(parameter_init = init_negexp, parameter_init_params={'size_index':0, 'first_value':0.002, 'lambda':3}))
engine_force = local_model(torque.sw(n), fuzzi_gear)

# Create neural network output
out = Output('accelleration', air_drag_force+breaking_force+gravity_force+engine_force)

# Add the neural model to the neu4mes structure and neuralization of the model
vehicle.addModel('acc',[out])
vehicle.addMinimize('acc_error', acc.last(), out, loss_function='rmse')
vehicle.neuralizeModel(0.05)
vehicle.neuralizeModel(0.05)

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
optimizer_params = [{'params':'gravity','weight_decay': 0.1}]
optimizer_defaults = {'weight_decay': 0.00001}
training_params = {'num_of_epochs':200, 'val_batch_size':128, 'train_batch_size':128, 'lr':0.00003}
vehicle.trainModel(train_dataset='trainingset', validation_dataset='validationset', shuffle_data=True, add_optimizer_params=optimizer_params, add_optimizer_defaults=optimizer_defaults, training_params=training_params)

## Neural network Predict
# sample = vehicle.getSamples(dataset='validationset', window=5)
# result = vehicle(sample, sampled=True)
# print(result)
# result = vehicle(sample)
# print(result)
# start = time.time()
# for _ in range(10000):
#     result = vehicle(sample, sampled=True)
# print('Inference Time: ', time.time() - start)
# print('Predicted accelleration: ', result['accelleration'])
# print('True accelleration: ', sample['acc'])

# python, python_onnx, onnx = vehicle.exportTracer()
# #vehicle.import_onnx(onnx)
# #vehicle.exportONNX(file_name)
#
# ## Import the tracer model
# vehicle.importTracer(file_path=python)
# start = time.time()
# for _ in range(10000):
#     result = vehicle(sample, sampled=True)
# print('Inference Time: ', time.time() - start)
# print('Predicted accelleration: ', result['accelleration'])
# print('True accelleration: ', sample['acc'])
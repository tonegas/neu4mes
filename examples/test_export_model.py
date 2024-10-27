import sys
import os
import torch
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *

# Create neu4mes structure
#result_path = os.path.join(os.getcwd(), "results")
result_path = './results'
test = Neu4mes(seed=42, workspace=result_path)

x = Input('x')
y = Input('y')
z = Input('z')

## create the relations
def myFun(K1,p1,p2):
    return K1*p1*p2

K_x = Parameter('k_x', dimensions=1, tw=1)
K_y = Parameter('k_y', dimensions=1, tw=1)
w = Parameter('w', dimensions=1, tw=1)
t = Parameter('t', dimensions=1, tw=1)
c_v = Constant('c_v', tw=1, values=[[1],[2]])
c = 5
w_5 = Parameter('w_5', dimensions=1, tw=5)
t_5 = Parameter('t_5', dimensions=1, tw=5)
c_5 = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
parfun_x = ParamFun(myFun, parameters = [K_x], constants=[c_v])
parfun_y = ParamFun(myFun, parameters = [K_y])
parfun_z = ParamFun(myFun)
fir_w = Fir(parameter=w_5)(x.tw(5))
fir_t = Fir(parameter=t_5)(y.tw(5))
time_part = TimePart(x.tw(5),i=1,j=3)
sample_select = SampleSelect(x.sw(5),i=1)

def fuzzyfun(x):
    return torch.tan(x)
fuzzy = Fuzzify(output_dimension=4, range=[0,4], functions=fuzzyfun)(x.tw(1))

out = Output('out', Fir(parfun_x(x.tw(1))+parfun_y(y.tw(1),c_v)))
#out = Output('out', Fir(parfun_x(x.tw(1))+parfun_y(y.tw(1),c_v)+parfun_z(x.tw(5),t_5,c_5)))
out2 = Output('out2', Add(w,x.tw(1))+Add(t,y.tw(1))+Add(w,c))
out3 = Output('out3', Add(fir_w, fir_t))
out4 = Output('out4', Linear(output_dimension=1)(fuzzy))
out5 = Output('out5', Fir(time_part)+Fir(sample_select))
out6 = Output('out6', LocalModel(output_function = Fir())(x.tw(1),fuzzy))

test.addModel('modelA', out)
test.addModel('modelB',[out2,out3,out4])
test.addModel('modelC',[out4,out5,out6])
test.addMinimize('error1', x.last(), out)
test.addMinimize('error2', y.last(), out3, loss_function='rmse')
test.addMinimize('error3', z.sw([-5,-4]), out6, loss_function='rmse')
test.neuralizeModel(0.5)

print("-----------------------------------EXAMPLE 1------------------------------------")
# Export torch file .pt
# Save torch model and load it
old_out = test({'x':[1,2,3,4,5,6,7,8,9,10],'y':[2,3,4,5,6,7,8,9,10,11]})
test.exporter.saveTorchModel()
test.neuralizeModel(0.5, clear_model=True)
# The new_out is different from the old_out because the model is cleared
new_out = test({'x':[1,2,3,4,5,6,7,8,9,10],'y':[2,3,4,5,6,7,8,9,10,11]})
# The new_out_after_load is the same as the old_out because the model is loaded with the same parameters
test.exporter.loadTorchModel()
new_out_after_load = test({'x':[1,2,3,4,5,6,7,8,9,10],'y':[2,3,4,5,6,7,8,9,10,11]})
print('old_out: ', old_out)
print('new_out: ', new_out)
print('new_out_after_load: ', new_out_after_load)
print(f'the output are equal: {old_out == new_out_after_load}')

try:
    # The model can't be loaded on a new neu4mes obj because
    # the neu4mes model is needed for loading the torch model
    test2 = Neu4mes(seed=43, workspace=result_path)
    # You need aneuralized model to load a torch model
    test2.exporter.loadTorchModel()
except Exception as e:
    print(f"{e}")

print("-----------------------------------EXAMPLE 2------------------------------------")
# Export json of neu4mes model
# Save a untrained neu4mes json model and load it
# the new_out and new_out_after_load are different because the model saved model is not trained
test.saveModel() # Save a model without parameter values
test.neuralizeModel(clear_model = True) # Create a new torch model
new_out = test({'x':[1,2,3,4,5,6,7,8,9,10],'y':[2,3,4,5,6,7,8,9,10,11]})
test.loadModel() # Load the neu4mes model without parameter values
# Use the preloaded torch model for inference
new_out_after_load = test({'x':[1,2,3,4,5,6,7,8,9,10],'y':[2,3,4,5,6,7,8,9,10,11]})
print('old_out: ', old_out)
print('new_out: ', new_out)
print('new_out_after_load: ', new_out_after_load)

print("-----------------------------------EXAMPLE 3------------------------------------")
# Export json of neu4mes model with parameter valuess
# The old_out is the same as the new_out_after_load because the model is loaded with the same parameters
old_out = new_out_after_load
test.neuralizeModel() # Load the parameter from torch model to neu4mes model json
test.saveModel() # Save the model with and without parameter values
test.neuralizeModel(clear_model=True) # Create a new torch model
new_out = test({'x':[1,2,3,4,5,6,7,8,9,10],'y':[2,3,4,5,6,7,8,9,10,11]})
test.loadModel() # Load the neu4mes model with parameter values
new_out_after_load = test({'x':[1,2,3,4,5,6,7,8,9,10],'y':[2,3,4,5,6,7,8,9,10,11]})
print('old_out: ', old_out)
print('new_out: ', new_out)
print('new_out_after_load: ', new_out_after_load)
print(f'the output are equal: {old_out == new_out_after_load}')

print("-----------------------------------EXAMPLE 4------------------------------------")
# Import neu4mes json model in a new object
test2 = Neu4mes(seed=43, workspace=test.getWorkspace())
test2.loadModel() # Load the neu4mes model with parameter values
new_model_out_after_load = test2({'x':[1,2,3,4,5,6,7,8,9,10],'y':[2,3,4,5,6,7,8,9,10,11]})
print('new_out_after_load: ', new_out_after_load)
print('new_model_out_after_load: ', new_model_out_after_load)
print(f'the output are equal: {old_out == new_model_out_after_load}')

print("-----------------------------------EXAMPLE 5------------------------------------")
# Export and import of a torch script .py
# The old_out is the same as the new_out_after_load because the model is loaded with the same parameters
test.exportPythonModel() # Export the trace model
test.neuralizeModel(clear_model=True) # Create a new torch model
new_out = test({'x':[1,2,3,4,5,6,7,8,9,10],'y':[2,3,4,5,6,7,8,9,10,11]})
test.importPythonModel() # Import the tracer model
# Perform inference with the imported tracer model
new_out_after_load = test({'x':[1,2,3,4,5,6,7,8,9,10],'y':[2,3,4,5,6,7,8,9,10,11]})
print('old_out: ', old_out)
print('new_out: ', new_out)
print('new_out_after_load: ', new_out_after_load)
print(f'the output are equal: {old_out == new_out_after_load}')

print("-----------------------------------EXAMPLE 6------------------------------------")
# Import of a torch script .py
test2 = Neu4mes(seed=43, workspace=test.getWorkspace())
test2.importPythonModel() # Load the neu4mes model with parameter values
new_out_after_load = test2({'x':[1,2,3,4,5,6,7,8,9,10],'y':[2,3,4,5,6,7,8,9,10,11]})
print(f'the output are equal: {old_out == new_out_after_load}')

print("-----------------------------------EXAMPLE 7------------------------------------")
# Perform training on an imported tracer model
data_x = np.arange(0.0,1,0.1)
data_y = np.arange(0.0,1,0.1)
a,b = -1.0, 2.0
dataset = {'x': data_x, 'y': data_y, 'z':a*data_x+b*data_y}
params = {'num_of_epochs': 1, 'lr': 0.01}
test.loadData(name='dataset', source=dataset) # Create the dataset
test.trainModel(optimizer='SGD',training_params=params) # Train the traced model
new_out_after_train = test({'x':[1,2,3,4,5,6,7,8,9,10],'y':[2,3,4,5,6,7,8,9,10,11]})
print(test.model)
print('new_out_after_load: ', new_out_after_load)
print('new_out_after_train: ', new_out_after_train)

print("-----------------------------------EXAMPLE 8------------------------------------")
# Perform training on an imported new tracer model
test2.loadData(name='dataset', source=dataset) # Create the dataset
test2.trainModel(optimizer='SGD',training_params=params) # Train the traced model
new_out_after_trai_new = test({'x':[1,2,3,4,5,6,7,8,9,10],'y':[2,3,4,5,6,7,8,9,10,11]})
print('new_out_after_train: ', new_out_after_train)
print('new_out_after_trai_new: ', new_out_after_trai_new)
print(f'the output are equal: {new_out_after_train == new_out_after_trai_new}')

print("-----------------------------------EXAMPLE 9------------------------------------")
# Export the all models in onnx format
test.exportONNX(['x','y'],['out','out2','out3','out4','out5','out6']) # Export the onnx model

print("-----------------------------------EXAMPLE 10-----------------------------------")
# Export only the modelB in onnx format
test.exportONNX(['x','y'],['out3','out4','out2'], ['modelB']) # Export the onnx model

print("-----------------------------------EXAMPLE 11-----------------------------------")
# Export of the report of training and results performances
test.exportReport()
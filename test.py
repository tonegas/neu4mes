from neu4mes import *

in1 = Input('in1', values=[2,3,4])
in2 = Input('in2')
rel = LocalModel(in2.tw(1), in1)
fun = Output(in2.z(-1),rel)

test = Neu4mes(verbose=True)
test.addModel(fun)
test.neuralizeModel(0.5)

test_layer = Model(inputs=[test.inputs_for_model['in1']], outputs=test.inputs[('in1', 1)])
print(test_layer.predict([[2]]))
print(test_layer.predict([[3]]))
print(test_layer.predict([[4]]))

test_layer = Model(inputs=[test.inputs_for_model['in2'],test.inputs_for_model['in1']], outputs=test.outputs['in2__-z1'])
weights = test_layer.get_weights()
print(weights[0].shape) #
print(weights[0]) #
print(test_layer.predict([np.array([[0,1]]),np.array([[2]])]))
print(weights[0][1][0])
print(test_layer.predict([np.array([[0,1]]),np.array([[3]])]))
print(test_layer.predict([np.array([[0,1]]),np.array([[4]])]))

# test_layer = Model(inputs=test.inputs_for_model['in2'], outputs=test.inputs[('in2',5)])
# print(test_layer.predict([[0,1,2,3,4,5,6]]))


# data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time','']
# data_folder = './tests/data/'
# test.loadData(data_struct, folder = data_folder, skiplines = 4)

print(test.inout_asarray)
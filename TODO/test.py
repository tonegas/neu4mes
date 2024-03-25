from neu4mes import *

in1 = Input('in1')
in2 = Input('in2')
output = Input('out')
rel = Linear(in2.tw(1))+Linear(in1)
fun = Output(in2.z(-1),rel)

# input1 = Input('in1')
# output = Input('out')
# rel1 = Linear(input1.tw(0.05))
# fun = Output(output.z(-1),rel1)

test = Neu4mes()
test.addModel(fun)
test.neuralizeModel(0.5, prediction_window = 0.5)

data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time','']
data_folder = '../tests/data/'
test.loadData(data_struct, folder = data_folder, skiplines = 4)

training_params = {}
training_params['batch_size'] = 5
training_params['learning_rate'] = 0.1
training_params['num_of_epochs'] = 5
training_params['rnn_learning_rate'] = 0.1
training_params['rnn_num_of_epochs'] = 5

test.trainModel(states = [fun], training_params = training_params, test_percentage = 50,  show_results = True)
#test.trainRecurrentModel(states = [fun], training_params = training_params, test_percentage = 50,  show_results = True)

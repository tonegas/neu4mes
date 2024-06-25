import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *

## TEST JSON
'''
inin = Input('in').last()
par = Parameter('par',sw=1)
inin4 = Input('in',dimensions=4).last()
par4 = Parameter('par4',dimensions=4,sw=1)
add = inin+par+5.2
sub = inin-par-5.2
mul = inin*par*5.2
div = inin/par/5.2
pow = inin**par**5.2
sin = Sin(5.2)
cos = Cos(par)+Cos(5.2)
tan = Tan(par)+Tan(5.2)
relu = Relu(par)+Relu(5.2)
tanh = Tanh(par)+Tanh(5.2)

add4 = inin4+par4+5.2
sub4 = inin4-par4-5.2
mul4 = inin4*par4*5.2
div4 = inin4/par4/5.2
pow4 = inin4**par4**5.2
sin4 = Sin(par4)+Sin(5.2)
cos4 = Cos(par4)+Cos(5.2)
tan4 = Tan(par4)+Tan(5.2)
relu4 = Relu(par4)+Relu(5.2)
tanh4 = Tanh(par4)+Tanh(5.2)
out = Output('out',add+sub+mul+div+pow+Linear(add4+sub4+mul4+div4+pow4)+sin+cos+tan+relu+tanh+Linear(sin4+cos4+tan4+relu4+tanh4))
test = Neu4mes()
test.addModel(out)
test.neuralizeModel()
'''

## TEST TRAIN
x = Input('x', dimensions=1)
y = Input('y', dimensions=4)
x_label = Input('x_label', dimensions=1)
y_label = Input('y_label', dimensions=4)

k = Parameter('k', dimensions=1, sw=1)
w = Parameter('w', dimensions=4, sw=1)

add = x.last()+k+5.0
add_vectorial = y.last()+w+5.0

out = Output('out', add)
out_vectorial = Output('out_vec', add_vectorial)

test = Neu4mes()
test.addModel(out)
test.addModel(out_vectorial)
test.minimizeError('error', out, x_label.last())
test.minimizeError('error_vectorial', out_vectorial, y_label.last())
test.neuralizeModel()

data_x = np.random.random((200, 1, 1))
data_x_label = 2*data_x + 5
data_y = np.random.random((200, 1, 4))
data_y_label = 3*data_y - 4
test.loadData(source={'x':data_x, 'y':data_y, 'x_label':data_x_label, 'y_label':data_y_label})

sample = test.get_random_samples(window=1)
print('random sample: ', sample)
print('prediction before train', test(sample, sampled=True))

test.trainModel(test_percentage=10, training_params={'num_of_epochs':100, 'train_batch_size':4, 'test_batch_size':4})

print('prediction after train', test(sample, sampled=True))
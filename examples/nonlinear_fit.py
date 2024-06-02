import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
from neu4mes.visualizer import StandardVisualizer

x = Input('x')
target_y = Input('target_y')

a = Parameter('a')
b = Parameter('b')
c = Parameter('c')
d = Parameter('d')
e = Parameter('e')

# Linear function
def linear_fun1(x,a,b):
    return x*a+b

def linear_fun2(x,a,b,c):
    return x**2*a+x*b+c

def linear_fun3(x,a,b,c,d):
    return x**3*a+x**2*b+x*c+d

def linear_fun4(x,a,b,c,d,e):
    return x**4*a+x**3*b+x**2*c+x*d+e

y1 = Output('y1', ParamFun(linear_fun1)(x)) # TODO controllare che tipo di problema quando uso il nome dell'input uguale al nome dell'output
y2 = Output('y2', ParamFun(linear_fun2)(x))
y3 = Output('y3', ParamFun(linear_fun3)(x))
y4 = Output('y4', ParamFun(linear_fun4)(x))

data_x = np.random.rand(250)*20-10
data_a = 2
data_b = -3
data_c = 2
dataset = {'x': data_x, 'target_y': linear_fun2(data_x,data_a,data_b,data_c)}

opt_fun = Neu4mes(verbose = True)
#opt_fun.addModel(y)
opt_fun.minimizeError('x', target_y, y1, 'mse') # la stringa di input ne indica il nome
opt_fun.minimizeError('x^2', target_y, y2, 'mse')
opt_fun.minimizeError('x^3', target_y, y3, 'mse')
opt_fun.minimizeError('x^4', target_y, y4, 'mse')

opt_fun.neuralizeModel() # rimuovere il sample rate se non serve
opt_fun.loadData(dataset)  # Caricamento del dataset tramite un dizionario crafted

opt_fun.trainModel(test_percentage = 50,training_params={'num_of_epochs':500})

random_sample = opt_fun.get_random_samples()
pred = opt_fun(random_sample)
#print('prediction variables: ', pred.keys())
#print('Prediction y: ', pred['y'])
#print('label: ', pred['out_target_y'])
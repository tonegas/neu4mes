import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
from neu4mes.visualizer import StandardVisualizer

x = Input('x')
target_y = Input('target_y')

# Linear function
def linear_fun(x,a,b):
    return x*a+b

parfun = ParamFun(linear_fun)

y = Output('y', parfun(x)) # TODO controllare che tipo di problema quando uso il nome dell'input uguale al nome dell'output

data_x = np.random.rand(50)*20-10
data_a = 2
data_b = -3
dataset = {'x': data_x, 'target_y': linear_fun(data_x,data_a,data_b)}

opt_fun = Neu4mes(verbose = True,  visualizer = StandardVisualizer())
opt_fun.addModel(y)
opt_fun.minimizeError('out', target_y, y, 'mse') # la stringa di input ne indica il nome

opt_fun.neuralizeModel() # rimuovere il sample rate se non serve
opt_fun.loadData(dataset)  # Caricamento del dataset tramite un dizionario crafted

opt_fun.trainModel(test_percentage = 10, show_results = False)

random_sample = opt_fun.get_random_samples()
pred = opt_fun(random_sample)
print('prediction variables: ', pred.keys())
print('Prediction y: ', pred['y'])
print('label: ', pred['out_target_y'])

## TODO Recurrent training
## TODO test recurrent
import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
from neu4mes.visualizer import StandardVisualizer

x = Input('x')
target_y = Input('target_y')

# Linear function
def linear_fun(x,a,b,c):
    return x**2*a+x*b+c

parfun = ParamFun(linear_fun)

y = Output('y', parfun(x)) # TODO controllare che tipo di problema quando uso il nome dell'input uguale al nome dell'output

data_x = np.random.rand(250)*20-10
data_a = 2
data_b = -3
data_c = 2
dataset = {'x': data_x, 'target_y': linear_fun(data_x,data_a,data_b,data_c)}

opt_fun = Neu4mes(verbose = True,  visualizer = StandardVisualizer())
opt_fun.addModel(y)
opt_fun.minimizeError('out', target_y, y, 'mse') # TODO mettere una stringa di input che indica il nome


opt_fun.neuralizeModel() # TODO rimuovere il sample rate se non serve
opt_fun.loadData(dataset)  # TODO Caricamento del dataset tramite un dizionario fatto come per la predict

opt_fun.trainModel(test_percentage = 10, show_results = False)

print(opt_fun(dataset))
print('Prediction only for y: ', opt_fun(dataset)['y']) # TODO potrei mettere un input per ottenere direttamente quella variabile e non un dizionario


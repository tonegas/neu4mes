from neu4mes import *
from neu4mes.visualizer import StandardVisualizer

x = Input('x')
target_y = Input('target_y')

# Linear function
def linear_fun(x,a,b):
    return x*a+b

parfun = ParamFun(linear_fun)

y = Output('y', parfun(x)) # TODO controllare che tipo di problema quando uso il nome dell'input uguale al nome dell'output

data_x = np.random.rand(1,200)*20-10
data_a = 2
data_b = -3
dataset = {'x': data_x, 'target_y': linear_fun(data_x,data_a,data_b)}

opt_fun = Neu4mes(verbose = True,  visualizer = StandardVisualizer())
opt_fun.addModel(y)
opt_fun.minimizeError('ciao', target_y, y, 'mse') # TODO mettere una stringa di input che indica il nome


opt_fun.neuralizeModel() # TODO rimuovere il sample rate se non serve
print(opt_fun(dataset))   # TODO dovrebbe essere un vettore di uscita dovrebbe ignorare le variabi in più
opt_fun.loadData(dataset) # TODO Caricamento del dataset tramite un dizionario fatto come per la predict

opt_fun.trainModel(test_percentage = 10, show_results = True)

print(opt_fun(dataset)['y']) # TODO potrei mettere un input per ottenere direttamente quella variabile e non un dizionario


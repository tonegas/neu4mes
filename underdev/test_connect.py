import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *

class FunctionVisualizer(TextVisualizer):
    def showResults(self):
        super().showResults()
        import matplotlib.pyplot as plt
        data_x = np.arange(-30,30,0.1)
        plt.title('Function Data')
        plt.plot(data_x, fun(data_x, data_a, data_b, data_c, data_d), label=f'target')
        for key in self.n4m.model_def['Outputs'].keys():
            plt.plot(data_x, test({'x': data_x})[key],  '-.', label=key)

        plt.grid(True)
        plt.legend(loc='best')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

x = Input('x')
y = Input('y')
def fun(x,a,b,c,d):
    return a*x**3+b*x**2+c*x+d

out_fun = ParamFun(fun)(x.last())
out_y = Output('out_y',out_fun)
test = Neu4mes(visualizer=FunctionVisualizer())
test.addModel('in_model',out_y)
test.addMinimize('err_in',y.last(),out_y)
test.neuralizeModel()

# Costruisco il dataset
data_x = np.arange(-5,5,0.01)
data_a = 2
data_b = -3
data_c = 4
data_d = 5
dataset = {'x': data_x, 'y': fun(data_x,data_a,data_b,data_c,data_d)}

test.loadData('data',dataset)
test.trainModel(training_params={"num_of_epochs":15})

# Connect caso 1
# Voglio invertire l'uscita
y_in = State('y_in')
yy_in = State('yy_in')
out_fun_out = ParamFun(fun)(y_in.last())
out_x = Output('out_x',out_fun_out)

test.addModel('out_model',out_x)
test.addMinimize('err_out',x.last(),out_x)

test.addConnect(out_fun,y_in)
test.neuralizeModel()

test.addConnect(out_y,[y_in,yy_in])
test.neuralizeModel()

# Connect caso 2
# Voglio creare una finestra di una relazione
# Modalita estesa

fir_in = State('fir_in')
out_fir = Fir(fir_in.tw(5))
out_fir_connect = Connect(out_fun, fir_in) # Questa è la stessa funzione
out_y_fir = Output('out_y_fir',out_fir)
test.addModel('out_fir_model',[out_y_fir,out_fir_connect])
test.neuralizeModel()

out_fir = Fir(out_fun.tw(5))
out_y_fir = Output('out_y_fir2',out_fir)
test.addModel('out_fir_model_2',out_y_fir)
test.neuralizeModel()

test.trainModel(training_params={"num_of_epochs":15})


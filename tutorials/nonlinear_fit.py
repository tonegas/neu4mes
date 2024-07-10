import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
from neu4mes.visualizer import MPLVisulizer

# Input of the function
x = Input('x')
# Output the function
target_y = Input('target_y')

# Linear function
def parametric_fun1(x,a,b):
    return x*a+b

# Quadratic function
def parametric_fun2(x,a,b,c):
    return x**2*a+x*b+c

# Cubic function
def parametric_fun3(x,a,b,c,d):
    return x**3*a+x**2*b+x*c+d

# x^4 function
def parametric_fun4(x,a,b,c,d,e):
    return x**4*a+x**3*b+x**2*c+x*d+e

# Create the parametric functions
y1 = Output('y1', ParamFun(parametric_fun1)(x.last()))
y2 = Output('y2', ParamFun(parametric_fun2)(x.last()))
y3 = Output('y3', ParamFun(parametric_fun3)(x.last()))
y4 = Output('y4', ParamFun(parametric_fun4)(x.last()))

# Create the target functions
data_x = np.random.rand(250)*20-10
data_a = 2
data_b = -3
#data_c = 2
dataset = {'x': data_x, 'target_y': parametric_fun1(data_x,data_a,data_b)}

# Create the neu4mes object
#opt_fun = Neu4mes(visualizer=MPLVisulizer())
opt_fun = Neu4mes()

# Create objectives of the minimization
opt_fun.minimizeError('x^1', target_y.last(), y1, 'mse')
opt_fun.minimizeError('x^2', target_y.last(), y2, 'mse')
opt_fun.minimizeError('x^3', target_y.last(), y3, 'mse')
opt_fun.minimizeError('x^4', target_y.last(), y4, 'mse')

# Neuralize the models
opt_fun.neuralizeModel()

# Load the dataset create with the target function
opt_fun.loadData('dataset', dataset)

# Train the models
opt_fun.trainModel(splits=[60,10,30], training_params={'num_of_epochs':250})


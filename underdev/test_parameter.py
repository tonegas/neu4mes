import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *

x = Input('x')
F = Input('F')

print("------------------------EXAMPLE 1------------------------")
# Example 1
# Create a parameter k of dimension 3
k = Parameter('k', dimensions=3, tw=4)
fir1 = Fir(3, parameter=k)
fir2 = Fir(3, parameter=k)
out = Output('out', fir1(x.tw(4))+fir2(F.tw(4)))
example = Neu4mes(verbose = True)
example.addModel(out)
example.neuralizeModel(0.05)
#

print("------------------------EXAMPLE 2------------------------")
# Example 2
# Create
g = Parameter('g', dimensions=3 )
t = Parameter('t', dimensions=3 )
def fun(x, k, t):
    return x*k*t
p = ParamFun(fun, output_dimension=3, parameters=[k,t])
out = Output('out', p(x.tw(1)))
example = Neu4mes(verbose = True)
example.addModel(out)
example.neuralizeModel(0.05)
#




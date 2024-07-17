import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *

x = Input('x')
F = Input('F')

print("------------------------EXAMPLE 1------------------------")
# Example 1
# Create a parameter k of dimension 3 and use this parameter in a Fir Layer.
# The two Fir have shared parameter
k = Parameter('k', dimensions=3, tw=4)
fir1 = Fir(parameter=k)
fir2 = Fir(3, parameter=k)
out = Output('out', fir1(x.tw(4))+fir2(F.tw(4)))
example = Neu4mes()
example.addModel(out)
example.neuralizeModel(1)
print(example({'x':[1,2,3,4],'F':[1,2,3,4]}))
#

print("------------------------EXAMPLE 2------------------------")
# Example 2
# Create a parameter with sample window equal to 5 and dimension equal to 3.
# The parameter is used inside a Fir
g = Parameter('g', dimensions=3 , sw=5)
fir = Fir(parameter=g)(x.sw(5))
out = Output('out', fir)
example = Neu4mes()
example.addModel(out)
example.neuralizeModel(0.05)
print(example({'x':[1,2,3,4,6]}))
#

print("------------------------EXAMPLE 3------------------------")
# Example 3
# Create two parameters and use them inside a parametric function the parameters are inizialized
g = Parameter('g', dimensions=3, values=[4,5,6])
t = Parameter('t', dimensions=3, values=[1,2,3])
def fun(x, k, t):
    import torch
    return x+(k+t)
p = ParamFun(fun, parameters=[g,t])
out = Output('out', p(x.tw(1)))
example = Neu4mes()
example.addModel(out)
example.neuralizeModel(0.5)
print(example({'x':[1,6,0]}))
#

print("------------------------EXAMPLE 4------------------------")
# Example 4
# Create two parameters fo initializa a Fir and a Linear the two parameters are initialized.
g = Parameter('g', dimensions=(4,1),values=[[[1],[2],[3],[4]]])
o = Parameter('o', sw=3, dimensions=2, values=[[2,3],[1,2],[1,2]])
x = Input('x', dimensions=4)
y = Input('y')
out = Output('out', Linear(W=g)(x.sw(3)))
out2 = Output('out2', Fir(parameter=o)(y.sw(3)))
example = Neu4mes()
example.addModel([out,out2])
example.neuralizeModel()
print('result: ', example({'x':[[1,2,4,4],[1,2,4,4],[1,2,4,4]],'y':[1,2,4,2]}))
#
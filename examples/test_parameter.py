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
fir1 = Fir(parameter=k)
fir2 = Fir(3, parameter=k)
out = Output('out', fir1(x.tw(4))+fir2(F.tw(4)))
example = Neu4mes()
example.addModel(out)
example.neuralizeModel(0.05)
#

print("------------------------EXAMPLE 2------------------------")
# Example 2
# Create
g = Parameter('g', dimensions=3 )
t = Parameter('t', dimensions=3 )
def fun(x, k, t):
    import torch
    return x+torch.transpose((k+t),0,1)
p = ParamFun(fun, parameters=[g,t])
out = Output('out', p(x.tw(1)))
example = Neu4mes()
example.addModel(out)
example.neuralizeModel(0.05)
#

print("------------------------EXAMPLE 3------------------------")
# Example 3
# Create
g = Parameter('g', dimensions=3 , sw=5)
def fun(x, g):
    import torch
    return torch.stack([x[:,5:9,0],x[:,5:9,0],x[:,5:9,0]],dim=2)*g[:,0]
p = ParamFun(fun, parameters=[g])
out = Output('out', p(x.tw(1)))
example = Neu4mes()
example.addModel(out)
example.neuralizeModel(0.05)
example({'x':[1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0]})

print("------------------------EXAMPLE 4------------------------")
# Example 4
# Create
g = Parameter('g',sw=1)
out = Output('out', x+g)
example = Neu4mes()
example.addModel(out)
example.neuralizeModel(0.5)
example({'x':[1,2]})

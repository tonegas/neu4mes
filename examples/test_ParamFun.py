import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *

x = Input('x')
F = Input('F')

print("------------------------EXAMPLE 1------------------------")
# Example 1 Parametric Function Basic
# This function has two parameters p1 and p2 of size 1 and two inputs K1 and K2
# The output size is user defined
# if it is not specified the output is expected to be 1
def myFun(K1,K2,p1,p2):
    import torch
    return p1*K1+p2*torch.sin(K2)

parfun = ParamFun(myFun)
out = Output('fun',parfun(x.last(),F.last()))
example = Neu4mes(visualizer=None)
example.addModel(out)
example.neuralizeModel()
print(example({'x':[1],'F':[1]}))
print(example({'x':[1,2],'F':[1,2]}))
#

print("------------------------EXAMPLE 2------------------------")
# Example 2
# I define the output of the function which is now 4 and then if the function does not output with size 4 I give an error
# also in this case I have two parameters p1 and p2
# in the function there is a product between a vector and a scalar and then a sum with a scalar
# the size of parameters p1 and p2 is 1
def myFun(K1,K2,p1,p2):
    import torch
    return torch.tensor([p1,p1,p1,p1])*K1+p2*torch.sin(K2)
parfun = ParamFun(myFun) # definisco una funzione scalare basata su myFun
out = Output('out',parfun(x.last(),F.last()))
example = Neu4mes(visualizer=None)
example.addModel(out)
example.neuralizeModel()
print(example({'x':[1],'F':[1]}))
print(example({'x':[1,2],'F':[1,2]}))
#

print("------------------------EXAMPLE 3------------------------")
# Example 2
# I define the output of the function which is now 4 and then if the function does not output with size 4 I give an error
# also in this case I have two parameters p1 and p2
# in the function there is a product between a vector and a scalar and then a sum with a scalar
# the size of parameters p1 and p2 is 1
def myFun(K1,K2,p1,p2):
    import torch
    return p1*K1+p2*torch.sin(K2)
parfun = ParamFun(myFun) # definisco una funzione scalare basata su myFun
out = Output('out',parfun(x.tw(2),F.tw(2)))
example = Neu4mes(visualizer=None)
example.addModel(out)
example.neuralizeModel(1)
print(example({'x':[1,1],'F':[1,1]}))
print(example({'x':[1,2,3],'F':[1,2,3]}))
#

print("------------------------EXAMPLE 4------------------------")
#Example 4
# This case I define the specific size of the parameters
# the first p1 is a 4 row column vector
# The output size of the function is 1
# in this case I make a dot product between the vector and p1 which is a vector [4,1]
# The time dimension of the output is not defined but depends on the input
# In the first call parfun(x.tw(1),F.tw(1)) the time output is a 1 sec window
# In the second call parfun(x,F) is an instant output
def myFun(K1,K2,p1):
    import torch
    return torch.stack([K1,2*K1,3*K1,4*K1],dim=2).squeeze(-1)*p1+K2
parfun = ParamFun(myFun,parameters_dimensions = {'p1':(1,4)})
out = Output('out',parfun(x.last(),F.last()))
example = Neu4mes(visualizer=None)
example.addModel(out)
example.neuralizeModel()
print(example({'x':[1],'F':[1]}))
#

print("------------------------EXAMPLE 5------------------------")
# Example 5
# This case I create a parameter that I pass to the parametric function
# The parametric function takes a parameter of size 4
# The function has three inputs, the first two are inputs and the second is a K parameter
# The function creates a tensor performs a dot product between input 1 and p1 (which is effectively K Parameter)
def myFun(K1,p1):
    return K1*p1
K = Parameter('k', dimensions =  1, tw = 1)
parfun = ParamFun(myFun, parameters = [K] )
out = Output('out',parfun(x.tw(1)))
example = Neu4mes(visualizer=None)
example.addModel(out)
example.neuralizeModel(0.25)
print(example({'x':[1,1,1,1],'F':[1,1,1,1]}))
#



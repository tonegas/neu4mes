import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())
from torch.fx import symbolic_trace

from neu4mes import *

x = Input('x')
F = Input('F')
'''
print("------------------------EXAMPLE 1------------------------")
# Example 1
# Here a fuzzify function is created with 5 membership functions in a range [1,5] of the input variable
fuz = Fuzzify(5,[1,5])
out = Output('out',fuz(x.last()))
example = Neu4mes()
example.addModel('out',out)
example.neuralizeModel()
print(example({'x':[2]}))
#

print("------------------------EXAMPLE 2------------------------")
# Example 2
# Here a fuzzify function is created with 5 membership functions in a range [1,5] of the input variable
# and triangular activation function
fuz = Fuzzify(5,[1,5], functions = 'Triangular')
out = Output('out',fuz(x.last()))
example = Neu4mes()
example.addModel('out',out)
example.neuralizeModel()
print(example({'x':[2]}))  ## should give [0, 1, 0, 0, 0]
print(example({'x':[2.5]})) ## should give [0, 0.5, 0.5, 0, 0]
print(example({'x':[3]})) ## should give [0, 0, 1, 0, 0]
#

print("------------------------EXAMPLE 3------------------------")
# Example 3
# Create 6 membership functions by dividing the range from 1, 6 with rectangular functions
# centers are in [1,2,3,4,5,6] functions are 1 wide except the first and last
# [-inf, 1.5] [1.5,2.5] [2.5,3.5] [3.5,4.5] [4.5,5.5] [5.5.inf]
fuz = Fuzzify(6,[1,6], functions = 'Rectangular')
out = Output('out',fuz(x.last()))
example = Neu4mes()
example.addModel('out',out)
example.neuralizeModel()
print(example({'x':[2]}))  ## should give [0, 1, 0, 0, 0, 0]
print(example({'x':[2.5]})) ## should give [0, 0, 1, 0, 0, 0]
print(example({'x':[3]})) ## should give [0, 0, 1, 0, 0, 0]
#

print("------------------------EXAMPLE 4------------------------")
# Example 4
# Create 10 membership functions by dividing the range from -5, 5 with custom function fun
# the centers are in [-5,-4,-3,-2,-1,0,1,2,3,4,5]
def fun(x):
    import torch
    return torch.tanh(x)
fuz = Fuzzify(output_dimension = 11, range = [-5,5], functions = fun)
out = Output('out',fuz(x.last()))
example = Neu4mes()
example.addModel('out',out)
example.neuralizeModel()
print(example({'x':[2.5]}))
print(example({'x':[0]})) ## should return 0 near the center, 0.99 near -5 and -0.99 near 5
#

print("------------------------EXAMPLE 5------------------------")
# Example 5
# Create 2 custom membership functions that are positioned at -1 and 5
def fun1(x):
    import torch
    return torch.sin(x)
def fun2(x):
    import torch
    return torch.cos(x)
fuz = Fuzzify(2,range=[-1,5],functions=[fun1,fun2])
out = Output('out',fuz(x.last()))
example = Neu4mes()
example.addModel('out',out)
example.neuralizeModel()
print(example({'x':[1]}))
print(example({'x':[2]}))
print(example({'x':[3]}))
print(example({'x':[4]}))
print(example({'x':[5]}))
#

print("------------------------EXAMPLE 6------------------------")
# Example 6
# Create 4 custom membership functions that are positioned at [-1,0,3,5]
import torch
def fun1(x):
    return torch.sin(x)
def fun2(x):
    return torch.cos(x)
fuz = Fuzzify(centers=[-1,0,3,5],functions=[fun1,fun2,fun1,fun2])
out = Output('out',fuz(x.last())+fuz(F.last()))
example = Neu4mes()
example.addModel('out',out)
example.neuralizeModel()
print(example({'x':[-1,0], 'F':[-1,0]}))
print(example({'x':[0,3], 'F':[0,3]}))
print(example({'x':[3,5], 'F':[3,5]}))
#

print("------------------------EXAMPLE 7------------------------")
# Example 7
## In this example we create two custom functions with 4 centers,
# the first and third center will use the first activation function
# while the second and forth center will use the second activation function
import torch
def fun1(x):
    return torch.cos(x)
def fun2(x):
    return torch.sin(x)
fuz = Fuzzify(centers=[-9.0,-3.0,3.0,9.0],functions=[fun1,fun2])
out = Output('out', fuz(x.last()))
example = Neu4mes()
example.addModel('out',out)
example.neuralizeModel()
print(example({'x':[-9,-3.0,3.0,9.0]}))
#

print("------------------------EXAMPLE 8------------------------")
# Example 8
## In this example we create one custom tangent function with 4 centers,
def fun(x):
    import torch
    return torch.tan(x)

fuz = Fuzzify(range=[-15.0, 15.0], centers=[-5, -2, 1, 4], functions=fun)
out = Output('out', fuz(x.last()))
example = Neu4mes()
example.addModel('out',out)
example.neuralizeModel()
print(example({'x':[-9,-3.0,3.0,9.0]}))
#
'''
print("------------------------EXAMPLE 9------------------------")
# Example 3
# Create 6 membership functions by dividing the range from 1, 6 with rectangular functions
# centers are in [1,2,3,4,5,6] functions are 1 wide except the first and last
# [-inf, 1.5] [1.5,2.5] [2.5,3.5] [3.5,4.5] [4.5,5.5] [5.5.inf]
fuz = Fuzzify(6,[1,6], functions = 'Triangular')
out = Output('out',fuz(x.last()))
result_path = os.path.join(os.getcwd(), "results", "example1")
example = Neu4mes(folder=result_path)
example.addModel('out',out)
example.neuralizeModel()
print(example({'x':[2]}))  ## should give [0, 1, 0, 0, 0, 0]
print(example({'x':[2.5]})) ## should give [0, 0, 1, 0, 0, 0]
print(example({'x':[3]})) ## should give [0, 0, 1, 0, 0, 0]
trace = symbolic_trace(example.model)
#print(dir(trace))
#attributes = [line.replace('self.', '') for line in trace.code.split() if 'self.' in line]
#print(attributes)
#for i in attributes:
#    print(f'{i} : {getattr(trace, i)}')

file_name = example.exportTracer()
example.importTracer(file_name=os.path.join(result_path, file_name))
print(example({'x':[2]}))  ## should give [0, 1, 0, 0, 0, 0]
print(example({'x':[2.5]})) ## should give [0, 0, 1, 0, 0, 0]
print(example({'x':[3]})) ## should give [0, 0, 1, 0, 0, 0]


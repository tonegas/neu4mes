import copy

import torch.nn as nn
import torch

from neu4mes.relation import NeuObj, Stream, AutoToStream, merge
from neu4mes.input import Input
from neu4mes.model import Model
from neu4mes.parameter import Parameter
from neu4mes.utilis import check

linear_relation_name = 'Linear'
class Linear(NeuObj, AutoToStream):
    def __init__(self, output_dimension:int = None, W:Parameter = None, b:bool = True):
        self.relation_name = linear_relation_name
        self.parameter = W
        self.bias = b

        if W is None:
            self.output_dimension = 1 if output_dimension is None else output_dimension
            super().__init__('P' + linear_relation_name + str(NeuObj.count))
        else:
            check(type(W) is Parameter, TypeError, 'The parameter must be of type Parameter')
            window = 'tw' if 'tw' in W.dim else ('sw' if 'sw' in W.dim else None)
            check(window == None, ValueError, 'The parameter must not have window dimension')
            check(len(W.dim['dim']) == 2, ValueError,'The parameter dimensions must be a tuple of 2.')
            self.output_dimension = W.dim['dim'][0]
            if output_dimension is not None:
                check(W.dim['dim'][0] == output_dimension, ValueError, 'output_dimension must be equal to the second dim of the parameter')
            super().__init__(W.name)
            #self.json['Parameters'][self.name] = copy.deepcopy(W.dim)
            self.json['Parameters'][self.name] = copy.deepcopy(W.json['Parameters'][W.name])

    def __call__(self, obj):
        stream_name = linear_relation_name + str(Stream.count)
        window = 'tw' if 'tw' in obj.dim else ('sw' if 'sw' in obj.dim else None)

        if self.parameter is None:
            self.json['Parameters'][self.name] = { 'dim': (self.output_dimension,obj.dim['dim'],) }
        else:
            #self.json['Parameters'][self.name] = {'dim': self.parameter.dim['dim']}
            check(self.parameter.dim['dim'][1] == obj.dim['dim'], ValueError,
                  'the input dimension must be equal to the first dim of the parameter')

        stream_json = merge(self.json,obj.json)
        if type(obj) is Stream:
            stream_json['Relations'][stream_name] = [linear_relation_name, [obj.name], self.name, self.bias]
            return Stream(stream_name, stream_json,{'dim': self.output_dimension, window:obj.dim[window]})
        else:
            raise Exception(f'The type of the input \'{obj.name}\' for the Linear is not correct.')

class Linear_Layer(nn.Module):
    def __init__(self, weights, bias):
        super(Linear_Layer, self).__init__()
        self.lin = nn.Linear(in_features=weights.size(0), out_features=weights.size(1), bias=bias)
        self.lin.weight = nn.Parameter(weights.t())

    def forward(self, x):
        return self.lin(x)

def createLinear(self, weights, bias):
    return Linear_Layer(weights, bias)

setattr(Model, linear_relation_name, createLinear)

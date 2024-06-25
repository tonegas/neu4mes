import copy

import torch.nn as nn
import torch

from neu4mes.relation import NeuObj, Stream, AutoToStream, merge
from neu4mes.utilis import check
from neu4mes.model import Model
from neu4mes.parameter import Parameter

fir_relation_name = 'Fir'

class Fir(NeuObj, AutoToStream):
    def __init__(self, output_dimension = None, parameter = None):
        self.relation_name = fir_relation_name
        self.parameter = parameter

        if parameter is None:
            self.output_dimension = 1 if output_dimension is None else output_dimension
            super().__init__('P' + fir_relation_name + str(NeuObj.count))
            self.json['Parameters'][self.name] = { 'dim': self.output_dimension }
        else:
            check(type(parameter) is Parameter, TypeError, 'Input parameter must be of type Parameter')
            if output_dimension is None:
                check(type(parameter.dim['dim']) is int, TypeError, 'Dimension of the parameter must be an integer for the Fir')
                self.output_dimension = parameter.dim['dim']
            else:
                self.output_dimension = output_dimension
                check(parameter.dim['dim'] == self.output_dimension, ValueError, 'output_dimension must be equal to dim of the Parameter')
            super().__init__(parameter.name)
            #self.json['Parameters'][self.name] = copy.deepcopy(parameter.dim)
            self.json['Parameters'][self.name] = copy.deepcopy(parameter.json['Parameters'][parameter.name])

    def __call__(self, obj):
        stream_name = fir_relation_name + str(Stream.count)
        check('dim' in obj.dim and obj.dim['dim'] == 1, ValueError, 'Input dimension must be scalar')
        window = 'tw' if 'tw' in obj.dim else ('sw' if 'sw' in obj.dim else None)
        if window:
            if self.parameter:
                check(window in self.json['Parameters'][self.name],
                      KeyError,
                      f"The window \'{window}\' of the input is not in the parameter")
                check(self.json['Parameters'][self.name][window] == obj.dim[window],
                      ValueError,
                      f"The window \'{window}\' of the input must be the same of the parameter")
            else:
                self.json['Parameters'][self.name][window] = obj.dim[window]
        else:
            if self.parameter:
                cond = 'sw' not in self.json['Parameters'][self.name] and 'tw' not in self.json['Parameters'][self.name]
                check(cond, KeyError,'The parameter have a time window and the input no')

        stream_json = merge(self.json,obj.json)
        if type(obj) is Stream:
            stream_json['Relations'][stream_name] = [fir_relation_name, [obj.name], self.name]
            return Stream(stream_name, stream_json,{'dim':self.output_dimension, 'sw': 1})
        else:
            raise Exception(f'The type of the input \'{obj.name}\' for the Fir is not correct.')

class Fir_Layer(nn.Module):
    def __init__(self, weights):
        super(Fir_Layer, self).__init__()
        self.lin = nn.Linear(in_features=weights.size(0), out_features=weights.size(1), bias=False)
        self.lin.weight = nn.Parameter(weights.t())

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.lin(x)

def createFir(self, weights):
    return Fir_Layer(weights)

setattr(Model, fir_relation_name, createFir)

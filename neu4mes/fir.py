import copy
import inspect

import torch.nn as nn
import torch

from neu4mes.relation import NeuObj, Stream, AutoToStream
from neu4mes.utilis import check, merge
from neu4mes.model import Model
from neu4mes.parameter import Parameter
from neu4mes.input import Input

fir_relation_name = 'Fir'

class Fir(NeuObj, AutoToStream):
    def __init__(self, output_dimension:int|None = None, parameter_init:None = None, parameter_init_params:None = None,
                 parameter:Parameter|None|str = None, dropout:int = 0):
        self.relation_name = fir_relation_name
        self.parameter_init = parameter_init
        self.parameter_init_params = parameter_init_params
        self.parameter = parameter
        self.namep = None
        self.dropout = dropout
        super().__init__('P' + fir_relation_name + str(NeuObj.count))

        if parameter is None:
            self.output_dimension = 1 if output_dimension is None else output_dimension
            self.namep = self.name+'p'
            self.json['Parameters'][self.namep] = { 'dim': self.output_dimension }
        elif type(parameter) is str:
            self.output_dimension = 1 if output_dimension is None else output_dimension
            self.namep = parameter
            self.json['Parameters'][self.namep] = { 'dim': self.output_dimension }
        else:
            check(type(parameter) is Parameter, TypeError, 'Input parameter must be of type Parameter')
            check(len(parameter.dim) == 2,ValueError,f"The values of the parameters must be have two dimensions (tw/sample_rate or sw,output_dimension).")
            if output_dimension is None:
                check(type(parameter.dim['dim']) is int, TypeError, 'Dimension of the parameter must be an integer for the Fir')
                self.output_dimension = parameter.dim['dim']
            else:
                self.output_dimension = output_dimension
                check(parameter.dim['dim'] == self.output_dimension, ValueError, 'output_dimension must be equal to dim of the Parameter')
            self.namep = parameter.name
            self.json['Parameters'][self.namep] = copy.deepcopy(parameter.json['Parameters'][parameter.name])

    def __call__(self, obj:Stream) -> Stream:
        stream_name = fir_relation_name + str(Stream.count)
        check(type(obj) is not Input, TypeError,
              f"The type of {obj.name} is Input not a Stream create a Stream using the functions: tw, sw, z, last, next.")
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for Fir operation.")
        check('dim' in obj.dim and obj.dim['dim'] == 1, ValueError, f"Input dimension is {obj.dim['dim']} and not scalar")
        window = 'tw' if 'tw' in obj.dim else ('sw' if 'sw' in obj.dim else None)
        if window:
            if type(self.parameter) is Parameter:
                check(window in self.json['Parameters'][self.namep],
                      KeyError,
                      f"The window \'{window}\' of the input is not in the parameter")
                check(self.json['Parameters'][self.namep][window] == obj.dim[window],
                      ValueError,
                      f"The window \'{window}\' of the input must be the same of the parameter")
            else:
                self.json['Parameters'][self.namep][window] = obj.dim[window]
        else:
            if type(self.parameter) is Parameter:
                cond = 'sw' not in self.json['Parameters'][self.namep] and 'tw' not in self.json['Parameters'][self.nampe]
                check(cond, KeyError,'The parameter have a time window and the input no')

        if self.parameter_init is not None:
            check('values' not in self.json['Parameters'][self.namep], ValueError, f"The parameter {self.namep} is already initialized.")
            check(inspect.isfunction(self.parameter_init), ValueError,
                  f"The parameter_init parameter must be a function.")
            self.json['Parameters'][self.namep]['init_fun'] = { 'code' : inspect.getsource(self.parameter_init), 'name' : self.parameter_init.__name__ }
            if self.parameter_init_params is not None:
                self.json['Parameters'][self.namep]['init_fun']['params'] = self.parameter_init_params

        stream_json = merge(self.json,obj.json)
        stream_json['Relations'][stream_name] = [fir_relation_name, [obj.name], self.namep, self.dropout]
        return Stream(stream_name, stream_json,{'dim':self.output_dimension, 'sw': 1})
'''
class Fir_Layer(nn.Module):
    def __init__(self, weights, dropout):
        super(Fir_Layer, self).__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.lin = nn.Linear(in_features=weights.size(0), out_features=weights.size(1), bias=False)
        self.lin.weight = nn.Parameter(weights.t())

    def forward(self, x):
        x = x.permute(0, 2, 1)
        print('x shape: ', x.shape)
        print('weight shape: ', self.lin.weight.shape)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.lin(x)
        print('result shape: ', x.shape)
        return x
'''
class Fir_Layer(nn.Module):
    def __init__(self, weights, dropout=0):
        super(Fir_Layer, self).__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.weights = weights

    def forward(self, x):
        # x is expected to be of shape [batch, window, 1]
        batch_size = x.size(0)
        output_features = self.weights.size(1)
        # Remove the last dimension (1) to make x shape [batch, window]
        x = x.squeeze(-1)
        # Perform the linear transformation: y = xW^T
        x = torch.matmul(x, self.weights)
        # Reshape y to be [batch, 1, output_features]
        x = x.view(batch_size, 1, output_features)
        # Add dropout if necessary
        if self.dropout:
            x = self.dropout(x)
        return x

def createFir(self, weights, dropout):
    return Fir_Layer(weights, dropout)

setattr(Model, fir_relation_name, createFir)

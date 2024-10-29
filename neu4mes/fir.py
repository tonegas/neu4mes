import copy, inspect, textwrap, torch

import torch.nn as nn

from collections.abc import Callable

from neu4mes.relation import NeuObj, Stream, AutoToStream
from neu4mes.utils import check, merge, enforce_types
from neu4mes.model import Model
from neu4mes.parameter import Parameter
from neu4mes.input import Input

fir_relation_name = 'Fir'

class Fir(NeuObj, AutoToStream):
    @enforce_types
    def __init__(self, output_dimension:int|None = None,
                 parameter_init:Callable|None = None,
                 parameter_init_params:dict|None = None,
                 bias_init:Callable|None = None,
                 bias_init_params:dict|None = None,
                 parameter:Parameter|str|None = None,
                 bias:bool|str|Parameter|None = None,
                 dropout:int|float = 0):

        self.relation_name = fir_relation_name
        self.parameter_init = parameter_init
        self.parameter_init_params = parameter_init_params
        self.parameter = parameter
        self.bias_init = bias_init
        self.bias_init_params = bias_init_params
        self.bias = bias
        self.pname = None
        self.bname = None
        self.dropout = dropout
        super().__init__('P' + fir_relation_name + str(NeuObj.count))

        if parameter is None:
            self.output_dimension = 1 if output_dimension is None else output_dimension
            self.pname = self.name + 'p'
            self.json['Parameters'][self.pname] = {'dim': self.output_dimension}
        elif type(parameter) is str:
            self.output_dimension = 1 if output_dimension is None else output_dimension
            self.pname = parameter
            self.json['Parameters'][self.pname] = {'dim': self.output_dimension}
        else:
            check(type(parameter) is Parameter, TypeError, 'Input parameter must be of type Parameter')
            check(len(parameter.dim) == 2,ValueError,f"The values of the parameters must be have two dimensions (tw/sample_rate or sw,output_dimension).")
            if output_dimension is None:
                check(type(parameter.dim['dim']) is int, TypeError, 'Dimension of the parameter must be an integer for the Fir')
                self.output_dimension = parameter.dim['dim']
            else:
                self.output_dimension = output_dimension
                check(parameter.dim['dim'] == self.output_dimension, ValueError, 'output_dimension must be equal to dim of the Parameter')
            self.pname = parameter.name
            self.json['Parameters'][self.pname] = copy.deepcopy(parameter.json['Parameters'][parameter.name])

        if bias is not None:
            check(type(bias) is Parameter or type(bias) is bool or type(bias) is str, TypeError, 'The "bias" must be of type Parameter, bool or str.')
            if type(bias) is Parameter:
                check(type(bias.dim['dim']) is int, ValueError, 'The "bias" dimensions must be an integer.')
                if output_dimension is not None:
                    check(bias.dim['dim'] == output_dimension, ValueError,
                          'output_dimension must be equal to the dim of the "bias".')
                self.bname = bias.name
                self.json['Parameters'][bias.name] = copy.deepcopy(bias.json['Parameters'][bias.name])
            elif type(bias) is str:
                self.bname = bias
                self.json['Parameters'][self.bname] = { 'dim': self.output_dimension }
            else:
                self.bname = self.name + 'b'
                self.json['Parameters'][self.bname] = { 'dim': self.output_dimension }


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
                check(window in self.json['Parameters'][self.pname],
                      KeyError,
                      f"The window \'{window}\' of the input is not in the parameter")
                check(self.json['Parameters'][self.pname][window] == obj.dim[window],
                      ValueError,
                      f"The window \'{window}\' of the input must be the same of the parameter")
            else:
                self.json['Parameters'][self.pname][window] = obj.dim[window]
        else:
            if type(self.parameter) is Parameter:
                cond = 'sw' not in self.json['Parameters'][self.pname] and 'tw' not in self.json['Parameters'][self.nampe]
                check(cond, KeyError,'The parameter have a time window and the input no')

        if self.parameter_init is not None:
            check('values' not in self.json['Parameters'][self.pname], ValueError, f"The parameter {self.pname} is already initialized.")
            check(inspect.isfunction(self.parameter_init), ValueError,
                  f"The parameter_init parameter must be a function.")
            code = textwrap.dedent(inspect.getsource(self.parameter_init)).replace('\"', '\'')
            self.json['Parameters'][self.pname]['init_fun'] = {'code' : code, 'name' : self.parameter_init.__name__}
            if self.parameter_init_params is not None:
                self.json['Parameters'][self.pname]['init_fun']['params'] = self.parameter_init_params

        if self.bias_init is not None:
            check(self.bname is not None, ValueError,f"The bias is missing.")
            check('values' not in self.json['Parameters'][self.bname], ValueError, f"The parameter {self.bname} is already initialized.")
            check(inspect.isfunction(self.bias_init), ValueError,
                  f"The bias_init parameter must be a function.")
            code = textwrap.dedent(inspect.getsource(self.bias_init)).replace('\"', '\'')
            self.json['Parameters'][self.bname]['init_fun'] = { 'code' : code, 'name' : self.bias_init.__name__ }
            if self.bias_init_params is not None:
                self.json['Parameters'][self.bname]['init_fun']['params'] = self.bias_init_params

        stream_json = merge(self.json,obj.json)
        stream_json['Relations'][stream_name] = [fir_relation_name, [obj.name], self.pname, self.bname, self.dropout]
        return Stream(stream_name, stream_json,{'dim':self.output_dimension, 'sw': 1})


class Fir_Layer(nn.Module):
    def __init__(self, weights, bias=None, dropout=0):
        super(Fir_Layer, self).__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.weights = weights
        self.bias = bias

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
        # Add bias if necessary
        if self.bias is not None:
            x += self.bias  # Add bias
        # Add dropout if necessary
        if self.dropout is not None:
            x = self.dropout(x)
        return x

def createFir(self, *inputs):
    return Fir_Layer(weights=inputs[0], bias=inputs[1], dropout=inputs[2])

setattr(Model, fir_relation_name, createFir)

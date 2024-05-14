import copy

import torch.nn as nn

from neu4mes.relation import NeuObj, Stream, AutoToStream, merge
from neu4mes.input import Input
from neu4mes.model import Model
from neu4mes.parameter import Parameter

fir_relation_name = 'Fir'

class Fir(NeuObj, AutoToStream):
    def __init__(self, output_dimension = 1, parameter = None):
        self.relation_name = fir_relation_name
        self.output_dimension = output_dimension
        self.parameter = parameter

        if parameter is None:
            super().__init__('P' + fir_relation_name + str(NeuObj.count))
            self.json['Parameters'][self.name] = { 'dim': self.output_dimension }
        else:
            assert type(parameter) is Parameter, 'input parameter must be of type Parameter'
            assert parameter.dim['dim'] == self.output_dimension, 'output_dimension must be equal to dim of the parameter'
            super().__init__(parameter.name)
            self.json['Parameters'][self.name] = copy.deepcopy(parameter.dim)

    def __call__(self, obj):
        stream_name = fir_relation_name + str(Stream.count)
        #TODO remove this limit the input can have different dimension
        #The output dimensions will be equal to the input dimension
        assert 'dim' in obj.dim and obj.dim['dim'] == 1, 'Input dimension must be scalar'
        if 'tw' in obj.dim:
            if type(obj.dim['tw']) is list:
                assert obj.dim['tw'][0] <= obj.dim['tw'][1], 'first element of the time interval must be less than the second element'
                tw = obj.dim['tw'][1] - obj.dim['tw'][0]
            else:
                tw = obj.dim['tw']
            if self.parameter:
                assert self.json['Parameters'][self.name]['tw'] == tw, 'Time window of the input dimension must be the same of the parameter'
            else:
                self.json['Parameters'][self.name]['tw'] = tw
        else:
            if self.parameter:
                assert 'tw' not in self.json['Parameters'][self.name], 'The parameter have a time window and the input no'

        stream_json = merge(self.json,obj.json)
        if type(obj) is Input or type(obj) is Stream:
            stream_json['Relations'][stream_name] = [fir_relation_name, [obj.name], self.name]
            return Stream(stream_name, stream_json,{'dim':self.output_dimension})
        else:
            raise Exception('Type is not supported!')

def createLinear(self, input_size, output_size):
    return nn.Linear(in_features=input_size, out_features=output_size, bias=False)

def createLinearBias(self, input_size, output_size):
    return nn.Linear(in_features=input_size, out_features=output_size, bias=True)

setattr(Model, fir_relation_name, createLinear)
#setattr(Model, fir_bias_relation_name, createLinearBias)
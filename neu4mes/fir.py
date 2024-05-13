import copy

import torch.nn as nn

from neu4mes.relation import NeuObj, Stream, Relation, AutoToStream, merge
from neu4mes.input import Input
from neu4mes.model import Model

fir_relation_name = 'Fir'
#fir_bias_relation_name = 'FirBias'

class Fir(NeuObj, AutoToStream):
    def __init__(self, output_dimension = 1):
        self.relation_name = fir_relation_name
        self.output_dimension = output_dimension
        super().__init__('P' + fir_relation_name + str(NeuObj.count))
        self.json['Parameters'][self.name] = {
            'dim_out': self.output_dimension
        }

    def __call__(self, obj):
        stream_name = fir_relation_name + str(Stream.count)
        self.json['Parameters'][self.name].update(**obj.dim)
        stream_json = merge(self.json,obj.json)
        if type(obj) is Input or type(obj) is Stream:
            stream_json['Relations'][stream_name] = [fir_relation_name, [obj.name], self.name]
            return Stream(stream_name, stream_json,{'dim_in':self.output_dimension})
        else:
            raise Exception('Type is not supported!')

def createLinear(self, input_size, output_size):
    return nn.Linear(in_features=input_size, out_features=output_size, bias=False)

def createLinearBias(self, input_size, output_size):
    return nn.Linear(in_features=input_size, out_features=output_size, bias=True)

setattr(Model, fir_relation_name, createLinear)
#setattr(Model, fir_bias_relation_name, createLinearBias)
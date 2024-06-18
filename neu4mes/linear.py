import copy

import torch.nn as nn
import torch

from neu4mes.relation import NeuObj, Stream, AutoToStream, merge
from neu4mes.input import Input
from neu4mes.model import Model
from neu4mes.parameter import Parameter

FC_relation_name = 'FullyConnected'
list_of_activations = ['ReLU','Tanh','Sign','Elu']
class FullyConnected(NeuObj, AutoToStream):
    def __init__(self, sequence = []):
        self.relation_name = FC_relation_name
        self.sequence = sequence
        pass

        # if parameter is None:
        #     super().__init__('P' + fir_relation_name + str(NeuObj.count))
        #     self.json['Parameters'][self.name] = { 'dim': self.output_dimension }
        # else:
        #     assert type(parameter) is Parameter, 'input parameter must be of type Parameter'
        #     assert parameter.dim['dim'] == self.output_dimension, 'output_dimension must be equal to dim of the parameter'
        #     super().__init__(parameter.name)
        #     self.json['Parameters'][self.name] = copy.deepcopy(parameter.dim)

    def __call__(self, obj):
        pass
        # stream_name = fir_relation_name + str(Stream.count)
        # #TODO remove this limit the input can have different dimension
        # #The output dimensions will be equal to the input dimension
        # assert 'dim' in obj.dim and obj.dim['dim'] == 1, 'Input dimension must be scalar'
        # window = 'tw' if 'tw' in obj.dim else ('sw' if 'sw' in obj.dim else None)
        # if window:
        #     if self.parameter:
        #         assert self.json['Parameters'][self.name][window] == obj.dim[window], 'Time window of the input dimension must be the same of the parameter'
        #     else:
        #         self.json['Parameters'][self.name][window] = obj.dim[window]
        # else:
        #     if self.parameter:
        #         assert window not in self.json['Parameters'][self.name], 'The parameter have a time window and the input no'
        #
        # stream_json = merge(self.json,obj.json)
        # if type(obj) is Input or type(obj) is Stream:
        #     stream_json['Relations'][stream_name] = [fir_relation_name, [obj.name], self.name]
        #     return Stream(stream_name, stream_json,{'dim':self.output_dimension})
        # else:
        #     raise Exception('Type is not supported!')
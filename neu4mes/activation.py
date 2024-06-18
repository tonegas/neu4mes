import copy
import torch.nn as nn

from neu4mes.relation import Stream, ToStream
from neu4mes.model import Model

relu_relation_name = 'ReLU'
tanh_relation_name = 'Tanh'

class Relu(Stream, ToStream):
    def __init__(self, obj):
        super().__init__(relu_relation_name + str(Stream.count),obj.json,obj.dim)
        if type(obj) is Stream:
            self.json['Relations'][self.name] = [relu_relation_name,[obj.name]]
        else:
            raise Exception('Type is not supported!')

class Tanh(Stream, ToStream):
    def __init__(self, obj):
        super().__init__(tanh_relation_name + str(Stream.count),obj.json,obj.dim)
        if type(obj) is Stream:
            self.json['Relations'][self.name] = [tanh_relation_name,[obj.name]]
        else:
            raise Exception('Type is not supported!')

def createTanh(self, *input):
    return nn.Tanh()

def createRelu(self, *input):
    return nn.ReLU()

setattr(Model, relu_relation_name, createRelu)
setattr(Model, tanh_relation_name, createTanh)
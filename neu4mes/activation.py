import copy
import torch.nn as nn

from neu4mes.relation import Stream, ToStream, toStream
from neu4mes.model import Model
from neu4mes.utilis import check

relu_relation_name = 'ReLU'
tanh_relation_name = 'Tanh'

class Relu(Stream, ToStream):
    def __init__(self, obj:Stream) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream,TypeError,
              f"The type of {obj.name} is {type(obj)} and is not supported for Relu operation.")
        super().__init__(relu_relation_name + str(Stream.count),obj.json,obj.dim)
        self.json['Relations'][self.name] = [relu_relation_name,[obj.name]]

class Tanh(Stream, ToStream):
    def __init__(self, obj:Stream) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream,TypeError,
              f"The type of {obj.name} is {type(obj)} and is not supported for Tanh operation.")
        super().__init__(tanh_relation_name + str(Stream.count),obj.json,obj.dim)
        self.json['Relations'][self.name] = [tanh_relation_name,[obj.name]]

def createTanh(self, *input):
    return nn.Tanh()

def createRelu(self, *input):
    return nn.ReLU()

setattr(Model, relu_relation_name, createRelu)
setattr(Model, tanh_relation_name, createTanh)
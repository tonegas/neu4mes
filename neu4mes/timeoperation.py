import torch.nn as nn

from neu4mes.relation import Stream, ToStream, toStream
from neu4mes.model import Model
from neu4mes.utilis import check

int_relation_name = 'Int'
diff_relation_name = 'Diff'

class Int(Stream, ToStream):
    def __init__(self, obj:Stream) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for Int operation.")
        super().__init__(int_relation_name + str(Stream.count),obj.json,obj.dim)
        self.json['Relations'][self.name] = [int_relation_name,[obj.name]]

class Diff(Stream, ToStream):
    def __init__(self, obj:Stream) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream,TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for Diff operation.")
        super().__init__(diff_relation_name + str(Stream.count),obj.json,obj.dim)
        self.json['Relations'][self.name] = [diff_relation_name,[obj.name]]

def createTanh(self, *input):
    return nn.Tanh()

def createRelu(self, *input):
    return nn.ReLU()

setattr(Model, relu_relation_name, createRelu)
setattr(Model, tanh_relation_name, createTanh)



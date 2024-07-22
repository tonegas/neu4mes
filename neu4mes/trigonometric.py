import torch
import torch.nn as nn

from neu4mes.relation import ToStream, Stream, toStream
from neu4mes.model import Model
from neu4mes.utilis import check

sin_relation_name = 'Sin'
cos_relation_name = 'Cos'
tan_relation_name = 'Tan'

class Sin(Stream, ToStream):
    def __init__(self, obj:Stream) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for Sin operation.")
        super().__init__(sin_relation_name + str(Stream.count),obj.json,obj.dim)
        self.json['Relations'][self.name] = [sin_relation_name, [obj.name]]

class Cos(Stream, ToStream):
    def __init__(self, obj:Stream) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for Cos operation.")
        super().__init__(cos_relation_name + str(Stream.count),obj.json,obj.dim)
        self.json['Relations'][self.name] = [cos_relation_name, [obj.name]]

class Tan(Stream, ToStream):
    def __init__(self, obj:Stream) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for Tan operation.")
        super().__init__(tan_relation_name + str(Stream.count),obj.json,obj.dim)
        self.json['Relations'][self.name] = [tan_relation_name, [obj.name]]


class Sin_Layer(nn.Module):
    def __init__(self,):
        super(Sin_Layer, self).__init__()
    def forward(self, x):
        return torch.sin(x)

def createSin(self, *inputs):
    return Sin_Layer()

class Cos_Layer(nn.Module):
    def __init__(self,):
        super(Cos_Layer, self).__init__()
    def forward(self, x):
        return torch.cos(x)

def createCos(self, *inputs):
    return Cos_Layer()

class Tan_Layer(nn.Module):
    def __init__(self,):
        super(Tan_Layer, self).__init__()
    def forward(self, x):
        return torch.tan(x)

def createTan(self, *inputs):
    return Tan_Layer()


setattr(Model, sin_relation_name, createSin)
setattr(Model, cos_relation_name, createCos)
setattr(Model, tan_relation_name, createTan)
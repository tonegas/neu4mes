import torch.nn as nn
import torch

from neu4mes.relation import NeuObj, AutoToStream, Stream
from neu4mes.utilis import check, merge
from neu4mes.input import Input
from neu4mes.model import Model

# Binary operators
dropout_relation_name = 'Dropout'

class Dropout(NeuObj, AutoToStream):
    def __init__(self, probability:float|None = 0.01):
        self.probability = probability
        super().__init__(dropout_relation_name + str(NeuObj.count))

    def __call__(self, obj:Stream) -> Stream:
        stream_name = dropout_relation_name + str(Stream.count)
        check(type(obj) is not Input, TypeError,
              f"The type of {obj.name} is Input not a Stream create a Stream using the functions: tw, sw, z, last, next.")
        stream_json = merge(self.json, obj.json)
        stream_json['Relations'][stream_name] = [dropout_relation_name, [obj.name], self.probability]
        return Stream(stream_name, stream_json, obj.dim)

class Dropout_Layer(nn.Module):
    def __init__(self, probability=0):
        super(Dropout_Layer, self).__init__()
        self.dropout = nn.Dropout(p=probability)

    def forward(self, x):
        x = self.dropout(x)
        return x

def createDropout(self, *inputs):
    return Dropout_Layer(probability=inputs[0])

setattr(Model, dropout_relation_name, createDropout)
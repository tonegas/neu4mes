import copy
import torch.nn as nn

from neu4mes.relation import NeuObj, AutoToStream, Stream, ToStream
from neu4mes.input import Input
from neu4mes.model import Model

relu_relation_name = 'ReLU'

class Relu(Stream, ToStream):
    def __init__(self, obj):
        super().__init__(relu_relation_name + str(Stream.count),obj.json,obj.dim)
        if (type(obj) is Input or type(obj) is Stream):
            self.json['Relations'][self.name] = [relu_relation_name,[obj.name]]


def createRelu(self, *input):
    return nn.ReLU()

setattr(Model, relu_relation_name, createRelu)
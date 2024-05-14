import copy

import torch.nn as nn
import torch

from neu4mes.relation import ToStream, merge, NeuObj, Stream, Relation
from neu4mes.model import Model
from neu4mes.input import Input

part_relation_name = 'Part'
select_relation_name = 'Select'
timepart_relation_name = 'TimePart'
timeselect_relation_name = 'TimeSelect'

class Part(Stream, ToStream):
    def __init__(self, obj, i, j):
        assert 'dim' in obj.dim and type(obj.dim['dim']) is int, 'Input dimension must be vector'
        assert i >= 0 and j >= 1 and i < obj.dim['dim']-1 and j < obj.dim['dim'], 'i and j must be in the input dimension'
        dim = copy.deepcopy(obj.dim)
        dim['dim'] = j - i
        super().__init__(part_relation_name + str(Stream.count),obj.json,dim)
        if (type(obj) is Input or type(obj) is Stream):
            self.json['Relations'][self.name] = [part_relation_name,[obj.name],[i,j]]

class Select(Stream, ToStream):
    def __init__(self, obj, i):
        assert 'dim' in obj.dim and type(obj.dim['dim']) is int, 'Input dimension must be vector'
        assert i >= 0 and i < obj.dim['dim'], 'i must be in the input dimension'
        dim = copy.deepcopy(obj.dim)
        dim['dim'] = 1
        super().__init__(select_relation_name + str(Stream.count),obj.json,dim)
        if (type(obj) is Input or type(obj) is Stream):
            self.json['Relations'][self.name] = [select_relation_name,[obj.name],i]


class TimePart(Stream, ToStream):
    def __init__(self, obj, i, j):
        assert 'tw' in obj.dim, 'Input must have a time window'
        assert i <= j, 'i must be smaller than j'
        if type(obj.dim['tw']) is int:
            backward = -obj.dim['tw']
            forward = 0
        else:
            backward = obj.dim['tw'][0]
            forward = obj.dim['tw'][1]
        assert i >= backward and i <= forward, 'i must be in the time window of the input'
        assert j >= backward and j <= forward, 'j must be in the time window of the input'
        dim = copy.deepcopy(obj.dim)
        dim['tw']  = j - i
        super().__init__(timepart_relation_name + str(Stream.count),obj.json,dim)
        if (type(obj) is Input or type(obj) is Stream):
            self.json['Relations'][self.name] = [timepart_relation_name,[obj.name],[i,j]]


class TimeSelect(Stream, ToStream):
    def __init__(self, obj, i):
        assert 'tw' in obj.dim, 'Input must have a time window'
        if type(obj.dim['tw']) is int:
            backward = -obj.dim['tw']
            forward = 0
        else:
            backward = obj.dim['tw'][0]
            forward = obj.dim['tw'][1]
        assert i >= backward and i <= forward, 'i must be in the time window of the input'
        dim = copy.deepcopy(obj.dim)
        del dim['tw']
        super().__init__(timeselect_relation_name + str(Stream.count),obj.json,dim)
        if (type(obj) is Input or type(obj) is Stream):
            self.json['Relations'][self.name] = [timeselect_relation_name,[obj.name],i]

class Part_Layer(nn.Module):
    def __init__(self):
        super(Part_Layer, self).__init__()
    def forward(self, x):
        return torch.pow(x,2)

def createPart(self, *inputs):
    return Part_Layer()

setattr(Model, part_relation_name, createPart)

import copy

import torch.nn as nn

from neu4mes.relation import ToStream, Stream
from neu4mes.model import Model
from neu4mes.utilis import check

part_relation_name = 'Part'
select_relation_name = 'Select'
timepart_relation_name = 'TimePart'
timeselect_relation_name = 'TimeSelect'
samplepart_relation_name = 'SamplePart'
sampleselect_relation_name = 'SampleSelect'

class Part(Stream, ToStream):
    def __init__(self, obj, i, j):
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for Part operation.")
        check(i >= 0 and j > 0 and i < obj.dim['dim'] and j <= obj.dim['dim'],
              IndexError,
              f"i={i} or j={j} are not in the range [0,{obj.dim['dim']}]")
        dim = copy.deepcopy(obj.dim)
        dim['dim'] = j - i
        super().__init__(part_relation_name + str(Stream.count),obj.json,dim)
        self.json['Relations'][self.name] = [part_relation_name,[obj.name],[i,j]]

class Part_Layer(nn.Module):
    def __init__(self, i, j):
        super(Part_Layer, self).__init__()
        self.i, self.j = i, j

    def forward(self, x):
        assert x.ndim >= 3, 'The Part Relation Works only for 3D inputs'
        return x[:, :, self.i:self.j]

## Select elements on the third dimension in the range [i,j]
def createPart(self, i, j):
    return Part_Layer(i, j)

class Select(Stream, ToStream):
    def __init__(self, obj, i):
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for Select operation.")
        check(i >= 0 and i < obj.dim['dim'],
              IndexError,
              f"i={i} are not in the range [0,{obj.dim['dim']}]")
        dim = copy.deepcopy(obj.dim)
        dim['dim'] = 1
        super().__init__(select_relation_name + str(Stream.count),obj.json,dim)
        self.json['Relations'][self.name] = [select_relation_name,[obj.name],i]

class Select_Layer(nn.Module):
    def __init__(self, idx):
        super(Select_Layer, self).__init__()
        self.idx = idx

    def forward(self, x):
        assert x.ndim >= 3, 'The Part Relation Works only for 3D inputs'
        return x[:, :, self.idx:self.idx + 1]

## Select an element i on the third dimension
def createSelect(self, idx):
    return Select_Layer(idx)

class SamplePart(Stream, ToStream):
    def __init__(self, obj, i, j, offset = None):
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for SamplePart operation.")
        check('sw' in obj.dim, KeyError, 'Input must have a sample window')
        check(i < j, ValueError, 'i must be smaller than j')
        if obj.name in obj.json['Inputs']:
            backward_idx = obj.json['Inputs'][obj.name]['sw'][0]
            forward_idx = obj.json['Inputs'][obj.name]['sw'][1]
        else:
            backward_idx = 0
            forward_idx = obj.dim['sw']
        check(i >= backward_idx and i < forward_idx, ValueError, 'i must be in the sample window of the input')
        check(j > backward_idx and j <= forward_idx, ValueError, 'j must be in the sample window of the input')
        dim = copy.deepcopy(obj.dim)
        dim['sw']  = j - i
        super().__init__(samplepart_relation_name + str(Stream.count),obj.json,dim)
        rel = [samplepart_relation_name,[obj.name],[i,j]]
        if offset is not None:
            check(i <= offset < j, IndexError,"The offset must be inside the sample window")
            rel.append(offset)
        self.json['Relations'][self.name] = rel

class SamplePart_Layer(nn.Module):
    def __init__(self, part, offset):
        super(SamplePart_Layer, self).__init__()
        self.back, self.forw = part[0], part[1]
        self.offset = offset

    def forward(self, x):
        if self.offset is not None:
            x = x - x[:, self.offset].unsqueeze(1)
        return x[:, self.back:self.forw]

def createSamplePart(self, part, offset):
    return SamplePart_Layer(part=part, offset=offset)

class SampleSelect(Stream, ToStream):
    def __init__(self, obj, i):
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for SampleSelect operation.")
        check('sw' in obj.dim, KeyError, 'Input must have a sample window')
        backward_idx = 0
        forward_idx = obj.dim['sw']
        check(i >= backward_idx and i < forward_idx, ValueError, 'i must be in the sample window of the input')
        dim = copy.deepcopy(obj.dim)
        del dim['sw']
        super().__init__(sampleselect_relation_name + str(Stream.count),obj.json,dim)
        self.json['Relations'][self.name] = [sampleselect_relation_name,[obj.name],i]

class SampleSelect_Layer(nn.Module):
    def __init__(self, idx):
        super(SampleSelect_Layer, self).__init__()
        self.idx = idx

    def forward(self, x):
        assert x.ndim >= 2, 'The Part Relation Works only for 2D inputs'
        return x[:, self.idx:self.idx + 1, :]

def createSampleSelect(self, idx):
    return SampleSelect_Layer(idx)

class TimePart(Stream, ToStream):
    def __init__(self, obj, i, j, offset = None):
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for TimePart operation.")
        check('tw' in obj.dim, KeyError, 'Input must have a time window')
        check(i < j, ValueError, 'i must be smaller than j')
        if obj.name in obj.json['Inputs']:
            backward_idx = obj.json['Inputs'][obj.name]['tw'][0]
            forward_idx = obj.json['Inputs'][obj.name]['tw'][1]
        else:
            backward_idx = 0
            forward_idx = obj.dim['tw']
        check(i >= backward_idx and i < forward_idx, ValueError, 'i must be in the time window of the input')
        check(j > backward_idx and j <= forward_idx, ValueError, 'j must be in the time window of the input')
        dim = copy.deepcopy(obj.dim)
        dim['tw']  = j - i
        super().__init__(timepart_relation_name + str(Stream.count),obj.json,dim)
        rel = [timepart_relation_name,[obj.name],[i,j]]
        if offset is not None:
            check(i <= offset < j, IndexError,"The offset must be inside the time window")
            rel.append(offset)
        self.json['Relations'][self.name] = rel

class TimePart_Layer(nn.Module):
    def __init__(self, part, offset):
        super(TimePart_Layer, self).__init__()
        self.back, self.forw = part[0], part[1]
        self.offset = offset

    def forward(self, x):
        if self.offset is not None:
            x = x - x[:, self.offset].unsqueeze(1)
        return x[:, self.back:self.forw]

def createTimePart(self, part, offset):
    return TimePart_Layer(part=part, offset=offset)

class TimeSelect(Stream, ToStream):
    def __init__(self, obj, i):
        check('tw' in obj.dim, KeyError, 'Input must have a time window')
        backward_idx = 0
        forward_idx = obj.dim['tw']
        check(i >= backward_idx and i < forward_idx, ValueError, 'i must be in the time window of the input')
        dim = copy.deepcopy(obj.dim)
        del dim['tw']
        super().__init__(timeselect_relation_name + str(Stream.count),obj.json,dim)
        if (type(obj) is Stream):
            self.json['Relations'][self.name] = [timeselect_relation_name,[obj.name],i]

setattr(Model, part_relation_name, createPart)
setattr(Model, select_relation_name, createSelect)

setattr(Model, samplepart_relation_name, createSamplePart)
setattr(Model, sampleselect_relation_name, createSampleSelect)

setattr(Model, timepart_relation_name, createTimePart)

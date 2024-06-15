import copy

import torch.nn as nn
import torch

from neu4mes.relation import ToStream, merge, NeuObj, Stream, Relation
from neu4mes.model import Model
#from neu4mes.input import Input
from neu4mes.utilis import check

part_relation_name = 'Part'
select_relation_name = 'Select'
timepart_relation_name = 'TimePart'
input_timepart_relation_name = 'InputTimePart'
timeselect_relation_name = 'TimeSelect'
samplepart_relation_name = 'SamplePart'
input_samplepart_relation_name = 'InputSamplePart'
sampleselect_relation_name = 'SampleSelect'

class Part(Stream, ToStream):
    def __init__(self, obj, i, j):
        check(i >= 0 and j >= 1 and i < obj.dim['dim']-1 and j < obj.dim['dim'],
              IndexError,
              f"i={i} or j={j} are not in the range [0,{obj.dim['dim']}]")
        dim = copy.deepcopy(obj.dim)
        dim['dim'] = j - i
        super().__init__(part_relation_name + str(Stream.count),obj.json,dim)
        if (type(obj) is Stream):
            self.json['Relations'][self.name] = [part_relation_name,[obj.name],[i,j]]

class Select(Stream, ToStream):
    def __init__(self, obj, i):
        check(i >= 0 and i < obj.dim['dim'] - 1,
              IndexError,
              f"i={i} are not in the range [0,{obj.dim['dim']}]")
        dim = copy.deepcopy(obj.dim)
        dim['dim'] = 1
        super().__init__(select_relation_name + str(Stream.count),obj.json,dim)
        if (type(obj) is Stream):
            self.json['Relations'][self.name] = [select_relation_name,[obj.name],i]


class SamplePart(Stream, ToStream):
    def __init__(self, obj, i, j, offset = None):
        check('sw' in obj.dim, KeyError, 'Input must have a sample window')
        check(i <= j, ValueError, 'i must be smaller than j')
        backward_idx = 0
        forward_idx = obj.dim['sw']
        check(i >= backward_idx and i <= forward_idx, ValueError, 'i must be in the sample window of the input')
        check(j >= backward_idx and j <= forward_idx, ValueError, 'j must be in the sample window of the input')
        dim = copy.deepcopy(obj.dim)
        dim['sw']  = j - i
        super().__init__(samplepart_relation_name + str(Stream.count),obj.json,dim)
        rel = [samplepart_relation_name,[obj.name],[i,j]]
        if offset is not None:
            check(i < offset <= j, IndexError,"The offset must be inside the sample window")
            rel.append(offset)
        if (type(obj) is Stream):
            self.json['Relations'][self.name] = rel
        else:
            raise Exception('Type is not supported!')

class InputSamplePart(Stream, ToStream):
    def __init__(self, obj, i, j, offset = None):
        check('sw' in obj.dim, KeyError, 'Input must have a sample window')
        check(i <= j, ValueError, 'i must be smaller than j')
        check(obj.name in obj.json['Inputs'], KeyError, 'InputSamplePart must be call on an input')
        backward_idx = obj.json['Inputs'][obj.name]['sw'][0]
        forward_idx = obj.json['Inputs'][obj.name]['sw'][1]
        check(i >= backward_idx and i <= forward_idx, ValueError, 'i must be in the sample window of the input')
        check(j >= backward_idx and j <= forward_idx, ValueError, 'j must be in the sample window of the input')
        dim = copy.deepcopy(obj.dim)
        dim['sw']  = j - i
        super().__init__(samplepart_relation_name + str(Stream.count),obj.json,dim)
        rel = [samplepart_relation_name,[obj.name],[i,j]]
        if offset is not None:
            check(i < offset <= j, IndexError,"The offset must be inside the sample window")
            rel.append(offset)
        if (type(obj) is Stream):
            self.json['Relations'][self.name] = rel
        else:
            raise Exception('Type is not supported!')

class SampleSelect(Stream, ToStream):
    def __init__(self, obj, i):
        check('sw' in obj.dim, KeyError, 'Input must have a sample window')
        backward_idx = 0
        forward_idx = obj.dim['sw']
        check(i >= backward_idx and i <= forward_idx, ValueError, 'i must be in the sample window of the input')
        dim = copy.deepcopy(obj.dim)
        del dim['sw']
        super().__init__(timeselect_relation_name + str(Stream.count),obj.json,dim)
        if (type(obj) is Stream):
            self.json['Relations'][self.name] = [timeselect_relation_name,[obj.name],i]

class TimePart(Stream, ToStream):
    def __init__(self, obj, i, j, offset = None):
        check('tw' in obj.dim, KeyError, 'Input must have a time window')
        check(i <= j, ValueError, 'i must be smaller than j')
        backward_idx = 0
        forward_idx = obj.dim['tw']
        check(i >= backward_idx and i <= forward_idx, ValueError, 'i must be in the time window of the input')
        check(j >= backward_idx and j <= forward_idx, ValueError, 'j must be in the time window of the input')
        dim = copy.deepcopy(obj.dim)
        dim['tw']  = j - i
        super().__init__(timepart_relation_name + str(Stream.count),obj.json,dim)
        rel = [timepart_relation_name,[obj.name],[i,j]]
        if offset is not None:
            check(i < offset <= j, IndexError,"The offset must be inside the time window")
            rel.append(offset)
        if (type(obj) is Stream):
            self.json['Relations'][self.name] = rel
        else:
            raise Exception('Type is not supported!')

class InputTimePart(Stream, ToStream):
    def __init__(self, obj, i, j, offset = None):
        check('tw' in obj.dim, KeyError, 'Input must have a time window')
        check(i <= j, ValueError, 'i must be smaller than j')
        check(obj.name in obj.json['Inputs'], KeyError, 'InputTimePart must be call on an input')
        backward_idx = obj.json['Inputs'][obj.name]['tw'][0]
        forward_idx = obj.json['Inputs'][obj.name]['tw'][1]
        check(i >= backward_idx and i <= forward_idx, ValueError, 'i must be in the time window of the input')
        check(j >= backward_idx and j <= forward_idx, ValueError, 'j must be in the time window of the input')
        dim = copy.deepcopy(obj.dim)
        dim['tw']  = j - i
        super().__init__(timepart_relation_name + str(Stream.count),obj.json,dim)
        rel = [timepart_relation_name,[obj.name],[i,j]]
        if offset is not None:
            check(i < offset <= j, IndexError,"The offset must be inside the time window")
            rel.append(offset)
        if (type(obj) is Stream):
            self.json['Relations'][self.name] = rel
        else:
            raise Exception('Type is not supported!')

class TimeSelect(Stream, ToStream):
    def __init__(self, obj, i):
        check('tw' in obj.dim, KeyError, 'Input must have a time window')
        backward_idx = 0
        forward_idx = obj.dim['tw']
        check(i >= backward_idx and i <= forward_idx, ValueError, 'i must be in the time window of the input')
        dim = copy.deepcopy(obj.dim)
        del dim['tw']
        super().__init__(timeselect_relation_name + str(Stream.count),obj.json,dim)
        if (type(obj) is Stream):
            self.json['Relations'][self.name] = [timeselect_relation_name,[obj.name],i]


def createPart(self, *inputs):
    pass

def createSelect(self, *inputs):
    pass

def createTimePart(self, *inputs):
    pass

def createInputTimePart(self, *inputs):
    pass

def createTimeSelect(self, *inputs):
    pass

def createSamplePart(self, *inputs):
    pass

def createInputSamplePart(self, *inputs):
    pass

def createSampleSelect(self, *inputs):
    pass

setattr(Model, part_relation_name, createPart)
setattr(Model, select_relation_name, createSelect)
setattr(Model, timepart_relation_name, createTimePart)
setattr(Model, input_timepart_relation_name, createInputTimePart)
setattr(Model, timeselect_relation_name, createTimeSelect)
setattr(Model, samplepart_relation_name, createSamplePart)
setattr(Model, input_samplepart_relation_name, createInputSamplePart)
setattr(Model, sampleselect_relation_name, createSampleSelect)


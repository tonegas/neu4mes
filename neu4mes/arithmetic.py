import torch.nn as nn
import torch

from neu4mes.relation import ToStream, merge, NeuObj, Stream, Relation
from neu4mes.model import Model
from neu4mes.input import Input

add_relation_name = 'Add'
sub_relation_name = 'Sub'
mul_relation_name = 'Mul'
neg_relation_name = 'Neg'
square_relation_name = 'Square'

class Add(Stream, ToStream):
    def __init__(self, obj1, obj2):
        assert obj1.dim == obj2.dim
        super().__init__(add_relation_name + str(Stream.count),merge(obj1.json,obj2.json),obj1.dim)
        if ((type(obj1) is Input or type(obj1) is Stream) and
                (type(obj2) is Input or type(obj2) is Stream)):
            self.json['Relations'][self.name] = [add_relation_name,[obj1.name,obj2.name]]

class Sub(Stream, ToStream):
    def __init__(self, obj1, obj2):
        super().__init__(sub_relation_name + str(Stream.count),merge(obj1.json,obj2.json),obj1.dim)
        if ((type(obj1) is Input or type(obj1) is Stream) and
                (type(obj2) is Input or type(obj2) is Stream)):
            self.json['Relations'][self.name] = [sub_relation_name,[obj1.name,obj2.name]]

class Mul(Stream, ToStream):
    def __init__(self, obj1, obj2):
        assert obj1.dim == obj2.dim
        super().__init__(mul_relation_name + str(Stream.count),merge(obj1.json,obj2.json),obj1.dim)
        if ((type(obj1) is Input or type(obj1) is Stream) and
                (type(obj2) is Input or type(obj2) is Stream)):
            self.json['Relations'][self.name] = [mul_relation_name,[obj1.name,obj2.name]]

class Neg(Stream, ToStream):
    def __init__(self, obj):
        super().__init__(neg_relation_name+str(Stream.count), obj.json, obj.dim)
        self.json['Relations'][self.name] = [neg_relation_name,[obj.name]]

class Square(NeuObj, Relation, ToStream):
    def __init__(self, obj):
        super().__init__(square_relation_name+str(Stream.count), obj.json, obj.dim)
        self.json['Relations'][self.name] = [square_relation_name,[obj.name]]

class Minus_Layer(nn.Module):
    def __init__(self):
        super(Minus_Layer, self).__init__()

    def forward(self, x):
        return -x

def createMinus(self, *inputs):
    return Minus_Layer()

class Sum_Layer(nn.Module):
    def __init__(self):
        super(Sum_Layer, self).__init__()

    def forward(self, inputs):
        return torch.add(inputs[0], inputs[1])

def createSum(name, *inputs):
    return Sum_Layer()

def createMul(name, *inputs):
    pass

class Diff_Layer(nn.Module):
    def __init__(self):
        super(Diff_Layer, self).__init__()

    def forward(self, inputs):
        # Perform element-wise subtraction
        return torch.stack(inputs).diff(dim=0)

def createSubtract(self, *inputs):
    return Diff_Layer()

class Square_Layer(nn.Module):
    def __init__(self):
        super(Square_Layer, self).__init__()
    def forward(self, x):
        return torch.pow(x,2)

def createSquare(self, *inputs):
    return Square_Layer()

setattr(Model, neg_relation_name, createMinus)
setattr(Model, add_relation_name, createSum)
setattr(Model, mul_relation_name, createMul)
setattr(Model, sub_relation_name, createSubtract)
setattr(Model, square_relation_name, createSquare)
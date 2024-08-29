import torch.nn as nn
import torch

from neu4mes.relation import ToStream, Stream, toStream
from neu4mes.model import Model
from neu4mes.utilis import check, merge


# Binary operators
add_relation_name = 'Add'
sub_relation_name = 'Sub'
mul_relation_name = 'Mul'
div_relation_name = 'Div'
pow_relation_name = 'Pow'

# Unary operators
neg_relation_name = 'Neg'
# square_relation_name = 'Square'

# Merge operator
sum_relation_name = 'Sum'
class Add(Stream, ToStream):
    def __init__(self, obj1:Stream, obj2:Stream) -> Stream:
        obj1,obj2 = toStream(obj1),toStream(obj2)
        check(type(obj1) is Stream,TypeError,
              f"The type of {obj1} is {type(obj1)} and is not supported for add operation.")
        check(type(obj2) is Stream,TypeError,
              f"The type of {obj2} is {type(obj2)} and is not supported for add operation.")
        check(obj1.dim == obj2.dim or obj2.dim == {}, ValueError,
              f"For addition operators (+) the dimension of {obj1.name} = {obj1.dim} must be the same of {obj2.name} = {obj2.dim}.")
        super().__init__(add_relation_name + str(Stream.count),merge(obj1.json,obj2.json),obj1.dim)
        self.json['Relations'][self.name] = [add_relation_name,[obj1.name,obj2.name]]

## TODO: check the scalar dimension, helpful for the offset
class Sub(Stream, ToStream):
    def __init__(self, obj1:Stream, obj2:Stream) -> Stream:
        obj1, obj2 = toStream(obj1), toStream(obj2)
        check(type(obj1) is Stream,TypeError,
              f"The type of {obj1} is {type(obj1)} and is not supported for sub operation.")
        check(type(obj2) is Stream,TypeError,
              f"The type of {obj2} is {type(obj2)} and is not supported for sub operation.")
        check(obj1.dim == obj2.dim or obj2.dim == {}, ValueError,
              f"For subtraction operators (-) the dimension of {obj1.name} = {obj1.dim} must be the same of {obj2.name} = {obj2.dim}.")
        super().__init__(sub_relation_name + str(Stream.count),merge(obj1.json,obj2.json),obj1.dim)
        self.json['Relations'][self.name] = [sub_relation_name,[obj1.name,obj2.name]]

class Mul(Stream, ToStream):
    def __init__(self, obj1:Stream, obj2:Stream) -> Stream:
        obj1, obj2 = toStream(obj1), toStream(obj2)
        check(type(obj1) is Stream, TypeError,
              f"The type of {obj1} is {type(obj1)} and is not supported for mul operation.")
        check(type(obj2) is Stream, TypeError,
              f"The type of {obj2} is {type(obj2)} and is not supported for mul operation.")
        check(obj1.dim == obj2.dim or obj2.dim == {}, ValueError,
              f"For multiplication operators (*) the dimension of {obj1.name} = {obj1.dim} must be the same of {obj2.name} = {obj2.dim}.")
        super().__init__(mul_relation_name + str(Stream.count),merge(obj1.json,obj2.json),obj1.dim)
        self.json['Relations'][self.name] = [mul_relation_name,[obj1.name,obj2.name]]

class Div(Stream, ToStream):
    def __init__(self, obj1:Stream, obj2:Stream) -> Stream:
        obj1, obj2 = toStream(obj1), toStream(obj2)
        check(type(obj1) is Stream, TypeError,
              f"The type of {obj1} is {type(obj1)} and is not supported for div operation.")
        check(type(obj2) is Stream, TypeError,
              f"The type of {obj2} is {type(obj2)} and is not supported for div operation.")
        check(obj1.dim == obj2.dim or obj2.dim == {}, ValueError,
              f"For division operators (*) the dimension of {obj1.name} = {obj1.dim} must be the same of {obj2.name} = {obj2.dim}.")
        super().__init__(div_relation_name + str(Stream.count),merge(obj1.json,obj2.json),obj1.dim)
        self.json['Relations'][self.name] = [div_relation_name,[obj1.name,obj2.name]]

class Pow(Stream, ToStream):
    def __init__(self, obj1:Stream, obj2:Stream) -> Stream:
        obj1, obj2 = toStream(obj1), toStream(obj2)
        check(type(obj1) is Stream, TypeError,
              f"The type of {obj1} is {type(obj1)} and is not supported for exp operation.")
        check(type(obj2) is Stream, TypeError,
              f"The type of {obj2} is {type(obj2)} but must be int or float and is not supported for exp operation.")
        check(obj1.dim == obj2.dim or obj2.dim == {}, ValueError,
              f"For division operators (*) the dimension of {obj1.name} = {obj1.dim} must be the same of {obj2.name} = {obj2.dim}.")
        super().__init__(pow_relation_name + str(Stream.count),merge(obj1.json,obj2.json),obj1.dim)
        self.json['Relations'][self.name] = [pow_relation_name,[obj1.name,obj2.name]]

class Neg(Stream, ToStream):
    def __init__(self, obj:Stream) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for neg operation.")
        super().__init__(neg_relation_name+str(Stream.count), obj.json, obj.dim)
        self.json['Relations'][self.name] = [neg_relation_name,[obj.name]]

# class Square(Stream, ToStream):
#     def __init__(self, obj:Stream) -> Stream:
#         check(type(obj) is Stream, TypeError,
#               f"The type of {obj.name} is {type(obj)} and is not supported for neg operation.")
#         super().__init__(square_relation_name+str(Stream.count), obj.json, obj.dim)
#         self.json['Relations'][self.name] = [square_relation_name,[obj.name]]

class Sum(Stream, ToStream):
    def __init__(self, obj:Stream) -> Stream:
        obj = toStream(obj)
        check(type(obj) is Stream, TypeError,
              f"The type of {obj} is {type(obj)} and is not supported for sum operation.")
        super().__init__(sum_relation_name + str(Stream.count),obj.json,obj.dim)
        self.json['Relations'][self.name] = [sum_relation_name,[obj.name]]

class Add_Layer(nn.Module):
    def __init__(self):
        super(Add_Layer, self).__init__()

    def forward(self, inputs):
        return torch.add(inputs[0], inputs[1])

def createAdd(name, *inputs):
    return Add_Layer()

class Sub_Layer(nn.Module):
    def __init__(self):
        super(Sub_Layer, self).__init__()

    def forward(self, inputs):
        # Perform element-wise subtraction
        return torch.add(inputs[0],-inputs[1])

def createSub(self, *inputs):
    return Sub_Layer()


class Mul_Layer(nn.Module):
    def __init__(self):
        super(Mul_Layer, self).__init__()

    def forward(self, inputs):
        return inputs[0] * inputs[1]

def createMul(name, *inputs):
    return Mul_Layer()

class Div_Layer(nn.Module):
    def __init__(self):
        super(Div_Layer, self).__init__()

    def forward(self, inputs):
        return inputs[0] / inputs[1]

def createDiv(name, *inputs):
    return Div_Layer()

class Pow_Layer(nn.Module):
    def __init__(self):
        super(Pow_Layer, self).__init__()

    def forward(self, inputs):
        return torch.pow(inputs[0], inputs[1])

def createPow(name, *inputs):
    return Pow_Layer()

class Neg_Layer(nn.Module):
    def __init__(self):
        super(Neg_Layer, self).__init__()

    def forward(self, x):
        return -x

def createNeg(self, *inputs):
    return Neg_Layer()

# class Square_Layer(nn.Module):
#     def __init__(self):
#         super(Square_Layer, self).__init__()
#     def forward(self, x):
#         return torch.pow(x,2)

# def createSquare(self, *inputs):
#     return Square_Layer()

class Sum_Layer(nn.Module):
    def __init__(self):
        super(Sum_Layer, self).__init__()

    def forward(self, inputs):
        return torch.sum(inputs, dim = 2)

def createSum(name, *inputs):
    return Sum_Layer()

setattr(Model, add_relation_name, createAdd)
setattr(Model, sub_relation_name, createSub)
setattr(Model, mul_relation_name, createMul)
setattr(Model, div_relation_name, createDiv)
setattr(Model, pow_relation_name, createPow)

setattr(Model, neg_relation_name, createNeg)
# setattr(Model, square_relation_name, createSquare)

setattr(Model, sum_relation_name, createSum)



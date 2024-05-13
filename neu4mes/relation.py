import copy

def merge(source, destination, main = True):
    if main:
        result = copy.deepcopy(destination)
    else:
        result = destination
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = result.setdefault(key, {})
            merge(value, node, False)
        else:
            if key in result and isinstance(result[key], list):
                result[key] = list(set(result[key] + value))
            else:
                result[key] = value

    return result

class NeuObj():
    count = 0
    def __init__(self, name = '', json = {}, dim = 0):
        NeuObj.count = NeuObj.count + 1
        self.name = name
        self.dim = dim
        if json:
            self.json = copy.deepcopy(json)
        else:
            self.json = {
                'SampleTime': 0,
                'Inputs' : {},
                'Parameters' : {},
                'Outputs': {},
                'Relations': {},
                #'Objects': {}
            }
            #self.json['Objects'][name] = self

class Relation():
    def __add__(self, obj):
        from neu4mes.arithmetic import Add
        return Add(self, obj)

    def __sub__(self, obj):
        from neu4mes.arithmetic import Sub
        return Sub(self, obj)

    def __neg__(self):
        from neu4mes.arithmetic import Neg
        return Neg(self)

    def __xor__(self, val):
        from neu4mes.arithmetic import Square
        if val == 2:
            return Square(self)
        else:
            raise Exception("Operation not supported yet")

class Stream(Relation):
    count = 0
    def __init__(self, name, json, dim, count = 1):
        Stream.count = Stream.count + count
        self.name = name
        self.json = copy.deepcopy(json)
        self.dim = dim

class ToStream():
    def __new__(cls, *args):
        out = super(ToStream,cls).__new__(cls)
        out.__init__(*args)
        return Stream(out.name,out.json,out.dim,0)

class AutoToStream():
    def __new__(cls, *args):
        if len(args) > 0 and (issubclass(type(args[0]),NeuObj) or type(args[0]) is Stream):
            instance = super().__new__(cls)
            instance.__init__()
            return instance(args[0])
        instance = super().__new__(cls)
        return instance

#
# object.__mul__(self, other)
# object.__matmul__(self, other)
# object.__truediv__(self, other)
# object.__floordiv__(self, other)
# object.__mod__(self, other)
# object.__divmod__(self, other)
# object.__pow__(self, other[, modulo])
# object.__lshift__(self, other)
# object.__rshift__(self, other)
# object.__and__(self, other)
# object.__xor__(self, other)
# object.__or__(self, other)

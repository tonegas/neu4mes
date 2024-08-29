import copy

MAIN_JSON = {
                'SampleTime': 0,
                'Inputs' : {},
                'States' : {},
                'Functions' : {},
                'Parameters' : {},
                'Outputs': {},
                'Relations': {},
            }


def toStream(obj):
    from neu4mes.parameter import Parameter
    obj = Stream(obj, MAIN_JSON, {}) if type(obj) in (int,float) else obj
    obj = Stream(obj.name, obj.json, obj.dim) if type(obj) is Parameter else obj
    return obj


class NeuObj():
    count = 0
    @classmethod
    def reset_count(self):
        NeuObj.count = 0
    def __init__(self, name, json = {}, dim = 0):
        NeuObj.count += 1
        if CHECK_NAMES == True:
            check(name not in NeuObj_names, NameError, f"The name {name} is already used change the name of NeuObj.")
            NeuObj_names.append(name)
        self.name = name
        self.dim = dim
        if json:
            self.json = copy.deepcopy(json)
        else:
            self.json = copy.deepcopy(MAIN_JSON)

class Relation():
    def __add__(self, obj):
        from neu4mes.arithmetic import Add
        return Add(self, obj)

    def __sub__(self, obj):
        from neu4mes.arithmetic import Sub
        return Sub(self, obj)

    def __truediv__(self, obj):
        from neu4mes.arithmetic import Div
        return Div(self, obj)

    def __mul__(self, obj):
        from neu4mes.arithmetic import Mul
        return Mul(self, obj)

    def __pow__(self, obj):
        from neu4mes.arithmetic import Pow
        return Pow(self, obj)

    def __neg__(self):
        from neu4mes.arithmetic import Neg
        return Neg(self)

class Stream(Relation):
    count = 0
    @classmethod
    def reset_count(self):
        Stream.count = 0

    def __init__(self, name, json, dim, count = 1):
        Stream.count += count
        self.name = name
        self.json = copy.deepcopy(json)
        self.dim = dim

    def tw(self, tw, offset = None):
        from neu4mes.input import State, Connect
        from neu4mes.utilis import merge
        s = State(self.name+"_state",dimensions=self.dim)
        if type(tw) == int:
            out_connect = Connect(self, s)
            win_state = s.tw(tw, offset)
            return Stream(win_state.name, merge(win_state.json, out_connect.json), win_state.dim,0 )

    def sw(self, sw, offset = None):
        from neu4mes.input import State, Connect
        from neu4mes.utilis import merge
        s = State(self.name+"_state",dimensions=self.dim)
        if type(sw) == int:
            out_connect = Connect(self, s)
            win_state = s.sw(sw, offset)
            return Stream(win_state.name, merge(win_state.json, out_connect.json), win_state.dim,0 )

    def z(self, delay):
        from neu4mes.input import State, Connect
        from neu4mes.utilis import merge
        s = State(self.name + "_state",dimensions=self.dim)
        if type(delay) == int and delay > 0:
            out_connect = Connect(self, s)
            win_state = s.z(delay)
            return Stream(win_state.name, merge(win_state.json, out_connect.json), win_state.dim,0 )


class ToStream():
    def __new__(cls, *args, **kwargs):
        out = super(ToStream,cls).__new__(cls)
        out.__init__(*args, **kwargs)
        return Stream(out.name,out.json,out.dim,0)

class AutoToStream():
    def __new__(cls, *args,  **kwargs):
        if len(args) > 0 and (issubclass(type(args[0]),NeuObj) or type(args[0]) is Stream):
            instance = super().__new__(cls)
            instance.__init__()
            return instance(args[0])
        instance = super().__new__(cls)
        return instance

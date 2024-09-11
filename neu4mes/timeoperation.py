import torch.nn as nn
import torch

from neu4mes.relation import NeuObj, AutoToStream, Stream
from neu4mes.utilis import check, merge
from neu4mes.input import Input, State
from neu4mes.model import Model

# Binary operators
int_relation_name = 'Int'


class Int(NeuObj, AutoToStream):
    def __init__(self, method:str = 'ForwardEuler'):
        self.method = method
        super().__init__(int_relation_name + str(NeuObj.count))

    def __call__(self, obj:Stream) -> Stream:
        from neu4mes.input import State, ClosedLoop
        from neu4mes.parameter import Parameter
        from neu4mes.initializer import init_constant
        s = State(self.name + "_last", dimensions=obj.dim['dim'])
        if self.method == 'ForwardEuler':
            DT = Parameter('DT',dimensions=obj.dim['dim'],init=init_constant,init_params={'value':'DT'})
            new_s = s.last() + obj * DT
        out_connect = ClosedLoop(new_s, s)
        return Stream(new_s.name, merge(new_s.json, out_connect.json), new_s.dim, 1)

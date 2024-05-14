import numpy as np

from neu4mes.relation import NeuObj, Stream

class Input(NeuObj, Stream):
    def __init__(self, name, values = None):
        NeuObj.__init__(self, name)
        self.json['Inputs'][self.name] = {}
        self.dim = {'dim_in':1}
        if values:
            self.values = values
            self.json['Inputs'][self.name] = {
                'Discrete' : values
            }
        Stream.__init__(self, name, self.json, self.dim)

    def tw(self, tw, offset = None):
         if offset is not None:
             return  Stream((self.name, tw, offset),self.json,{'tw_in':tw})
         return Stream((self.name, tw),self.json,{'tw_in':tw})

    def z(self, advance):
        if advance > 0:
            return Stream(self.name+'__+z'+str(advance),self.json,{'dim_in':1})
        else:
            return Stream(self.name+'__-z'+str(-advance),self.json, {'dim_in':1})
    #
    # def s(self, derivate):
    #     if derivate > 0:
    #         return self, '__+s'+str(derivate)
    #     else:
    #         return self, '__-s'+str(-derivate)

# class ControlInput(Input):
#     def __init__(self,name,values = None):
#         super().__init__(name,values)

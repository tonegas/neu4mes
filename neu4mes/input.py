import copy

import numpy as np

from neu4mes.relation import NeuObj, Stream

class Input(NeuObj, Stream):
    def __init__(self, name, dimensions = 1, values = None):
        NeuObj.__init__(self, name)
        self.json['Inputs'][self.name] = {'dim': dimensions, 'tw': [0, 0]}
        self.dim = {'dim': dimensions}
        if values:
            self.values = values
            self.json['Inputs'][self.name]['discrete'] = values
        Stream.__init__(self, name, self.json, self.dim)

    def tw(self, tw, offset = None):
        dim = copy.deepcopy(self.dim)
        if type(tw) is list:
            self.json['Inputs'][self.name]['tw'] = tw
        else:
            self.json['Inputs'][self.name]['tw'][0] = -tw
        dim['tw'] = tw
        if offset is not None:
            return Stream((self.name, self.json['Inputs'][self.name]['tw'], offset), self.json, dim)
        return Stream((self.name, self.json['Inputs'][self.name]['tw']), self.json, dim)

    def z(self, advance):
        if advance > 0:
            return Stream(self.name+'__+z'+str(advance),self.json,{'dim_in':1}) #TODO To be change
        else:
            return Stream(self.name+'__-z'+str(-advance),self.json, {'dim_in':1}) #TODO To be change
    #
    # def s(self, derivate):
    #     if derivate > 0:
    #         return self, '__+s'+str(derivate)
    #     else:
    #         return self, '__-s'+str(-derivate)

# class ControlInput(Input):
#     def __init__(self,name,values = None):
#         super().__init__(name,values)

import copy

import numpy as np

from neu4mes.relation import NeuObj, Stream

class Input(NeuObj, Stream):
    def __init__(self, name, dimensions = 1, values = None):
        NeuObj.__init__(self, name)
        self.json['Inputs'][self.name] = {'dim': dimensions, 'tw': [0, 0], 'sw': [0,0] }
        self.dim = {'dim': dimensions}
        if values:
            self.values = values
            self.json['Inputs'][self.name]['discrete'] = values
        Stream.__init__(self, name, self.json, self.dim)

    def tw(self, tw, offset = None):
        dim = copy.deepcopy(self.dim)
        json = copy.deepcopy(self.json)
        if type(tw) is list:
            json['Inputs'][self.name]['tw'] = tw
            tw = tw[1] - tw[0]
        else:
            assert tw > 0, "The time window must be positive"
            json['Inputs'][self.name]['tw'][0] = -tw
        dim['tw'] = tw
        if offset is not None:
            return Stream((self.name, {'tw':json['Inputs'][self.name]['tw'], 'offset':offset}), json, dim)
        return Stream((self.name,  {'tw':json['Inputs'][self.name]['tw']}), json, dim)

    def sw(self, sw, offset=None):
        dim = copy.deepcopy(self.dim)
        json = copy.deepcopy(self.json)
        if type(sw) is list:
            assert type(sw[1]) is int and type(sw[1]) is int, "The type of sample window must be an integer"
            json['Inputs'][self.name]['sw'] = sw
            sw = sw[1] - sw[0]
        else:
            assert type(sw) is int, "The type of sample window must be an integer"
            assert sw > 0, "The time window must be positive"
            json['Inputs'][self.name]['sw'][0] = -sw
        dim['sw'] = sw
        if offset is not None:
            return Stream((self.name, {'sw': json['Inputs'][self.name]['sw'], 'offset': offset}), json, dim)
        return Stream((self.name, {'sw': json['Inputs'][self.name]['sw']}), json, dim)

    def z(self, advance):
        if advance > 0:
            self.json['Inputs'][self.name]['sw'][0] = advance
        else:
            self.json['Inputs'][self.name]['sw'][1] = -advance
        return Stream((self.name, {'z':advance}), self.json, self.dim)

    def s(self, derivate):
        return Stream((self.name, {'s':derivate}), self.json, self.dim)

    # def s(self, derivate):
    #     if derivate > 0:
    #         return self, '__+s'+str(derivate)
    #     else:
    #         return self, '__-s'+str(-derivate)

# class ControlInput(Input):
#     def __init__(self,name,values = None):
#         super().__init__(name,values)

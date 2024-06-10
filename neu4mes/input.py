import copy

import numpy as np

from neu4mes.relation import NeuObj, Stream
from neu4mes.utilis import check

class Input(NeuObj, Stream):
    def __init__(self, name, dimensions:int = 1, values = None):
        NeuObj.__init__(self, name)
        check(type(dimensions) == int, TypeError,"The dimensions must be a integer")
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
            check(tw >0, ValueError,"The time window must be positive")
            json['Inputs'][self.name]['tw'][0] = -tw
        dim['tw'] = tw
        if offset is not None:
            return Stream((self.name, {'tw':json['Inputs'][self.name]['tw'], 'offset':offset}), json, dim)
        return Stream((self.name,  {'tw':json['Inputs'][self.name]['tw']}), json, dim)

    def sw(self, sw, offset=None):
        dim = copy.deepcopy(self.dim)
        json = copy.deepcopy(self.json)
        if type(sw) is list:
            check(type(sw[0]) == int and type(sw[1]) == int, TypeError, "The sample window must be integer")
            json['Inputs'][self.name]['sw'] = sw
            sw = sw[1] - sw[0]
        else:
            check(type(sw) == int, TypeError, "The sample window must be integer")
            check(sw > 0, ValueError, "The time window must be positive")
            json['Inputs'][self.name]['sw'][0] = -sw
        dim['sw'] = sw
        if offset is not None:
            return Stream((self.name, {'sw': json['Inputs'][self.name]['sw'], 'offset': offset}), json, dim)
        return Stream((self.name, {'sw': json['Inputs'][self.name]['sw']}), json, dim)

    def z(self, delay):
        self.json['Inputs'][self.name]['sw'] = [(-delay)-1,(-delay)]
        return Stream((self.name, {'sw':self.json['Inputs'][self.name]['sw']}), self.json, self.dim)

    # def s(self, derivate):
    #     return Stream((self.name, {'s':derivate}), self.json, self.dim)

import copy

import numpy as np

from neu4mes.relation import NeuObj, Stream
from neu4mes.utilis import check
from neu4mes.visualizer import Visualizer
from neu4mes.part import InputSamplePart, InputTimePart

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
            json['Inputs'][self.name]['tw'][0] = -tw
        check(tw > 0, ValueError, "The time window must be positive")
        dim['tw'] = tw
        if offset is not None:
            check(json['Inputs'][self.name]['tw'][0] < offset <= json['Inputs'][self.name]['tw'][1],
                  IndexError,
                  "The offset must be inside the time window")
        return InputTimePart(Stream(self.name, json, dim), json['Inputs'][self.name]['tw'][0], json['Inputs'][self.name]['tw'][1], offset)

    # Select a sample window
    # Example T = [-3,-2,-1,0,1,2]       # time vector 0 represent the last passed instant
    # If sw is an integer #1 represent the number of step in the past
    # T.s(2)                = [-1, 0]    # represents two time step in the past
    # If sw is a list [#1,#2] the numbers represent the time index in the vector second element excluded
    # T.s([-2,0])           = [-1, 0]    # represents two time step in the past zero in the future
    # T.s([0,1])            = [1]        # the first time in the future
    # T.s([-4,-2])          = [-3,-2]
    # The total number of samples can be computed #2-#1
    # The offset represent the index of the vector that need to be used to offset the window
    # T.s(2,offset=-1)      = [0, 1]      # the value of the window is [-1,0] offest by -1 the value at the index -1
    # T.s([-2,2],offset=0)  = [-1,0,1,2]  # the value of the window is [-1,0,1,2] offset by 0 the value at the index 0
    def sw(self, sw, offset = None):
        dim = copy.deepcopy(self.dim)
        json = copy.deepcopy(self.json)
        if type(sw) is list:
            check(type(sw[0]) == int and type(sw[1]) == int, TypeError, "The sample window must be integer")
            json['Inputs'][self.name]['sw'] = sw
            sw = sw[1] - sw[0]
        else:
            check(type(sw) == int, TypeError, "The sample window must be integer")
            json['Inputs'][self.name]['sw'][0] = -sw
        check(sw > 0, ValueError, "The time window must be positive")
        dim['sw'] = sw
        if offset is not None:
            check(json['Inputs'][self.name]['sw'][0] < offset <= json['Inputs'][self.name]['sw'][1],
                  IndexError,
                  "The offset must be inside the time window")
        return InputSamplePart(Stream(self.name, json, dim), json['Inputs'][self.name]['sw'][0], json['Inputs'][self.name]['sw'][1], offset)

    # def sw(self, sw, offset=None):
    #     dim = copy.deepcopy(self.dim)
    #     json = copy.deepcopy(self.json)
    #     if type(sw) is list:
    #         check(type(sw[0]) == int and type(sw[1]) == int, TypeError, "The sample window must be integer")
    #         json['Inputs'][self.name]['sw'] = sw
    #         sw = sw[1] - sw[0]
    #     else:
    #         check(type(sw) == int, TypeError, "The sample window must be integer")
    #         json['Inputs'][self.name]['sw'][0] = -sw
    #     check(sw > 0, ValueError, "The time window must be positive")
    #     dim['sw'] = sw
    #     if offset is not None:
    #         check(json['Inputs'][self.name]['sw'][0] < offset <= json['Inputs'][self.name]['sw'][1], IndexError,
    #               "The offset must be inside the time window")
    #         return Stream((self.name, {'sw': json['Inputs'][self.name]['sw'], 'offset': offset}), json, dim)
    #     return Stream((self.name, {'sw': json['Inputs'][self.name]['sw']}), json, dim)

    # Select the unitary delay
    # Example T = [-3,-2,-1,0,1,2] # time vector 0 represent the last passed instant
    # T.z(-1) = 1
    # T.z(0)  = 0 #the last passed instant
    # T.z(2)  = -2
    def z(self, delay):
        dim = copy.deepcopy(self.dim)
        json = copy.deepcopy(self.json)
        json['Inputs'][self.name]['sw'] = [(-delay)-1,(-delay)]
        return InputSamplePart(Stream(self.name, json, dim), json['Inputs'][self.name]['sw'][0], json['Inputs'][self.name]['sw'][1], None)

    # def s(self, derivate):
    #     return Stream((self.name, {'s':derivate}), self.json, self.dim)

import copy, inspect, textwrap
import numpy as np

from neu4mes.relation import NeuObj, Stream
from neu4mes.utilis import check

#values = zeros, ones, 1order, linear, quadratic

class Constant(NeuObj, Stream):
    def __init__(self, name:str, values:list, tw:int|None = None, sw:int|None = None):
        NeuObj.__init__(self, name)
        shape = np.array(values).shape
        if len(shape) == 0:
            self.dim = {'dim': 1}
        else:
            check(len(shape) >= 2, ValueError,
              f"The shape of a Constant must have at least 2 dimensions or zero.")
            dimensions = shape[1] if len(shape[1:]) == 1 else list(shape[1:])
            self.dim = {'dim': dimensions}
            if tw is not None:
                check(sw is None, ValueError, "If tw is set sw must be None")
                self.dim['tw'] = tw
            elif sw is not None:
                self.dim['sw'] = sw
                check(shape[0] == self.dim['sw'],ValueError, f"The sw = {sw} is different from sw = {shape[0]} of the values.")
            else:
                self.dim['sw'] = shape[0]

        # deepcopy dimention information inside Parameters
        self.json['Constants'][self.name] = copy.deepcopy(self.dim)
        self.json['Constants'][self.name]['values'] = values
        Stream.__init__(self, name, self.json, self.dim)

class Parameter(NeuObj, Stream):
    def __init__(self, name:str, dimensions:int|list|tuple|None = None, tw:int|None = None, sw:int|None = None, values:list|None = None, init:None = None, init_params:None = None):
        NeuObj.__init__(self, name)
        if values is None:
            if dimensions is None:
                dimensions = 1
            self.dim = {'dim': list(dimensions) if type(dimensions) is tuple else dimensions}
            if tw is not None:
                check(sw is None, ValueError, "If tw is set sw must be None")
                self.dim['tw'] = tw
            if sw is not None:
                self.dim['sw'] = sw

            # deepcopy dimention information inside Parameters
            self.json['Parameters'][self.name] = copy.deepcopy(self.dim)
        else:
            shape = np.array(values).shape
            check(len(shape) >= 2, ValueError,
                  f"The shape of a parameter must have at least 2 dimensions.")
            values_dimensions = shape[1] if len(shape[1:]) == 1 else shape[1:]
            if dimensions is None:
                dimensions = values_dimensions
            else:
                check(dimensions == values_dimensions, ValueError,
                      f"The dimensions = {dimensions} are different from dimensions = {values_dimensions} of the values.")
            self.dim = {'dim': dimensions}

            if tw is not None:
                check(sw is None, ValueError, "If tw is set sw must be None")
                self.dim['tw'] = tw
            elif sw is not None:
                self.dim['sw'] = sw
                check(shape[0] == self.dim['sw'],ValueError, f"The sw = {sw} is different from sw = {shape[0]} of the values.")
            else:
                self.dim['sw'] = shape[0]

            # deepcopy dimention information inside Parameters
            self.json['Parameters'][self.name] = copy.deepcopy(self.dim)
            self.json['Parameters'][self.name]['values'] = values

        if init is not None:
            check('values' not in self.json['Parameters'][self.name], ValueError, f"The parameter {self.name} is already initialized.")
            check(inspect.isfunction(init), ValueError,f"The init parameter must be a function.")
            code = textwrap.dedent(inspect.getsource(init)).replace('\"', '\'')
            self.json['Parameters'][self.name]['init_fun'] = { 'code' : code, 'name' : init.__name__}
            if init_params is not None:
                self.json['Parameters'][self.name]['init_fun']['params'] = init_params

        Stream.__init__(self, name, self.json, self.dim)

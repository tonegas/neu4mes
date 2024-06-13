import copy

from neu4mes.relation import NeuObj, Stream

#values = zeros, ones, 1order, linear, quadratic

class Parameter(NeuObj, Stream):
    def __init__(self, name:str, dimensions = 1, tw = None, sw = None):
        NeuObj.__init__(self, name)
        self.dim = {'dim': dimensions}
        if tw is not None:
            self.dim['tw'] = tw
        if sw is not None:
            self.dim['sw'] = sw
        # deepcopy dimention information inside Parameters
        self.json['Parameters'][self.name] = copy.deepcopy(self.dim)

        Stream.__init__(self, name, self.json, self.dim)

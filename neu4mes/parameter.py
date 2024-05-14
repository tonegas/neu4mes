import copy

from neu4mes.relation import NeuObj, Stream


class Parameter(NeuObj, Stream):
    def __init__(self, name, dimensions=1, tw=None):
        NeuObj.__init__(self, name)
        self.dim = {'dim': dimensions}
        if tw is not None:
            self.dim['tw'] = tw
        # deepcopy dimention information inside Parameters
        self.json['Parameters'][self.name] = copy.deepcopy(self.dim)

        Stream.__init__(self, name, self.json, self.dim)

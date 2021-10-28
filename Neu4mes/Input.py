from Neu4mes import NeuObj

class Input(NeuObj.NeuObj):
    def __init__(self,name):
        super().__init__()
        self.name = name
        self.max_tw = 0
        self.json['Inputs'][self.name] = {}

    def tw(self, seconds):
        self.max_tw = seconds
        return self, seconds 
    
    def z(self, advance):
        if advance > 0:
            return self, '__+z'+str(advance)
        else:
            return self, '__-z'+str(-advance)

    def s(self, derivate):
        if derivate > 0:
            return self, '__+s'+str(derivate)
        else:
            return self, '__-s'+str(-derivate)

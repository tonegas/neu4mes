from Neu4mes import NeuObj

class Relation(NeuObj.NeuObj):      
    def __init__(self,json):
        super().__init__(json)

    def __neg__(self):
        from Neu4mes import Minus
        return Minus(self)
    
    def __add__(self, obj): 
        from Neu4mes import Sum
        if type(obj) is not Sum and type(self) is not Sum:
            return Sum(self, obj)
        elif type(obj) is Sum:
            return obj.sum(self)
        else:
            return self.sum(obj)        

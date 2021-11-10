from Neu4mes import NeuObj
import types

class Relation(NeuObj.NeuObj):      
    def __init__(self,json):
        super().__init__(json)

    def __neg__(self):
        from Neu4mes import Minus
        return Minus(self)
    
    def __add__(self, obj): 
        from Neu4mes import Sum
        return Sum(self, obj)

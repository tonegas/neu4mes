import NeuObj
import Sum    

class Relation(NeuObj.NeuObj):      
    def __init__(self,json):
        super().__init__(json)

    def __neg__(self):
        obj_name = self.name
        r = Relation(self.json)
        r.name = self.name+'_minus'
        r.json['Relations'][r.name] = {
            'Minus':[obj_name]
        }
        return r
    
    def __add__(self, obj):
        if type(obj) is not Sum.Sum and type(self) is not Sum:
            return Sum.Sum(self, obj)
        elif type(obj) is Sum.Sum:
            return obj.sum(self)
        else:
            return self.sum(obj)        

    def setInput(self, model, relvalue):
        pass

    def createElem(self, model, relvalue, outel):
        pass
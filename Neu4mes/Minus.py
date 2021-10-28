from Neu4mes import Relation, NeuObj
import tensorflow.keras.layers

class Minus(Relation):
    def __init__(self, obj = None):
        if obj is None:
            return
        super().__init__(obj.json)
        obj_name = obj.name
        self.name = obj.name+'_minus'
        self.json['Relations'][self.name] = {
            'Minus':[obj_name]
        }
    
    def createElem(self, name, input):
        return -input
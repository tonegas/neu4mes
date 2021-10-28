from Neu4mes import Relation, Input
import tensorflow.keras.layers

class Linear(Relation):
    def __init__(self, obj = None):
        self.name = ''
        if obj is None:
            return
        if type(obj) is tuple:
            super().__init__(obj[0].json)
            self.name = obj[0].name+'_lin'
            self.json['Relations'][self.name] = {
                'Linear':[(obj[0].name,obj[1])],
            }
        elif type(obj) is Input:
            super().__init__(obj.json)
            self.name = obj.name+'_lin'
            self.json['Relations'][self.name] = {
                'Linear':[obj.name]
            }
        elif type(obj) is Relation:
            super().__init__(obj.json)
            self.name = obj.name+'_lin'
            self.json['Relations'][self.name] = {
                'Linear':[obj.name]
            }
        else:
            raise Exception('Type is not supported!')
    
    def createElem(self, name, input):
        return tensorflow.keras.layers.Dense(units = 1, activation = None, use_bias = None, name = name)(input)

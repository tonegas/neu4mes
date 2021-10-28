from Neu4mes import Relation, Input
import tensorflow.keras.layers

class Relu(Relation):
    def __init__(self, obj = None):
        if obj is None:
            return
        self.name = ''
        if type(obj) is tuple:
            super().__init__(obj[0].json)
            self.name = obj[0].name+'_relu'
            self.json['Relations'][self.name] = {
                'Relu':[(obj[0].name,obj[1])],
            }
        elif type(obj) is Input:
            super().__init__(obj.json)
            self.name = obj.name+'_relu'
            self.json['Relations'][self.name] = {
                'Relu':[obj.name]
            }
        elif issubclass(type(obj),Relation):
            super().__init__(obj.json)
            self.name = obj.name+'_relu'
            self.json['Relations'][self.name] = {
                'Relu':[obj.name]
            }
        else:
            raise Exception('Type is not supported!')

    def createElem(self, name, input):
        return tensorflow.keras.layers.ReLU(name = name)(input)



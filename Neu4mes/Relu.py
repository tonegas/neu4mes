import neu4mes
import tensorflow.keras.layers

relu_relation_name = 'ReLU'

class Relu(neu4mes.Relation):
    def __init__(self, obj = None):
        if obj is None:
            return
        self.name = ''
        if type(obj) is tuple:
            super().__init__(obj[0].json)
            self.name = obj[0].name+'_relu'
            self.json['Relations'][self.name] = {
                relu_relation_name:[(obj[0].name,obj[1])],
            }
        elif type(obj) is neu4mes.Input:
            super().__init__(obj.json)
            self.name = obj.name+'_relu'
            self.json['Relations'][self.name] = {
                relu_relation_name:[obj.name]
            }
        elif issubclass(type(obj),neu4mes.Relation):
            super().__init__(obj.json)
            self.name = obj.name+'_relu'
            self.json['Relations'][self.name] = {
                relu_relation_name:[obj.name]
            }
        else:
            raise Exception('Type is not supported!')

    def createElem(self, name, input):
        return 


def createRelu(self, name, input):
    return tensorflow.keras.layers.ReLU(name = name)(input)

setattr(neu4mes.Neu4mes, relu_relation_name, createRelu)
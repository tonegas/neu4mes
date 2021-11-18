import neu4mes
import tensorflow.keras.backend

class Sin(neu4mes.Relation):
    def __init__(self, obj):
        self.name = ''
        if type(obj) is tuple:
            super().__init__(obj[0].json)
            self.name = obj[0].name+'_sin'+str(neu4mes.NeuObj.count)
            self.json['Relations'][self.name] = {
                'Sin':[(obj[0].name,obj[1])],
            }
        elif (type(obj) is neu4mes.Input or
              type(obj) is neu4mes.Relation or
              issubclass(type(obj), neu4mes.Relation)):
            super().__init__(obj.json)
            self.name = obj.name+'_sin'+str(neu4mes.NeuObj.count)
            self.json['Relations'][self.name] = {
                'Sin':[obj.name]
            }
        else:
            raise Exception('Type is not supported!')



class Cos(neu4mes.Relation):
    def __init__(self, obj):
        self.name = ''
        if type(obj) is tuple:
            super().__init__(obj[0].json)
            self.name = obj[0].name+'_cos'+str(neu4mes.NeuObj.count)
            self.json['Relations'][self.name] = {
                'Cos':[(obj[0].name,obj[1])],
            }
        elif (type(obj) is neu4mes.Input or
              type(obj) is neu4mes.Relation or
              issubclass(type(obj), neu4mes.Relation)):
            super().__init__(obj.json)
            self.name = obj.name+'_cos'+str(neu4mes.NeuObj.count)
            self.json['Relations'][self.name] = {
                'Cos':[obj.name]
            }
        else:
            raise Exception('Type is not supported!')



class Tan(neu4mes.Relation):
    def __init__(self, obj):
        self.name = ''
        if type(obj) is tuple:
            super().__init__(obj[0].json)
            self.name = obj[0].name+'_tan'+str(neu4mes.NeuObj.count)
            self.json['Relations'][self.name] = {
                'Tan':[(obj[0].name,obj[1])],
            }
        elif (type(obj) is neu4mes.Input or
              type(obj) is neu4mes.Relation or
              issubclass(type(obj), neu4mes.Relation)):
            super().__init__(obj.json)
            self.name = obj.name+'_tan'+str(neu4mes.NeuObj.count)
            self.json['Relations'][self.name] = {
                'Tan':[obj.name]
            }
        else:
            raise Exception('Type is not supported!')


def createSin(self, name, input):
    return tensorflow.keras.backend.sin(input)
def createCos(self, name, input):
    return tensorflow.keras.backend.cos(input)
def createTan(self, name, input):
    return tensorflow.keras.backend.tan(input)


setattr(neu4mes.Neu4mes, 'Sin', createSin)
setattr(neu4mes.Neu4mes, 'Tan', createTan)
setattr(neu4mes.Neu4mes, 'Cos', createCos)
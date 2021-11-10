import Neu4mes
import tensorflow.keras.backend

class Sin(Neu4mes.Relation):
    def __init__(self, obj):
        self.name = ''
        if type(obj) is tuple:
            super().__init__(obj[0].json)
            self.name = obj[0].name+'_sin'+str(Neu4mes.NeuObj.count)
            self.json['Relations'][self.name] = {
                'Sin':[(obj[0].name,obj[1])],
            }
        elif (type(obj) is Neu4mes.Input or
              type(obj) is Neu4mes.Relation or
              issubclass(type(obj), Neu4mes.Relation)):
            super().__init__(obj.json)
            self.name = obj.name+'_sin'+str(Neu4mes.NeuObj.count)
            self.json['Relations'][self.name] = {
                'Sin':[obj.name]
            }
        else:
            raise Exception('Type is not supported!')

def createSin(self, name, input):
    return tensorflow.keras.backend.sin(input)

setattr(Neu4mes.Neu4mes, 'Sin', createSin)

class Cos(Neu4mes.Relation):
    def __init__(self, obj):
        self.name = ''
        if type(obj) is tuple:
            super().__init__(obj[0].json)
            self.name = obj[0].name+'_cos'+str(Neu4mes.NeuObj.count)
            self.json['Relations'][self.name] = {
                'Cos':[(obj[0].name,obj[1])],
            }
        elif (type(obj) is Neu4mes.Input or
              type(obj) is Neu4mes.Relation or
              issubclass(type(obj), Neu4mes.Relation)):
            super().__init__(obj.json)
            self.name = obj.name+'_cos'+str(Neu4mes.NeuObj.count)
            self.json['Relations'][self.name] = {
                'Cos':[obj.name]
            }
        else:
            raise Exception('Type is not supported!')

def createCos(self, name, input):
    return tensorflow.keras.backend.cos(input)

setattr(Neu4mes.Neu4mes, 'Cos', createCos)

class Tan(Neu4mes.Relation):
    def __init__(self, obj):
        self.name = ''
        if type(obj) is tuple:
            super().__init__(obj[0].json)
            self.name = obj[0].name+'_tan'+str(Neu4mes.NeuObj.count)
            self.json['Relations'][self.name] = {
                'Tan':[(obj[0].name,obj[1])],
            }
        elif (type(obj) is Neu4mes.Input or
              type(obj) is Neu4mes.Relation or
              issubclass(type(obj), Neu4mes.Relation)):
            super().__init__(obj.json)
            self.name = obj.name+'_tan'+str(Neu4mes.NeuObj.count)
            self.json['Relations'][self.name] = {
                'Tan':[obj.name]
            }
        else:
            raise Exception('Type is not supported!')

def createTan(self, name, input):
    return tensorflow.keras.backend.tan(input)

setattr(Neu4mes.Neu4mes, 'Tan', createTan)
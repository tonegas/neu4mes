import Neu4mes
import tensorflow.keras.layers

#Linear_json
# 'Relations':{
#     'singal_name':{
#         'Linear':[(obj[0].name,obj[1])]
#     }
# }
localmodel_relation_name = 'LocalModel'


import random
import string
def rand(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

class LocalModel(Neu4mes.Relation):
    def __init__(self, obj1, obj2):
        self.name = ''
        if type(obj1) is tuple and type(obj2) is Neu4mes.DiscreteInput:
            super().__init__(obj1[0].json)
            self.json = Neu4mes.NeuObj.merge(obj1[0].json,obj2.json)
            self.name = obj1[0].name+'X'+obj2.name+rand(3)
            self.json['Relations'][self.name] = {
                localmodel_relation_name:[(obj1[0].name,obj1[1]),obj2.name],
            }
        else:
            raise Exception('Type is not supported!')

def createLocalModel(self, name, input):
    localModels = tensorflow.keras.layers.Dense(units = 8, activation = None, use_bias = None, name = name)(input[0])
    return tensorflow.keras.layers.Multiply()([localModels,input[1]])

setattr(Neu4mes.Neu4mes, localmodel_relation_name, createLocalModel)
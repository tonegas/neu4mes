import neu4mes
import tensorflow.keras.layers

#Linear_json
# 'Relations':{
#     'singal_name':{
#         'Linear':[(obj[0].name,obj[1])]
#     }
# }
localmodel_relation_name = 'LocalModel'

class LocalModel(neu4mes.Relation):
    def __init__(self, obj1, obj2):
        self.name = ''
        if type(obj1) is tuple and obj2.values is not None:
            super().__init__(obj1[0].json)
            self.json = neu4mes.merge(obj1[0].json,obj2.json)
            self.name = obj1[0].name+'X'+obj2.name+'_loc'+str(neu4mes.NeuObj.count)
            self.json['Relations'][self.name] = {
                localmodel_relation_name:[(obj1[0].name,obj1[1]),obj2.name],
            }
        else:
            raise Exception('Type is not supported!')

def createLocalModel(self, name, input):
    localModels = tensorflow.keras.layers.Dense(units = 8, activation = None, use_bias = None, name = name)(input[0])
    return tensorflow.keras.layers.Dot(axes=1)([localModels,input[1]])

setattr(neu4mes.Neu4mes, localmodel_relation_name, createLocalModel)
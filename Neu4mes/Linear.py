import Neu4mes
import tensorflow.keras.layers

#Linear_json
# 'Relations':{
#     'singal_name':{
#         'Linear':[(obj[0].name,obj[1])]
#     }
# }
linear_relation_name = 'Linear'


import random
import string
def rand(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

class Linear(Neu4mes.Relation):
    def __init__(self, obj):
        self.name = ''
        if type(obj) is tuple:
            super().__init__(obj[0].json)
            self.name = obj[0].name+'_lin'+str(Neu4mes.NeuObj.count)
            self.json['Relations'][self.name] = {
                linear_relation_name:[(obj[0].name,obj[1])],
            }
        elif type(obj) is Neu4mes.Input or type(obj) is Neu4mes.DiscreteInput:
            super().__init__(obj.json)
            self.name = obj.name+'_lin'+str(Neu4mes.NeuObj.count)
            self.json['Relations'][self.name] = {
                linear_relation_name:[obj.name]
            }
        elif type(obj) is Neu4mes.Relation:
            super().__init__(obj.json)
            self.name = obj.name+'_lin'+str(Neu4mes.NeuObj.count)
            self.json['Relations'][self.name] = {
                linear_relation_name:[obj.name]
            }
        else:
            raise Exception('Type is not supported!')

def createLinear(self, name, input):
    return tensorflow.keras.layers.Dense(units = 1, activation = None, use_bias = None, name = name)(input)

setattr(Neu4mes.Neu4mes, linear_relation_name, createLinear)
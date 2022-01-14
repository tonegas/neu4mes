import neu4mes
import tensorflow.keras.layers
import tensorflow.keras.backend as K

#Linear_json
# 'Relations':{
#     'singal_name':{
#         'Linear':[(obj[0].name,obj[1])]
#     }
# }
linear_relation_name = 'Linear'
linear_bias_relation_name = 'LinearBias'

class Linear(neu4mes.Relation):
    def __init__(self, obj):
        self.name = ''
        if type(obj) is tuple:
            super().__init__(obj[0].json)
            self.name = obj[0].name+'_lin'+str(neu4mes.NeuObj.count)
            if type(obj[1]) is list:
                if len(obj) == 2:
                    self.json['Relations'][self.name] = {
                        linear_relation_name:[(obj[0].name,(obj[1][0],obj[1][1]))],
                    }
                elif len(obj) == 3:
                    self.json['Relations'][self.name] = {
                        linear_relation_name:[(obj[0].name,(obj[1][0],obj[1][1]),obj[2])],
                    }
                else:
                    raise Exception('Type is not supported!')
            else:
                self.json['Relations'][self.name] = {
                    linear_relation_name:[(obj[0].name,obj[1])],
                }
        elif (type(obj) is neu4mes.Input or
            issubclass(type(obj),neu4mes.Input) or
            type(obj) is neu4mes.Relation or
            issubclass(type(obj), neu4mes.Relation)):
            super().__init__(obj.json)
            self.name = obj.name+'_lin'+str(neu4mes.NeuObj.count)
            self.json['Relations'][self.name] = {
                linear_relation_name:[obj.name]
            }
        else:
            raise Exception('Type is not supported!')

class LinearBias(neu4mes.Relation):
    def __init__(self, obj):
        self.name = ''
        if type(obj) is tuple:
            super().__init__(obj[0].json)
            self.name = obj[0].name+'_lin_bias'+str(neu4mes.NeuObj.count)
            if type(obj[1]) is list:
                if len(obj) == 2:
                    self.json['Relations'][self.name] = {
                        linear_bias_relation_name:[(obj[0].name,(obj[1][0],obj[1][1]))],
                    }
                elif len(obj) == 3:
                    self.json['Relations'][self.name] = {
                        linear_bias_relation_name:[(obj[0].name,(obj[1][0],obj[1][1]),obj[2])],
                    }
                else:
                    raise Exception('Type is not supported!')
            else:
                self.json['Relations'][self.name] = {
                    linear_bias_relation_name:[(obj[0].name,obj[1])],
                }
        elif (type(obj) is neu4mes.Input or
            type(obj) is neu4mes.Relation or
            issubclass(type(obj), neu4mes.Relation)):
            super().__init__(obj.json)
            self.name = obj.name+'_lin_bias'+str(neu4mes.NeuObj.count)
            self.json['Relations'][self.name] = {
                linear_bias_relation_name:[obj.name]
            }
        else:
            raise Exception('Type is not supported!')

def createLinear(self, name, input):
    return tensorflow.keras.layers.Dense(units = 1, activation = None, use_bias = None, name = name)(input)

def createLinearBias(self, name, input):
    return tensorflow.keras.layers.Dense(units = 1, activation = None, use_bias = None, name = name)(input)

setattr(neu4mes.Neu4mes, linear_relation_name, createLinear)
setattr(neu4mes.Neu4mes, linear_bias_relation_name, createLinearBias)
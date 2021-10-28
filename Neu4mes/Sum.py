from Neu4mes import Relation, NeuObj
import tensorflow.keras.layers

class Sum(Relation):
    def __init__(self, obj1 = None, obj2 = None):
        if obj1 is None:
            return
        super().__init__(obj1.json)
        self.json = NeuObj.merge(obj1.json,obj2.json)
        self.name = obj1.name+'_sum'               
        self.json['Relations'][self.name] = {
            'Sum':[obj1.name,obj2.name]
        }

    def sum(self, obj):
        if type(obj) is Sum:
            self.json = NeuObj.merge(self.json,obj.json)
            self.json['Relations'][self.name]['Sum'] = self.json['Relations'][self.name]['Sum'] + obj.json['Relations'][obj.name]['Sum']
            del obj.json['Relations'][obj.name]
        else:
            self.json = NeuObj.merge(self.json,obj.json)
            self.json['Relations'][self.name]['Sum'].append(obj.name)
        return self

    
    def createElem(self, name, input):
        return tensorflow.keras.layers.Add(name = name)(input)
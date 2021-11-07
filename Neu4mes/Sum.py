import Neu4mes
import tensorflow.keras.layers


sum_relation_name = 'Sum'

class Sum(Neu4mes.Relation):
    def __init__(self, obj1, obj2):
        super().__init__(obj1.json)
        self.json = Neu4mes.merge(obj1.json,obj2.json)
        self.name = obj1.name+'_sum'+str(Neu4mes.NeuObj.count)
        self.json['Relations'][self.name] = {sum_relation_name:[]}
        if type(obj1) is Sum:
            for el in self.json['Relations'][obj1.name]['Sum']:
                self.json['Relations'][self.name][sum_relation_name].append(el)
            self.json['Relations'][self.name][sum_relation_name].append(obj2.name)
        elif type(obj2) is Sum:               
            for el in self.json['Relations'][obj2.name]['Sum']:
                self.json['Relations'][self.name][sum_relation_name].append(el)
            self.json['Relations'][self.name][sum_relation_name].append(obj1.name)
        else:
            self.json['Relations'][self.name][sum_relation_name].append(obj1.name)
            self.json['Relations'][self.name][sum_relation_name].append(obj2.name)

    # def sum(self, obj):
    #     if type(obj) is Sum:
    #         self.json = Neu4mes.NeuObj.merge(self.json,obj.json)
    #         self.json['Relations'][self.name][sum_relation_name] = self.json['Relations'][self.name][sum_relation_name] + obj.json['Relations'][obj.name][sum_relation_name]
    #         del obj.json['Relations'][obj.name]
    #     else:
    #         self.json = Neu4mes.NeuObj.merge(self.json,obj.json)
    #         self.json['Relations'][self.name][sum_relation_name].append(obj.name)
    #     return self



def createSum(self, name, input):
    return tensorflow.keras.layers.Add(name = name)(input)

setattr(Neu4mes.Neu4mes, sum_relation_name, createSum)
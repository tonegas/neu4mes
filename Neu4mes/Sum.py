import neu4mes
import tensorflow.keras.layers


sum_relation_name = 'Sum'

class Sum(neu4mes.Relation):
    def __init__(self, obj1, obj2):
        super().__init__(obj1.json)
        self.json = neu4mes.merge(obj1.json,obj2.json)
        self.name = obj1.name+'_sum'+str(neu4mes.NeuObj.count)
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


def createSum(self, name, input):
    return tensorflow.keras.layers.Add(name = name)(input)

setattr(neu4mes.Neu4mes, sum_relation_name, createSum)
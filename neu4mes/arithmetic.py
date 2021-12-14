import neu4mes
import tensorflow.keras.layers
import tensorflow.keras.backend as K



sum_relation_name = 'Sum'
minus_relation_name = 'Minus'
subtract_relation_name = 'Subtract'
square_relation_name = 'Square'

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

class Subtract(neu4mes.Relation):
    def __init__(self, obj1, obj2):
        super().__init__(obj1.json)
        self.json = neu4mes.merge(obj1.json,obj2.json)
        self.name = obj1.name+'_sub'+str(neu4mes.NeuObj.count)
        self.json['Relations'][self.name] = {sum_relation_name:[]}
        if type(obj1) is Subtract:
            for el in self.json['Relations'][obj1.name]['Sub']:
                self.json['Relations'][self.name][sum_relation_name].append(el)
            self.json['Relations'][self.name][sum_relation_name].append(obj2.name)
        elif type(obj2) is Subtract:               
            for el in self.json['Relations'][obj2.name]['Sub']:
                self.json['Relations'][self.name][sum_relation_name].append(el)
            self.json['Relations'][self.name][sum_relation_name].append(obj1.name)
        else:
            self.json['Relations'][self.name][sum_relation_name].append(obj1.name)
            self.json['Relations'][self.name][sum_relation_name].append(obj2.name)

class Minus(neu4mes.Relation):
    def __init__(self, obj = None):
        if obj is None:
            return
        super().__init__(obj.json)
        obj_name = obj.name
        self.name = obj.name+'_minus'
        self.json['Relations'][self.name] = {
            minus_relation_name:[obj_name]
        }

class Square(neu4mes.Relation):
    def __init__(self, obj):
        if obj is None:
            return
        super().__init__(obj.json)
        obj_name = obj.name
        self.name = obj.name+'_square'
        self.json['Relations'][self.name] = {
            square_relation_name:[obj_name]
        }

def createMinus(self, name, input):
    return -input

def createSum(self, name, input):
    return tensorflow.keras.layers.Add(name = name)(input)

def createSubtract(self, name, input):
    return tensorflow.keras.layers.Subtract(name = name)(input)

def createSquare(self, name, input):
    return K.pow(input,2)

setattr(neu4mes.Neu4mes, minus_relation_name, createMinus)
setattr(neu4mes.Neu4mes, sum_relation_name, createSum)
setattr(neu4mes.Neu4mes, subtract_relation_name, createSubtract)
setattr(neu4mes.Neu4mes, square_relation_name, createSquare)
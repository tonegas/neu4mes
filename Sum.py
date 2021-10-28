import Relation
from NeuObj import merge

class Sum(Relation.Relation):
    def __init__(self, obj1, obj2):
        super().__init__(obj1.json)
        self.json = merge(obj1.json,obj2.json)
        self.name = obj1.name+'_sum'               
        self.json['Relations'][self.name] = {
            'Sum':[obj1.name,obj2.name]
        }
        #self.elem_to_sum = obj

    def sum(self, obj):
        if type(obj) is Sum:
            self.json = merge(self.json,obj.json)
            self.json['Relations'][self.name]['Sum'] = self.json['Relations'][self.name]['Sum'] + obj.json['Relations'][obj.name]['Sum']
            del obj.json['Relations'][obj.name]
        else:
            self.json = merge(self.json,obj.json)
            self.json['Relations'][self.name]['Sum'].append(obj.name)
        return self


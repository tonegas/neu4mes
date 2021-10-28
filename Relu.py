import Relation
import Input

class Relu(Relation.Relation):
    def __init__(self, obj):
        self.name = ''
        if type(obj) is tuple:
            super().__init__(obj[0].json)
            self.name = obj[0].name+'_relu'
            self.json['Relations'][self.name] = {
                'Relu':[(obj[0].name,obj[1])],
            }
        elif type(obj) is Input.Input:
            super().__init__(obj.json)
            self.name = obj.name+'_relu'
            self.json['Relations'][self.name] = {
                'Relu':[obj.name]
            }
        elif issubclass(type(obj),Relation.Relation):
            super().__init__(obj.json)
            self.name = obj.name+'_relu'
            self.json['Relations'][self.name] = {
                'Relu':[obj.name]
            }
        else:
            raise Exception('Type is not supported!')



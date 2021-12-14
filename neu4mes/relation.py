import types, copy
import pprint

def merge(source, destination, main = True):
    if main:
        result = copy.deepcopy(destination)
    else:
        result = destination
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = result.setdefault(key, {})
            merge(value, node, False)
        else:
            result[key] = value

    return result

class NeuObj():
    count = 0
    def __init__(self,json = {}):
        NeuObj.count = NeuObj.count + 1
        if json:
            self.json = copy.deepcopy(json)
        else:
            self.json = {
                'SampleTime': 0,
                'Inputs' : {},
                'Outputs': {},
                'Relations': {}
            }

    def __xor__(self, val):
        from neu4mes import Square
        if val == 2:
            return Square(self)  
        else:
            raise Exception("Operation not supported yet")

class Relation(NeuObj):      
    def __init__(self,json):
        super().__init__(json)

    def __neg__(self):
        from neu4mes import Minus
        return Minus(self)
    
    def __add__(self, obj): 
        from neu4mes import Sum
        return Sum(self, obj)

    def __minus__(self, obj): 
        from neu4mes import Subtract
        return Subtract(self, obj)      
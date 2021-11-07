from Neu4mes import * 
import copy

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
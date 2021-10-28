from Neu4mes import * 
import copy

def merge(source, destination):
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            merge(value, node)
        else:
            destination[key] = value

    return destination

class NeuObj():
    def __init__(self,json = {}):
        if json:
            self.json = copy.deepcopy(json)
        else:
            self.json = {
                'SampleTime': 0,
                'Inputs' : {},
                'States' : {},
                'Outputs': {},
                'Relations': {}
            }
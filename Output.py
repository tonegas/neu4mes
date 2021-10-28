import NeuObj
import Input

class Output(NeuObj.NeuObj):              
    def __init__(self, obj, relation):
        super().__init__(relation.json)
        if type(obj) is tuple:
            self.name = obj[0].name+obj[1]
        elif type(obj) is Input.Input:
            self.name = obj.name
        self.json['Relations'][self.name] = self.json['Relations'][relation.name]
        del self.json['Relations'][relation.name]
        self.json['Outputs'][self.name] = {}
        for key, val in self.json['Relations'][self.name].items():
            for signal in val:
                self.navigateRelations(signal)
    
    def navigateRelations(self,signal):
        if signal in self.json['Relations']:
            self.json['Relations'][(self.name,signal)] = self.json['Relations'][signal]
            del self.json['Relations'][signal]        
            for key, val in self.json['Relations'][(self.name,signal)].items():
                for signal in val:
                    if type(signal) is tuple:
                        self.navigateRelations(signal[0])
                    else:
                        self.navigateRelations(signal)

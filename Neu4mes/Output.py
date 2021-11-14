from Neu4mes import NeuObj
import Neu4mes

class Output(NeuObj.NeuObj):              
    def __init__(self, obj, relation):
        super().__init__(relation.json)
        if type(obj) is tuple:
            self.name = obj[0].name+obj[1]
            self.signal_name = obj[0].name
        elif type(obj) is Neu4mes.Input:
            self.name = obj.name
            self.signal_name = obj[0].name
        self.json['Outputs'][self.name] = {}
        # print(self.json)
        if relation.name in self.json['Relations']:
            self.json['Relations'][self.name] = self.json['Relations'][relation.name]
            relations = self.json['Relations']
        # else:
        #     self.json['AbstractRelations'][self.name] = self.json['AbstractRelations'][relation.name]
        #     relations = self.json['AbstractRelations']
            
        for key, val in relations[self.name].items():
            for signal in val:
                self.navigateRelations(signal)
    
    def navigateRelations(self,signal):
        # if signal in self.json['AbstractRelations']:
        #     self.json['AbstractRelations'][(self.name,signal)] = self.json['AbstractRelations'][signal]
        #     del self.json['AbstractRelations'][signal]        
        #     for key, val in self.json['AbstractRelations'][(self.name,signal)].items():
        #         for signal in val:
        #             if type(signal) is tuple:
        #                 self.navigateRelations(signal[0])
        #             else:
        #                 self.navigateRelations(signal)

        if signal in self.json['Relations']:   
            for key, val in self.json['Relations'][signal].items():
                for signal in val:
                    if type(signal) is tuple:
                        self.navigateRelations(signal[0])
                    else:
                        self.navigateRelations(signal)

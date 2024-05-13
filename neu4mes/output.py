from neu4mes.relation import Stream
from neu4mes.input import Input

class Output(Stream):
    def __init__(self, obj, relation):
        super().__init__(relation.name, relation.json, relation.dim)
        self.json['Outputs'][obj.name] = {}
        self.json['Relations'][obj.name] = self.json['Relations'][relation.name]
        self.json['Relations'].__delitem__(relation.name)
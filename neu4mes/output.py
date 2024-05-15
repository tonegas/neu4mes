from neu4mes.relation import Stream
from neu4mes.input import Input

class Output(Stream):
    def __init__(self, name, relation):
        super().__init__(relation.name, relation.json, relation.dim)
        self.json['Outputs'][name] = {}
        self.json['Outputs'][name] = relation.name
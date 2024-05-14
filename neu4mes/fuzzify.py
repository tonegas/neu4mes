import inspect, copy
import numpy as np

import torch.nn as nn

from neu4mes.relation import NeuObj, Stream, merge
from neu4mes.input import Input
from neu4mes.model import Model

fuzzify_relation_name = 'Fuzzify'

class Fuzzify(NeuObj):
    def __init__(self, output_dimension, range = None, centers = None, functions = 'Triangular'):
        self.relation_name = fuzzify_relation_name
        super().__init__('F' + fuzzify_relation_name + str(NeuObj.count))
        self.output_dimension = {'dim' : output_dimension}
        self.json['Functions'][self.name] = {}
        self.json['Functions'][self.name]['dim_out'] = copy.deepcopy(self.output_dimension)
        if range is not None:
            assert centers is None, 'if output is an integer or use centers or use range'
            interval = ((range[1]-range[0])/(output_dimension-1))
            self.json['Functions'][self.name]['centers'] = [a for a in np.arange(range[0], range[1]+interval, interval)]
        elif centers is not None:
            assert range is None, 'if output is an integer or use centers or use range'
            assert len(centers) == output_dimension, 'number of centers must be equal to output_dimension'
            self.json['Functions'][self.name]['centers'] = centers

        if type(functions) is str:
            self.json['Functions'][self.name]['functions'] = functions
        elif type(functions) is list:
            assert len(functions) == self.output_dimension['dim'], 'number of functions must be equal to output_dimension'
            self.json['Functions'][self.name]['functions'] = []
            for func in functions:
                self.json['Functions'][self.name]['functions'].append(inspect.getsource(func))
        else:
            self.json['Functions'][self.name]['functions'] = inspect.getsource(functions)

    def __call__(self, obj):
        stream_name = fuzzify_relation_name + str(Stream.count)
        assert 'dim' in obj.dim and obj.dim['dim'] == 1, 'Input dimension must be scalar'
        output_dimension = copy.deepcopy(obj.dim)
        output_dimension.update(self.output_dimension)

        #if 'dim' in obj.dim:
        #    if obj.dim['dim_in'] == 1:
        #        self.json['Functions'][self.name]['dim_out'] = self.output_dimension
        #    else:
        #        self.json['Functions'][self.name]['dim_out'] = [self.output_dimension, obj.dim['dim_in']]
        #else:
        #    self.json['Functions'][self.name]['dim_out'] = [self.output_dimension, obj.dim]
        stream_json = merge(self.json, obj.json)
        if type(obj) is Input or type(obj) is Stream:
            stream_json['Relations'][stream_name] = [fuzzify_relation_name, [obj.name],self.name]
            return Stream(stream_name, stream_json,output_dimension)
        else:
            raise Exception('Type is not supported!')

def createFizzify(self):
    pass

setattr(Model, fuzzify_relation_name, createFizzify)

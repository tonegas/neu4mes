import inspect, copy

import torch.nn as nn

from neu4mes.relation import NeuObj
from neu4mes.model import Model
from neu4mes.part import Select
from neu4mes.utilis import check

localmodel_relation_name = 'LocalModel'

class LocalModel(NeuObj):
    def __init__(self, input_function = None, output_function = None, pass_indexes = False):
        self.relation_name = localmodel_relation_name
        self.pass_indexes = pass_indexes
        super().__init__(localmodel_relation_name + str(NeuObj.count))
        self.json['Functions'][self.name] = {}
        if input_function is not None:
            check(callable(input_function), TypeError, 'The input_function must be callable')
        self.input_function = input_function
        if output_function is not None:
            check(callable(output_function), TypeError, 'The output_function must be callable')
        self.output_function = output_function


    def __call__(self, inputs, activations):
        self.out_sum = []
        if type(activations) is not tuple:
            activations = (activations,)
        self.___activations_matrix(activations,inputs)

        out = self.out_sum[0]
        for ind in range(1,len(self.out_sum)):
            out = out + self.out_sum[ind]
        return out

    # Definisci una funzione ricorsiva per annidare i cicli for
    def ___activations_matrix(self, activations, inputs, idx=0, idx_list=[]):
        if idx != len(activations):
            for i in range(activations[idx].dim['dim']):
                self.___activations_matrix(activations, inputs, idx+1, idx_list+[i])
        else:
            if self.input_function is not None:
                if len(inspect.getfullargspec(self.input_function).args) == 0:
                    if type(inputs) is tuple:
                        out_in = self.input_function()(*inputs)
                    else:
                        out_in = self.input_function()(inputs)
                else:
                    if self.pass_indexes:
                        if type(inputs) is tuple:
                            out_in = self.input_function(idx_list)(*inputs)
                        else:
                            out_in = self.input_function(idx_list)(inputs)
                    else:
                        if type(inputs) is tuple:
                            out_in = self.input_function(*inputs)
                        else:
                            out_in = self.input_function(inputs)
            else:
                check(type(inputs) is not tuple, TypeError, 'The input cannot be a tuple without input_function')
                out_in = inputs

            act = Select(activations[0], idx_list[0])
            for ind, i  in enumerate(idx_list[1:]):
                act = act * Select(activations[ind+1], i)

            prod = out_in * act

            if self.output_function is not None:
                if len(inspect.getfullargspec(self.output_function).args) == 0:
                    self.out_sum.append(self.output_function()(prod))
                else:
                    if self.pass_indexes:
                        self.out_sum.append(self.output_function(idx_list)(prod))
                    else:
                        self.out_sum.append(self.output_function(prod))
            else:
                self.out_sum.append(prod)

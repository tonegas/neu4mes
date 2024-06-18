import inspect, copy
import numpy as np

import torch.nn as nn
import torch

from neu4mes.relation import NeuObj, Stream, merge
from neu4mes.model import Model

from neu4mes import LOG_LEVEL
from neu4mes.logger import logging
log = logging.getLogger(__name__)
log.setLevel(max(logging.ERROR, LOG_LEVEL))

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
            self.json['Functions'][self.name]['names'] = functions
        elif type(functions) is list:
            #assert len(functions) % self.output_dimension['dim'], 'number of functions must be equal to output_dimension'
            self.json['Functions'][self.name]['functions'] = []
            self.json['Functions'][self.name]['names'] = []
            for func in functions:
                self.json['Functions'][self.name]['functions'].append(inspect.getsource(func))
                self.json['Functions'][self.name]['names'].append(func.__name__)
        else:
            self.json['Functions'][self.name]['functions'] = inspect.getsource(functions)
            self.json['Functions'][self.name]['names'] = functions.__name__

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
        if type(obj) is Stream:
            stream_json['Relations'][stream_name] = [fuzzify_relation_name, [obj.name],self.name]
            return Stream(stream_name, stream_json,output_dimension)
        else:
            raise Exception('Type is not supported!')

def triangular(x, idx_channel, chan_centers):

    # Compute the number of channels
    num_channels = len(chan_centers)

    # First dimension of activation
    if idx_channel == 0:
        if num_channels != 1:
            ampl    = chan_centers[1] - chan_centers[0]
            act_fcn = torch.minimum(torch.maximum(-(x - chan_centers[0])/ampl + 1, torch.tensor(0.0)), torch.tensor(1.0))
        else:
            # In case the user only wants one channel
            act_fcn = 1
    elif idx_channel != 0 and idx_channel == (num_channels - 1):
        ampl    = chan_centers[-1] - chan_centers[-2]
        act_fcn = torch.minimum(torch.maximum((x - chan_centers[-2])/ampl, torch.tensor(0.0)), torch.tensor(1.0))
    else:
        ampl_1  = chan_centers[idx_channel] - chan_centers[idx_channel - 1]
        ampl_2  = chan_centers[idx_channel + 1] - chan_centers[idx_channel]
        act_fcn = torch.minimum(torch.maximum((x - chan_centers[idx_channel - 1])/ampl_1, torch.tensor(0.0)),torch.maximum(-(x - chan_centers[idx_channel])/ampl_2 + 1, torch.tensor(0.0)))
  
    return act_fcn

def rectangular(x, idx_channel, chan_centers):
    ## compute number of channels
    num_channels = len(chan_centers)

    ## First dimension of activation
    if idx_channel == 0:
        if num_channels != 1:
            width = abs(chan_centers[idx_channel+1] - chan_centers[idx_channel]) / 2
            act_fcn = torch.where(x < (chan_centers[idx_channel] + width), torch.tensor(1.0), torch.tensor(0.0))
        else:
            # In case the user only wants one channel
            act_fcn = torch.tensor(1.0)
    elif idx_channel != 0 and idx_channel == (num_channels - 1):
        width = abs(chan_centers[idx_channel] - chan_centers[idx_channel-1]) / 2
        act_fcn = torch.where(x >= chan_centers[idx_channel] - width, torch.tensor(1.0), torch.tensor(0.0))
    else:
        width_forward = abs(chan_centers[idx_channel+1] - chan_centers[idx_channel]) / 2  
        width_backward = abs(chan_centers[idx_channel] - chan_centers[idx_channel-1]) / 2
        act_fcn = torch.where((x >= chan_centers[idx_channel] - width_backward) & (x < chan_centers[idx_channel] + width_forward), torch.tensor(1.0), torch.tensor(0.0))
  
    return act_fcn

def custom_function(func, x, idx_channel, chan_centers):
    ## compute number of channels
    num_channels = len(chan_centers)

    ## First dimension of activation
    if idx_channel == 0:
        if num_channels != 1:
            width = abs(chan_centers[idx_channel+1] - chan_centers[idx_channel]) / 2
            act_fcn = torch.where(x < (chan_centers[idx_channel] + width), torch.where(x >= (chan_centers[idx_channel] - width), func(x-chan_centers[idx_channel]), torch.tensor(1.0)), torch.tensor(0.0))
        else:
            # In case the user only wants one channel
            act_fcn = torch.tensor(1.0)
    elif idx_channel != 0 and idx_channel == (num_channels - 1):
        width = abs(chan_centers[idx_channel] - chan_centers[idx_channel-1]) / 2
        act_fcn = torch.where(x >= chan_centers[idx_channel] - width, torch.where(x < (chan_centers[idx_channel] + width), func(x-chan_centers[idx_channel]), torch.tensor(1.0)), torch.tensor(0.0))
    else:
        width_forward = abs(chan_centers[idx_channel+1] - chan_centers[idx_channel]) / 2  
        width_backward = abs(chan_centers[idx_channel] - chan_centers[idx_channel-1]) / 2
        act_fcn = torch.where((x >= chan_centers[idx_channel] - width_backward) & (x < chan_centers[idx_channel] + width_forward), func(x-chan_centers[idx_channel]), torch.tensor(0.0))
  
    return act_fcn

class Fuzzify_Layer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.centers = params['centers']
        self.function = params['functions']
        self.dimension = params['dim_out']['dim']
        self.name = params['names']

        if type(self.name) is list:
            self.n_func = len(self.name)
            for func in self.function:
                ## Add the function to the globals
                try:
                    exec(func, globals())
                except Exception as e:
                    print(f"An error occurred: {e}")
        else:
            self.n_func = 1
            if self.name not in ['Triangular', 'Rectangular']: ## custom function
                ## Add the function to the globals
                try:
                    exec(self.function, globals())
                except Exception as e:
                    print(f"An error occurred: {e}")

    def forward(self, x):
        res = torch.empty((x.size(0), self.dimension), dtype=torch.float32)

        if self.function == 'Triangular':
            for i in range(len(self.centers)):
                res[:, i] = triangular(x, i, self.centers)
        elif self.function == 'Rectangular':
            for i in range(len(self.centers)):
                res[:, i] = rectangular(x, i, self.centers)
        else: ## Custom_function
            if self.n_func == 1:
                # Retrieve the function object from the globals dictionary
                function_to_call = globals()[self.name]
                for i in range(len(self.centers)):
                    res[:, i] = custom_function(function_to_call, x, i, self.centers)
            else: ## we have multiple functions
                for i in range(len(self.centers)):
                    if i >= self.n_func:
                        func_idx = i - self.n_func
                    else:
                        func_idx = i
                    function_to_call = globals()[self.name[func_idx]]
                    res[:, i] = custom_function(function_to_call, x, i, self.centers)
        return res

def createFuzzify(self, params):
    return Fuzzify_Layer(params)

setattr(Model, fuzzify_relation_name, createFuzzify)

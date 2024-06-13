import inspect, copy

import torch.nn as nn

from neu4mes.relation import NeuObj, Stream, AutoToStream, merge
from neu4mes.input import Input
from neu4mes.model import Model
from neu4mes.parameter import Parameter
from neu4mes.utilis import check

paramfun_relation_name = 'ParamFun'

class ParamFun(NeuObj):
    def __init__(self, param_fun, output_dimension = 1, parameters_dimensions = None, parameters = None):
        self.relation_name = paramfun_relation_name
        self.param_fun = param_fun
        self.output_dimension = {'dim' : output_dimension }
        super().__init__('F'+paramfun_relation_name + str(NeuObj.count))
        self.json['Functions'][self.name] = {
            'code' : inspect.getsource(param_fun),
            'name' : param_fun.__name__
        }
        self.json['Functions'][self.name]['out_dim'] = copy.deepcopy(self.output_dimension)

        self.json['Functions'][self.name]['parameters'] = []
        self.__set_params(parameters_dimensions = parameters_dimensions, parameters = parameters)

    def __call__(self, *obj):
        stream_name = paramfun_relation_name + str(Stream.count)
        output_dimension = copy.deepcopy(self.output_dimension)

        self.json['Functions'][self.name]['n_input'] = len(obj)
        self.__set_params(n_input = len(obj))
        names = []
        stream_json = copy.deepcopy(self.json)

        window_val = None
        for ind,o in enumerate(obj):
            window = 'tw' if 'tw' in o.dim else ('sw' if 'sw' in o.dim else None)
            if window_val is not None:
                check(o.dim[window] == window_val,ValueError, 'The time window of the input must be the same for all the inputs')
            if window:
                window_val = o.dim[window]

            stream_json = merge(stream_json,o.json)
            if type(o) is Input or type(o) is Stream:
                names.append(o.name)
            else:
                raise Exception('Type is not supported!')

        if window is not None:
            output_dimension[window] = window_val

        # self.json['Functions'][self.name]['out_dim'].update(self.output_dimension)
        stream_json['Relations'][stream_name] = [paramfun_relation_name, names, self.name]
        #self.__infer_output_dimensions(stream_json, stream_name)

        return Stream(stream_name, stream_json, output_dimension)

    def __infer_output_dimensions(self, stream_json, stream_name):
        import torch
        inputs = []
        for input in stream_json['Relations'][stream_name][1]:
            if type(input) is tuple:
                window = 'tw' if 'tw' in input[1] else ('sw' if 'sw' in input[1] else None)
                window_val = abs(input[1][window][0]-input[1][window][1])
                inputs.append(torch.rand(size=(window_val,)))
            else:
                inputs.append(torch.rand(size=(1,)))

        for name in stream_json['Functions'][self.name]['parameters']:
            param_dimensions = stream_json['Parameters'][name]
            if type(param_dimensions['dim']) is list:
                param_size = tuple(param_dimensions['dim'])
            else:
                param_size = (param_dimensions['dim'], 1)
            inputs.append(torch.rand(size=param_size))
        out = self.param_fun(*inputs)
        print(out.shape)

    def __set_params(self, n_input = None, parameters_dimensions = None, parameters = None):
        if parameters is not None:
            assert parameters_dimensions is None, 'parameters_dimensions must be None if parameters is set'
            assert type(parameters) is list, 'parameters must be a list'
            for param in parameters:
                if type(param) is Parameter:
                    self.json['Functions'][self.name]['parameters'].append(param.name)
                    self.json['Parameters'][param.name] = param.dim

        elif parameters_dimensions is not None:
            assert type(parameters_dimensions) is dict, 'parameters_dimensions must be a dict'
            funinfo = inspect.getfullargspec(self.param_fun)
            for i in range(len(parameters_dimensions)):
                param_name = self.name + str(funinfo.args[-1 - i])
                self.json['Functions'][self.name]['parameters'].append(param_name)
                self.json['Parameters'][param_name] = {'dim' :parameters_dimensions[funinfo.args[-1 - i]]}

        elif n_input is not None:
            funinfo = inspect.getfullargspec(self.param_fun)
            n_params = len(funinfo.args) - n_input
            if len(self.json['Functions'][self.name]['parameters']) != 0:
                assert n_params == len(self.json['Functions'][self.name]['parameters']), 'number of input are not correct for the number of parameters'
            else:
                for i in range(n_params):
                    param_name = self.name + str(funinfo.args[-1 - i])
                    self.json['Functions'][self.name]['parameters'].append(param_name)
                    self.json['Parameters'][param_name] = {'dim' : 1}

class Parametric_Layer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.name = params['name']

        ## Add the function to the globals
        try:
            exec(params['code'], globals())
            #print(f'executing {self.name}...')
        except Exception as e:
            print(f"An error occurred: {e}")

    def forward(self, inputs, parameters):
        args = inputs + parameters
        # Retrieve the function object from the globals dictionary
        function_to_call = globals()[self.name]
        # Call the function using the retrieved function object
        result = function_to_call(*args)
        return result

def createParamFun(self, func_params):
    return Parametric_Layer(params=func_params)

setattr(Model, paramfun_relation_name, createParamFun)

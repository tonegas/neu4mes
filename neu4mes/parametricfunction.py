import inspect, copy

import torch.nn as nn

from neu4mes.relation import NeuObj, Stream, merge
from neu4mes.model import Model
from neu4mes.parameter import Parameter
from neu4mes.utilis import check


paramfun_relation_name = 'ParamFun'

class ParamFun(NeuObj):
    def __init__(self, param_fun, parameters_dimensions = None, parameters = None):
        self.relation_name = paramfun_relation_name
        self.param_fun = param_fun
        #self.output_dimension = {'dim' : output_dimension }
        self.output_dimension = {}
        super().__init__('F'+paramfun_relation_name + str(NeuObj.count))
        self.json['Functions'][self.name] = {
            'code' : inspect.getsource(param_fun),
            'name' : param_fun.__name__
        }
        #self.json['Functions'][self.name]['out_dim'] = copy.deepcopy(self.output_dimension)
        self.json['Functions'][self.name]['parameters'] = []
        self.__set_params(parameters_dimensions = parameters_dimensions, parameters = parameters)

    def __call__(self, *obj):
        stream_name = paramfun_relation_name + str(Stream.count)
        self.json['Functions'][self.name]['n_input'] = len(obj)
        self.__set_params(n_input=len(obj))
        input_names = []
        input_dimensions = []
        stream_json = copy.deepcopy(self.json)
        for ind,o in enumerate(obj):
            stream_json = merge(stream_json,o.json)
            if type(o) is Stream:
                input_names.append(o.name)
                input_dimensions.append(o.dim)
            else:
                raise Exception('Type is not supported!')

        stream_json['Relations'][stream_name] = [paramfun_relation_name, input_names, self.name]

        self.__infer_output_dimensions(input_dimensions)



        # output_dimension = copy.deepcopy(self.output_dimension)
        #
        #
        #
        # window_val = None
        # for ind,o in enumerate(obj):
        #     window = 'tw' if 'tw' in o.dim else ('sw' if 'sw' in o.dim else None)
        #     if window:
        #         window_val = o.dim[window]
        #
        #     stream_json = merge(stream_json,o.json)
        #     if type(o) is Input or type(o) is Stream:
        #         names.append(o.name)
        #     else:
        #         raise Exception('Type is not supported!')
        #
        # if window is not None:
        #     output_dimension[window] = window_val
        #
        # # self.json['Functions'][self.name]['out_dim'].update(self.output_dimension)
        # stream_json['Relations'][stream_name] = [paramfun_relation_name, names, self.name]
        # self.__infer_output_dimensions(stream_json, stream_name)

        output_dimension = copy.deepcopy(self.output_dimension)
        self.json['Functions'][self.name]['out_dim'] = copy.deepcopy(self.output_dimension)

        return Stream(stream_name, stream_json, output_dimension)

    def __infer_output_dimensions(self, input_dimensions):
        import torch
        batch_dim = 10
        # Build input with right dimensions
        inputs = []
        inputs_win_type = []
        inputs_win = []
        for dim in input_dimensions:
            window = 'tw' if 'tw' in dim else ('sw' if 'sw' in dim else None)
            if window == 'tw':
                input_win = round(dim[window]*1000)
            elif window == 'sw':
                input_win = dim[window]
            else:
                input_win = 1
            inputs.append(torch.rand(size=(batch_dim, input_win, dim['dim'])))
            inputs_win_type.append(window)
            inputs_win.append(input_win)


        for name in self.json['Functions'][self.name]['parameters']:
            dim = self.json['Parameters'][name]
            window = 'tw' if 'tw' in dim else ('sw' if 'sw' in dim else None)
            if window == 'tw':
                dim_win = round(dim[window] * 1000)
            elif window == 'sw':
                dim_win = dim[window]
            else:
                dim_win = 1
            if type(dim['dim']) is tuple:
                inputs.append(torch.rand(size= dim['dim'] + (dim_win,) ))
            else:
                inputs.append(torch.rand(size=(dim['dim'], dim_win)))

        out = self.param_fun(*inputs)
        out_shape = out.shape
        out_dim = list(out_shape[2:])
        check(len(out_dim) == 1, ValueError, "The output dimension of the function is bigger than a vector.")
        out_win_from_input = False
        for idx, win in enumerate(inputs_win):
            if out_shape[1] == win:
                out_win_from_input = True
                out_win_type = inputs_win_type[idx]
                out_win = input_dimensions[idx][out_win_type]
        if out_win_from_input == False:
            out_win_type = 'sw'
            out_win = out_shape[1]
            print("The window dimension of the output is not referred to any input.")
        self.output_dimension = {'dim': out_dim[0], out_win_type : out_win}

    def __set_params(self, n_input = None, parameters_dimensions = None, parameters = None):
        if parameters is not None:
            assert parameters_dimensions is None, 'parameters_dimensions must be None if parameters is set'
            assert type(parameters) is list, 'parameters must be a list'
            for param in parameters:
                if type(param) is Parameter:
                    self.json['Functions'][self.name]['parameters'].append(param.name)
                    #self.json['Parameters'][param.name] = param.dim
                    self.json['Parameters'][param.name] = copy.deepcopy(param.json['Parameters'][param.name])

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

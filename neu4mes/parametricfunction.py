import inspect, copy, textwrap

import torch.nn as nn

from neu4mes.relation import NeuObj, Stream, toStream
from neu4mes.model import Model
from neu4mes.parameter import Parameter
from neu4mes.utilis import check, merge


paramfun_relation_name = 'ParamFun'

class ParamFun(NeuObj):
    def __init__(self, param_fun, n_input = None, parameters_dimensions = None, parameters = None):
        self.relation_name = paramfun_relation_name
        self.param_fun = param_fun
        self.output_dimension = {}
        super().__init__('F'+paramfun_relation_name + str(NeuObj.count))
        code = textwrap.dedent(inspect.getsource(param_fun)).replace('\"', '\'')
        self.json['Functions'][self.name] = {
            'code' : code,
            'name' : param_fun.__name__
        }
        self.json['Functions'][self.name]['parameters'] = []
        self.__set_params(n_input = n_input, parameters_dimensions = parameters_dimensions, parameters = parameters)

    def __call__(self, *obj):
        stream_name = paramfun_relation_name + str(Stream.count)
        self.json['Functions'][self.name]['n_input'] = len(obj)
        self.__set_params(n_input=len(obj))
        input_names = []
        input_dimensions = []
        stream_json = copy.deepcopy(self.json)
        for ind,o in enumerate(obj):
            o = toStream(o)
            stream_json = merge(stream_json,o.json)
            check(type(o) is Stream, TypeError,
                  f"The type of {o} is {type(o)} and is not supported for ParamFun operation.")
            input_names.append(o.name)
            input_dimensions.append(o.dim)

        stream_json['Relations'][stream_name] = [paramfun_relation_name, input_names, self.name]
        self.__infer_output_dimensions(input_dimensions)

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
                inputs.append(torch.rand(size= (dim_win,) + dim['dim'] ))
            else:
                inputs.append(torch.rand(size=(dim_win, dim['dim'])))

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
            #self.visualizer.warning("The window dimension of the output is not referred to any input.")
        self.output_dimension = {'dim': out_dim[0], out_win_type : out_win}

    def __set_params(self, n_input = None, parameters_dimensions = None, parameters = None):
        funinfo = inspect.getfullargspec(self.param_fun)
        if type(parameters) is list:
            check(parameters_dimensions is None, ValueError,
                  '\"parameters_dimensions\" must be None if \"parameters\" is set using list')
            check(n_input is None, ValueError,
                  '\"n_input\" must be None if \"parameters\" is set using list')
            for param in parameters:
                if type(param) is Parameter:
                    self.json['Functions'][self.name]['parameters'].append(param.name)
                    self.json['Parameters'][param.name] = copy.deepcopy(param.json['Parameters'][param.name])
                elif type(param) is str:
                    self.json['Functions'][self.name]['parameters'].append(param)
                    self.json['Parameters'][param] = {'dim': 1}
                else:
                    check(type(param) is Parameter or type(param) is str, TypeError,
                          'The element inside the \"parameters\" list must be a Parameter or str')
        elif type(parameters_dimensions) is list:
            check(n_input is None, ValueError,
                  '\"n_input\" must be None if \"parameters\" is set using list')
            for i, param_dim in enumerate(parameters_dimensions):
                idx = i + len(funinfo.args) - len(parameters_dimensions)
                param_name = self.name + str(idx)
                self.json['Functions'][self.name]['parameters'].append(param_name)
                self.json['Parameters'][param_name] = {'dim': parameters_dimensions[funinfo.args[idx]]}
        elif type(parameters) is dict or type(parameters_dimensions) is dict:
            check(n_input is not None, TypeError, 'if \"parameter\" or \"parameters_dimensions\" are dict the number of input must be set')
            for i, key in enumerate(funinfo.args):
                if i >= n_input:
                    if type(parameters) is dict and key in parameters:
                        param = parameters[key]
                        if type(parameters[key]) is Parameter:
                            self.json['Functions'][self.name]['parameters'].append(param.name)
                            self.json['Parameters'][param.name] = copy.deepcopy(param.json['Parameters'][param.name])
                        elif type(parameters[key]) is str:
                            self.json['Functions'][self.name]['parameters'].append(param)
                            self.json['Parameters'][param] = {'dim': 1}
                        else:
                            check(type(param) is Parameter or type(param) is str, TypeError,
                                  'The element inside the \"parameters\" dict must be a Parameter or str')
                    elif type(parameters_dimensions) is dict and key in parameters_dimensions:
                        param_name = self.name + key
                        check(isinstance(parameters_dimensions[key],(tuple,int)), TypeError,
                              'The element inside the \"parameters_dimensions\" dict must be a tuple or int')
                        self.json['Functions'][self.name]['parameters'].append(param_name)
                        self.json['Parameters'][param_name] = {'dim': parameters_dimensions[key]}
                    else:
                        param_name = self.name + key
                        self.json['Functions'][self.name]['parameters'].append(param_name)
                        self.json['Parameters'][param_name] = {'dim': 1}
        elif n_input is not None:
            funinfo = inspect.getfullargspec(self.param_fun)
            n_input_params = len(funinfo.args) - n_input
            if len(self.json['Functions'][self.name]['parameters']) != 0:
                n_params = len(self.json['Functions'][self.name]['parameters'])
                check(n_input_params == n_params, ValueError,f'The number of input params are {n_input_params} but the number of parameters are {n_params}')
            else:
                for i in range(n_input_params):
                    param_name = self.name + str(funinfo.args[-1 - i])
                    self.json['Functions'][self.name]['parameters'].append(param_name)
                    self.json['Parameters'][param_name] = {'dim' : 1}

class Parametric_Layer(nn.Module):
    def __init__(self, func, params):
        super().__init__()
        self.name = func['name']
        self.params = params
        ## Add the function to the globals
        try:
            code = 'import torch\n@torch.fx.wrap\n' + func['code']
            exec(code, globals())
        except Exception as e:
            print(f"An error occurred: {e}")

    def forward(self, *inputs):
        args = list(inputs) + self.params
        # Retrieve the function object from the globals dictionary
        function_to_call = globals()[self.name]
        # Call the function using the retrieved function object
        result = function_to_call(*args)
        return result

def createParamFun(self, *func_params):
    return Parametric_Layer(func=func_params[0], params=func_params[1])

setattr(Model, paramfun_relation_name, createParamFun)

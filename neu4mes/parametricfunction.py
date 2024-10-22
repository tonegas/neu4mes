import inspect, copy, textwrap, torch, math

import torch.nn as nn

from collections.abc import Callable

from neu4mes.relation import NeuObj, Stream, toStream
from neu4mes.model import Model
from neu4mes.parameter import Parameter, Constant
from neu4mes.utils import check, merge, enforce_types


paramfun_relation_name = 'ParamFun'

class ParamFun(NeuObj):
    @enforce_types
    def __init__(self, param_fun:Callable,
                 constants:list|dict|None = None,
                 parameters_dimensions:list|dict|None = None,
                 parameters:list|dict|None = None,
                 map_over_batch:bool = False) -> Stream:

        self.relation_name = paramfun_relation_name

        # input parameters
        self.param_fun = param_fun
        self.constants = constants
        self.parameters_dimensions = parameters_dimensions
        self.parameters = parameters
        self.map_over_batch = map_over_batch

        self.output_dimension = {}
        super().__init__('F'+paramfun_relation_name + str(NeuObj.count))
        code = textwrap.dedent(inspect.getsource(param_fun)).replace('\"', '\'')
        self.json['Functions'][self.name] = {
            'code' : code,
            'name' : param_fun.__name__,
        }
        self.json['Functions'][self.name]['params_and_consts'] = []

    def __call__(self, *obj):
        stream_name = paramfun_relation_name + str(Stream.count)

        funinfo = inspect.getfullargspec(self.param_fun)
        n_function_input = len(funinfo.args)
        n_call_input = len(obj)
        n_new_constants_and_params = n_function_input - n_call_input

        if 'n_input' not in self.json['Functions'][self.name]:
            self.json['Functions'][self.name]['n_input'] = n_call_input
            self.__set_params_and_consts(n_new_constants_and_params)

            input_dimensions = []
            input_types = []
            for ind, o in enumerate(obj):
                if type(o) in (int,float,list):
                    obj_type = Constant
                else:
                    obj_type = type(o)
                o = toStream(o)
                check(type(o) is Stream, TypeError,
                      f"The type of {o} is {type(o)} and is not supported for ParamFun operation.")
                input_types.append(obj_type)
                input_dimensions.append(o.dim)

            self.json['Functions'][self.name]['in_dim'] = copy.deepcopy(input_dimensions)
            self.__infer_output_dimensions(input_types, input_dimensions)
            self.json['Functions'][self.name]['out_dim'] = copy.deepcopy(self.output_dimension)

        # Create the missing parameters
        missing_params = n_new_constants_and_params - len(self.json['Functions'][self.name]['params_and_consts'])
        check(missing_params == 0, ValueError, f"The function is called with different number of inputs.")

        stream_json = copy.deepcopy(self.json)
        input_names = []
        for ind, o in enumerate(obj):
            o = toStream(o)
            check(type(o) is Stream, TypeError,
                  f"The type of {o} is {type(o)} and is not supported for ParamFun operation.")
            stream_json = merge(stream_json, o.json)
            input_names.append(o.name)

        output_dimension = copy.deepcopy(self.output_dimension)
        stream_json['Relations'][stream_name] = [paramfun_relation_name, input_names, self.name]

        return Stream(stream_name, stream_json, output_dimension)

    def __set_params_and_consts(self, n_new_constants_and_params):
        funinfo = inspect.getfullargspec(self.param_fun)

        # Create the missing constants from list
        if type(self.constants) is list:
            for const in self.constants:
                if type(const) is Constant:
                    self.json['Functions'][self.name]['params_and_consts'].append(const.name)
                    self.json['Constants'][const.name] = copy.deepcopy(const.json['Constants'][const.name])
                elif type(const) is str:
                    self.json['Functions'][self.name]['params_and_consts'].append(const)
                    self.json['Constants'][const] = {'dim': 1}
                else:
                    check(type(const) is Constant or type(const) is str, TypeError,
                          'The element inside the \"constants\" list must be a Constant or str')

        # Create the missing parameters from list
        if type(self.parameters) is list:
            check(self.parameters_dimensions is None, ValueError,
                  '\"parameters_dimensions\" must be None if \"parameters\" is set using list')
            for param in self.parameters:
                if type(param) is Parameter:
                    self.json['Functions'][self.name]['params_and_consts'].append(param.name)
                    self.json['Parameters'][param.name] = copy.deepcopy(param.json['Parameters'][param.name])
                elif type(param) is str:
                    self.json['Functions'][self.name]['params_and_consts'].append(param)
                    self.json['Parameters'][param] = {'dim': 1}
                else:
                    check(type(param) is Parameter or type(param) is str, TypeError,
                          'The element inside the \"parameters\" list must be a Parameter or str')
        elif type(self.parameters_dimensions) is list:
            for i, param_dim in enumerate(self.parameters_dimensions):
                idx = i + len(funinfo.args) - len(self.parameters_dimensions)
                param_name = self.name + str(idx)
                self.json['Functions'][self.name]['params_and_consts'].append(param_name)
                self.json['Parameters'][param_name] = {'dim': list(self.parameters_dimensions[i])}

        # Create the missing parameters and constants from dict
        missing_params = n_new_constants_and_params - len(self.json['Functions'][self.name]['params_and_consts'])
        if missing_params or type(self.constants) is dict or type(self.parameters) is dict or type(self.parameters_dimensions) is dict:
            n_input = len(funinfo.args) - missing_params
            n_elem_dict = (len(self.constants if type(self.constants) is dict else [])
                           + len(self.parameters if type(self.parameters) is dict else [])
                           + len(self.parameters_dimensions if type(self.parameters_dimensions) is dict else []))
            for i, key in enumerate(funinfo.args):
                if i >= n_input:
                    if type(self.parameters) is dict and key in self.parameters:
                        if self.parameters_dimensions:
                            check(key in self.parameters_dimensions, TypeError,
                                  f'The parameter {key} must be removed from \"parameters_dimensions\".')
                        param = self.parameters[key]
                        if type(self.parameters[key]) is Parameter:
                            self.json['Functions'][self.name]['params_and_consts'].append(param.name)
                            self.json['Parameters'][param.name] = copy.deepcopy(param.json['Parameters'][param.name])
                        elif type(self.parameters[key]) is str:
                            self.json['Functions'][self.name]['params_and_consts'].append(param)
                            self.json['Parameters'][param] = {'dim': 1}
                        else:
                            check(type(param) is Parameter or type(param) is str, TypeError,
                                  'The element inside the \"parameters\" dict must be a Parameter or str')
                        n_elem_dict -= 1
                    elif type(self.parameters_dimensions) is dict and key in self.parameters_dimensions:
                        param_name = self.name + key
                        dim = self.parameters_dimensions[key]
                        check(isinstance(dim,(list,tuple,int)), TypeError,
                              'The element inside the \"parameters_dimensions\" dict must be a tuple or int')
                        self.json['Functions'][self.name]['params_and_consts'].append(param_name)
                        self.json['Parameters'][param_name] = {'dim': list(dim) if type(dim) is tuple else dim}
                        n_elem_dict -= 1
                    elif type(self.constants) is dict and key in self.constants:
                        const = self.constants[key]
                        if type(self.constants[key]) is Constant:
                            self.json['Functions'][self.name]['params_and_consts'].append(const.name)
                            self.json['Constants'][const.name] = copy.deepcopy(const.json['Constants'][const.name])
                        elif type(self.constants[key]) is str:
                            self.json['Functions'][self.name]['params_and_consts'].append(const)
                            self.json['Constants'][const] = {'dim': 1}
                        else:
                            check(type(const) is Constant or type(const) is str, TypeError,
                                  'The element inside the \"constants\" dict must be a Constant or str')
                        n_elem_dict -= 1
                    else:
                        param_name = self.name + key
                        self.json['Functions'][self.name]['params_and_consts'].append(param_name)
                        self.json['Parameters'][param_name] = {'dim': 1}
            check(n_elem_dict == 0, ValueError, 'Some of the input parameters are not used in the function.')

    def __infer_output_dimensions(self, input_types, input_dimensions):
        import torch
        batch_dim = 5

        all_inputs_dim = input_dimensions
        all_inputs_type = input_types
        params_and_consts = self.json['Constants'] | self.json['Parameters']
        for name in self.json['Functions'][self.name]['params_and_consts']:
            all_inputs_dim.append(params_and_consts[name])
            all_inputs_type.append(Constant)

        n_samples_sec = 0.1
        is_int = False
        while is_int == False:
            n_samples_sec *= 10
            vect_input_time = [math.isclose(d['tw']*n_samples_sec,round(d['tw']*n_samples_sec)) for d in all_inputs_dim if 'tw' in d]
            if len(vect_input_time) == 0:
                is_int = True
            else:
                is_int = sum(vect_input_time) == len(vect_input_time)

        # Build input with right dimensions
        inputs = []
        inputs_win_type = []
        inputs_win = []
        input_map_dim = ()

        for t, dim in zip(all_inputs_type,all_inputs_dim):
            window = 'tw' if 'tw' in dim else ('sw' if 'sw' in dim else None)
            if window == 'tw':
                dim_win = round(dim[window] * n_samples_sec)
            elif window == 'sw':
                dim_win = dim[window]
            else:
                dim_win = 1
            if t in (Parameter, Constant):
                if self.map_over_batch:
                    input_map_dim += (None,)
                if type(dim['dim']) is list:
                    inputs.append(torch.rand(size=(dim_win,) + tuple(dim['dim'])))
                else:
                    inputs.append(torch.rand(size=(dim_win, dim['dim'])))
            else:
                inputs.append(torch.rand(size=(batch_dim, dim_win, dim['dim'])))
                if self.map_over_batch:
                    input_map_dim += (0,)

            inputs_win_type.append(window)
            inputs_win.append(dim_win)

        if self.map_over_batch:
            self.json['Functions'][self.name]['map_over_dim'] = list(input_map_dim)
            function_to_call = torch.func.vmap(self.param_fun,in_dims=input_map_dim)
        else:
            self.json['Functions'][self.name]['map_over_dim'] = False
            function_to_call = self.param_fun
        out = function_to_call(*inputs)
        out_shape = out.shape
        check(out_shape[0] == batch_dim, ValueError, "The batch output dimension it is not correct.")
        out_dim = list(out_shape[2:])
        check(len(out_dim) == 1, ValueError, "The output dimension of the function is bigger than a vector.")
        out_win_from_input = False
        for idx, win in enumerate(inputs_win):
            if out_shape[1] == win and all_inputs_type[idx] not in (Parameter, Constant):
                out_win_from_input = True
                out_win_type = inputs_win_type[idx]
                out_win = all_inputs_dim[idx][out_win_type]
        if out_win_from_input == False:
            out_win_type = 'sw'
            out_win = out_shape[1]
            #self.visualizer.warning("The window dimension of the output is not referred to any input.")
        self.output_dimension = {'dim': out_dim[0], out_win_type : out_win}


class Parametric_Layer(nn.Module):
    def __init__(self, func, params_and_consts, map_over_batch):
        super().__init__()
        self.name = func['name']
        self.params_and_consts = params_and_consts
        if type(map_over_batch) is list:
            self.map_over_batch = True
            self.input_map_dim = tuple(map_over_batch)
        else:
            self.map_over_batch = False
        ## Add the function to the globals
        try:
            code = 'import torch\n@torch.fx.wrap\n' + func['code']
            exec(code, globals())
        except Exception as e:
            print(f"An error occurred: {e}")

    def forward(self, *inputs):
        args = list(inputs) + self.params_and_consts
        # Retrieve the function object from the globals dictionary
        function_to_call = globals()[self.name]
        # Call the function using the retrieved function object
        if self.map_over_batch:
            function_to_call = torch.func.vmap(function_to_call,in_dims=self.input_map_dim)
        result = function_to_call(*args)
        return result

def createParamFun(self, *func_params):
    return Parametric_Layer(func=func_params[0], params_and_consts=func_params[1], map_over_batch=func_params[2])

setattr(Model, paramfun_relation_name, createParamFun)

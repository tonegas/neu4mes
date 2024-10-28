import copy

import numpy as np

from neu4mes.input import closedloop_name, connect_name
from neu4mes.utils import check, merge
from neu4mes.relation import MAIN_JSON, Stream
from neu4mes.input import Input, State
from neu4mes.output import Output

class ModelDef():
    def __init__(self, model_def = None):
        # Inizialize the model definition
        self.model_dict = {}
        self.minimize_dict = {}
        self.update_state_dict = {}

        # Models definition
        self.model_def = model_def
        self.sample_time = None

    def __contains__(self, key):
        return key in self.model_def

    def __getitem__(self, key):
        return self.model_def[key]

    def __setitem__(self, key, value):
        self.model_def[key] = value

    def isDefined(self):
        return self.model_def is not None

    def update(self, model_def = MAIN_JSON, model_dict = None, minimize_dict = None, update_state_dict = None):
        self.model_def = copy.deepcopy(model_def)
        model_dict = copy.deepcopy(model_dict) if model_dict is not None else self.model_dict
        minimize_dict = copy.deepcopy(minimize_dict) if minimize_dict is not None else self.minimize_dict
        update_state_dict = copy.deepcopy(
            update_state_dict) if update_state_dict is not None else self.update_state_dict

        # Add models to the model_def
        for key, stream_list in model_dict.items():
            for stream in stream_list:
                self.model_def = merge(self.model_def, stream.json)
        if len(model_dict) > 1:
            if 'Models' not in self.model_def:
                self.model_def['Models'] = {}
            for model_name, model_params in model_dict.items():
                self.model_def['Models'][model_name] = {'Inputs': [], 'States': [], 'Outputs': [], 'Parameters': [],
                                                        'Constants': []}
                parameters, constants, inputs, states = set(), set(), set(), set()
                for param in model_params:
                    self.model_def['Models'][model_name]['Outputs'].append(param.name)
                    parameters |= set(param.json['Parameters'].keys())
                    constants |= set(param.json['Constants'].keys())
                    inputs |= set(param.json['Inputs'].keys())
                    states |= set(param.json['States'].keys())
                self.model_def['Models'][model_name]['Parameters'] = list(parameters)
                self.model_def['Models'][model_name]['Constants'] = list(constants)
                self.model_def['Models'][model_name]['Inputs'] = list(inputs)
                self.model_def['Models'][model_name]['States'] = list(states)
        elif len(model_dict) == 1:
            self.model_def['Models'] = list(model_dict.keys())[0]

        if 'Minimizers' not in self.model_def:
            self.model_def['Minimizers'] = {}
        for key, minimize in minimize_dict.items():
            self.model_def = merge(self.model_def, minimize['A'].json)
            self.model_def = merge(self.model_def, minimize['B'].json)
            self.model_def['Minimizers'][key] = {}
            self.model_def['Minimizers'][key]['A'] = minimize['A'].name
            self.model_def['Minimizers'][key]['B'] = minimize['B'].name
            self.model_def['Minimizers'][key]['loss'] = minimize['loss']

        for key, update_state in update_state_dict.items():
            self.model_def = merge(self.model_def, update_state.json)

        if "SampleTime" in self.model_def['Info']:
            self.sample_time = self.model_def['Info']["SampleTime"]


    def __update_state(self, stream_out, state_list_in, UpdateState):
        from neu4mes.input import  State
        if type(state_list_in) is not list:
            state_list_in = [state_list_in]
        for state_in in state_list_in:
            check(isinstance(stream_out, (Output, Stream)), TypeError,
                  f"The {stream_out} must be a Stream or Output and not a {type(stream_out)}.")
            check(type(state_in) is State, TypeError,
                  f"The {state_in} must be a State and not a {type(state_in)}.")
            check(stream_out.dim['dim'] == state_in.dim['dim'], ValueError,
                  f"The dimension of {stream_out.name} is not equal to the dimension of {state_in.name} ({stream_out.dim['dim']}!={state_in.dim['dim']}).")
            if type(stream_out) is Output:
                stream_name = self.model_def['Outputs'][stream_out.name]
                stream_out = Stream(stream_name,stream_out.json,stream_out.dim, 0)
            self.update_state_dict[state_in.name] = UpdateState(stream_out, state_in)

    def addConnect(self, stream_out, state_list_in):
        from neu4mes.input import Connect
        self.__update_state(stream_out, state_list_in, Connect)
        self.update()

    def addClosedLoop(self, stream_out, state_list_in):
        from neu4mes.input import ClosedLoop
        self.__update_state(stream_out, state_list_in, ClosedLoop)
        self.update()

    def addModel(self, name, stream_list):
        if isinstance(stream_list, (Output,Stream)):
            stream_list = [stream_list]
        if type(stream_list) is list:
            self.model_dict[name] = copy.deepcopy(stream_list)
        else:
            raise TypeError(f'stream_list is type {type(stream_list)} but must be an Output or Stream or a list of them')
        self.update()

    def removeModel(self, name_list):
        if type(name_list) is str:
            name_list = [name_list]
        if type(name_list) is list:
            for name in name_list:
                check(name in self.model_dict, IndexError, f"The name {name} is not part of the available models")
                del self.model_dict[name]
        self.update()

    def addMinimize(self, name, streamA, streamB, loss_function='mse'):
        check(isinstance(streamA, (Output, Stream)), TypeError, 'streamA must be an instance of Output or Stream')
        check(isinstance(streamB, (Output, Stream)), TypeError, 'streamA must be an instance of Output or Stream')
        check(streamA.dim == streamB.dim, ValueError, f'Dimension of streamA={streamA.dim} and streamB={streamB.dim} are not equal.')
        self.minimize_dict[name]={'A':copy.deepcopy(streamA), 'B': copy.deepcopy(streamB), 'loss':loss_function}
        self.update()

    def removeMinimize(self, name_list):
        if type(name_list) is str:
            name_list = [name_list]
        if type(name_list) is list:
            for name in name_list:
                check(name in self.minimize_dict, IndexError, f"The name {name} is not part of the available minimuzes")
                del self.minimize_dict[name]
        self.update()

    def setBuildWindow(self, sample_time = None):
        check(self.model_def is not None, RuntimeError, "No model is defined!")
        if sample_time is not None:
            check(sample_time > 0, RuntimeError, 'Sample time must be strictly positive!')
            self.sample_time = sample_time
        else:
            if self.sample_time is None:
                self.sample_time = 1

        self.model_def['Info'] = {"SampleTime": self.sample_time}

        check(self.model_def['Inputs'] | self.model_def['States'] != {}, RuntimeError, "No model is defined!")
        json_inputs = self.model_def['Inputs'] | self.model_def['States']

        for key,value in self.model_def['States'].items():
            check(closedloop_name in self.model_def['States'][key] or connect_name in self.model_def['States'][key],
                  KeyError, f'Update function is missing for state {key}. Use Connect or ClosedLoop to update the state.')

        input_tw_backward, input_tw_forward, input_ns_backward, input_ns_forward = {}, {}, {}, {}
        for key, value in json_inputs.items():
            if value['sw'] == [0,0] and value['tw'] == [0,0]:
                assert(False), f"Input {key} has no time window or sample window"
            if value['sw'] == [0, 0] and self.sample_time is not None:
                input_ns_backward[key] = round(-value['tw'][0] / self.sample_time)
                input_ns_forward[key] = round(value['tw'][1] / self.sample_time)
            elif self.sample_time is not None:
                input_ns_backward[key] = max(round(-value['tw'][0] / self.sample_time),-value['sw'][0])
                input_ns_forward[key] = max(round(value['tw'][1] / self.sample_time),value['sw'][1])
            else:
                check(value['tw'] == [0,0], RuntimeError, f"Sample time is not defined for input {key}")
                input_ns_backward[key] = -value['sw'][0]
                input_ns_forward[key] = value['sw'][1]
            value['ns'] = [input_ns_backward[key], input_ns_forward[key]]
            value['ntot'] = sum(value['ns'])

        self.model_def['Info']['ns'] = [max(input_ns_backward.values()), max(input_ns_forward.values())]
        self.model_def['Info']['ntot'] = sum(self.model_def['Info']['ns'])

        for k, v in (self.model_def['Parameters']|self.model_def['Constants']).items():
            if 'values' in v:
                window = 'tw' if 'tw' in v.keys() else ('sw' if 'sw' in v.keys() else None)
                if window == 'tw':
                    check(np.array(v['values']).shape[0] == v['tw']/self.sample_time, ValueError,
                      f"{k} has a different number of values for this sample time.")


        # if self.model_def['Info']['ns'][0] < 0:
        #     self.visualizer.warning(
        #         f"The input is only in the far past the max_samples_backward is: {self.model_def['Info']['ns'][0]}")
        # if self.model_def['Info']['ns'][1] < 0:
        #     self.visualizer.warning(
        #         f"The input is only in the far future the max_sample_forward is: {self.model_def['Info']['ns'][1]}")

    def updateParameters(self, model):
        if model is not None:
            for key in self.model_def['Parameters'].keys():
                if key in model.all_parameters:
                    self.model_def['Parameters'][key]['values'] = model.all_parameters[key].tolist()
                    if 'init_fun' in self.model_def['Parameters'][key]:
                        del self.model_def['Parameters'][key]['init_fun']
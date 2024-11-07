from itertools import product
import numpy as np

import torch.nn as nn
import torch

import copy

class Model(nn.Module):
    def __init__(self, model_def):
        super(Model, self).__init__()
        model_def = copy.deepcopy(model_def)
        self.inputs = model_def['Inputs']
        self.outputs = model_def['Outputs']
        self.relations = model_def['Relations']
        self.params = model_def['Parameters']
        self.constants = model_def['Constants']
        self.sample_time = model_def['Info']['SampleTime']
        self.functions = model_def['Functions']
        self.state_model_main = model_def['States']
        self.minimizers = model_def['Minimizers']
        self.state_model = copy.deepcopy(self.state_model_main)
        self.input_ns_backward = {key:value['ns'][0] for key, value in (model_def['Inputs']|model_def['States']).items()}
        self.input_n_samples = {key:value['ntot'] for key, value in (model_def['Inputs']|model_def['States']).items()}
        self.minimizers_keys = [self.minimizers[key]['A'] for key in self.minimizers] + [self.minimizers[key]['B'] for key in self.minimizers]

        ## Build the network
        self.all_parameters = {}
        self.all_constants = {}
        self.relation_forward = {}
        self.relation_inputs = {}
        self.states = {}
        self.states_closed_loop = {}
        self.states_connect = {}
        #self.constants = set()

        self.connect_variables = {}
        self.connect = {}
        self.initialize_connect = False

        ## Define the correct slicing
        json_inputs = self.inputs | self.state_model
        for _, items in self.relations.items():
            if items[0] == 'SamplePart':
                if items[1][0] in json_inputs.keys():
                    items[2][0] = self.input_ns_backward[items[1][0]] + items[2][0]
                    items[2][1] = self.input_ns_backward[items[1][0]] + items[2][1]
                    if len(items) > 3: ## Offset
                        items[3] = self.input_ns_backward[items[1][0]] + items[3]
            if items[0] == 'TimePart':
                if items[1][0] in json_inputs.keys():
                    items[2][0] = self.input_ns_backward[items[1][0]] + round(items[2][0]/self.sample_time)
                    items[2][1] = self.input_ns_backward[items[1][0]] + round(items[2][1]/self.sample_time)
                    if len(items) > 3: ## Offset
                        items[3] = self.input_ns_backward[items[1][0]] + round(items[3]/self.sample_time)
                else:
                    items[2][0] = round(items[2][0]/self.sample_time)
                    items[2][1] = round(items[2][1]/self.sample_time)
                    if len(items) > 3: ## Offset
                        items[3] = round(items[3]/self.sample_time)

        ## Create all the parameters
        for name, param_data in self.params.items():
            window = 'tw' if 'tw' in param_data.keys() else ('sw' if 'sw' in param_data.keys() else None)
            aux_sample_time = self.sample_time if 'tw' == window else 1
            sample_window = round(param_data[window] / aux_sample_time) if window else 1
            param_size = (sample_window,)+tuple(param_data['dim']) if type(param_data['dim']) is list else (sample_window, param_data['dim'])
            if 'values' in param_data:
                self.all_parameters[name] = nn.Parameter(torch.tensor(param_data['values'], dtype=torch.float32), requires_grad=True)
            # TODO clean code
            elif 'init_fun' in param_data:
                exec(param_data['init_fun']['code'], globals())
                function_to_call = globals()[param_data['init_fun']['name']]
                values = np.zeros(param_size)
                for indexes in product(*(range(v) for v in param_size)):
                    if 'params' in param_data['init_fun']:
                        values[indexes] = function_to_call(indexes, param_size, param_data['init_fun']['params'])
                    else:
                        values[indexes] = function_to_call(indexes, param_size)
                self.all_parameters[name] = nn.Parameter(torch.tensor(values.tolist(), dtype=torch.float32), requires_grad=True)
            else:
                self.all_parameters[name] = nn.Parameter(torch.rand(size=param_size, dtype=torch.float32), requires_grad=True)

        ## Create all the constants
        for name, param_data in self.constants.items():
            self.all_constants[name] = nn.Parameter(torch.tensor(param_data['values'], dtype=torch.float32), requires_grad=False)


        ## Initialize state variables
        self.init_states(self.state_model_main, reset_states=True)
        all_params_and_consts = self.all_parameters | self.all_constants

        ## Create all the relations
        for relation, inputs in self.relations.items():
            ## Take the relation name
            rel_name = inputs[0]
            ## collect the inputs needed for the relation
            input_var = inputs[1]
            ## collect the constants of the model
            #self.constants.update([item for item in inputs[1] if not isinstance(item, str)])
            
            ## Create All the Relations
            func = getattr(self,rel_name)
            if func:
                layer_inputs = []
                for item in inputs[2:]:
                    if item in list(self.params.keys()): ## the relation takes parameters
                        layer_inputs.append(self.all_parameters[item])
                    elif item in list(self.constants.keys()): ## the relation takes parameters
                        layer_inputs.append(self.all_constants[item])
                    elif item in list(self.functions.keys()): ## the relation takes a custom function
                        layer_inputs.append(self.functions[item])
                        if 'params_and_consts' in self.functions[item].keys() and len(self.functions[item]['params_and_consts']) >= 0: ## Parametric function that takes parameters
                            layer_inputs.append([all_params_and_consts[par] for par in self.functions[item]['params_and_consts']])
                        if 'map_over_dim' in self.functions[item].keys():
                            layer_inputs.append(self.functions[item]['map_over_dim'])
                    else: 
                        layer_inputs.append(item)

                ## Initialize the relation
                self.relation_forward[relation] = func(*layer_inputs)
                ## Save the inputs needed for the relative relation
                self.relation_inputs[relation] = input_var

            else:
                print(f"Key Error: [{rel_name}] Relation not defined")

        ## Add the gradient to all the relations and parameters that requires it
        self.relation_forward = nn.ParameterDict(self.relation_forward)
        self.all_constants = nn.ParameterDict(self.all_constants)
        self.all_parameters = nn.ParameterDict(self.all_parameters)

        ## list of network outputs
        self.network_output_predictions = set(self.outputs.values())

        ## list of network minimization outputs
        self.network_output_minimizers = [] 
        for _,value in self.minimizers.items():
            self.network_output_minimizers.append(self.outputs[value['A']]) if value['A'] in self.outputs.keys() else self.network_output_minimizers.append(value['A'])
            self.network_output_minimizers.append(self.outputs[value['B']]) if value['B'] in self.outputs.keys() else self.network_output_minimizers.append(value['B'])
        self.network_output_minimizers = set(self.network_output_minimizers)

        ## list of all the network Outputs
        self.network_outputs = self.network_output_predictions.union(self.network_output_minimizers)

    def forward(self, kwargs):
        result_dict = {}

        ## Initially i have only the inputs from the dataset, the parameters, and the constants
        available_inputs = [key for key in self.inputs.keys() if key not in self.connect.keys()]  ## remove connected inputs
        available_states = [key for key in self.state_model.keys() if key not in self.states_connect.keys()] ## remove connected states
        available_keys = set(available_inputs + list(self.all_parameters.keys()) + list(self.all_constants.keys()) + available_states)

        ## Forward pass through the relations
        while not self.network_outputs.issubset(available_keys): ## i need to climb the relation tree until i get all the outputs
            for relation in self.relations.keys():
                ## if i have all the variables i can calculate the relation
                if set(self.relation_inputs[relation]).issubset(available_keys) and (relation not in available_keys):
                    ## Collect all the necessary inputs for the relation
                    layer_inputs = []
                    for key in self.relation_inputs[relation]:
                        if key in self.all_constants.keys(): ## relation that takes a constant
                            layer_inputs.append(self.all_constants[key])
                        elif key in self.states.keys(): ## relation that takes a state
                            layer_inputs.append(self.states[key])
                        elif key in available_inputs:  ## relation that takes inputs (self.inputs.keys())
                            layer_inputs.append(kwargs[key])
                        elif key in self.all_parameters.keys(): ## relation that takes parameters
                            layer_inputs.append(self.all_parameters[key])
                        else: ## relation than takes another relation or a connect variable
                            layer_inputs.append(result_dict[key])

                    ## Execute the current relation
                    result_dict[relation] = self.relation_forward[relation](*layer_inputs)
                    available_keys.add(relation)

                    ## Update the connect variables if necessary
                    for connect_in, connect_out in self.connect.items():
                        if relation == self.outputs[connect_out]:  ## we have to save the output
                            shift = result_dict[relation].shape[1]
                            self.connect_variables[connect_in] = torch.roll(self.connect_variables[connect_in], shifts=-1, dims=1)
                            self.connect_variables[connect_in][:, -shift:, :] = result_dict[relation]
                            result_dict[connect_in] = self.connect_variables[connect_in].clone()
                            available_keys.add(connect_in)

                    ## Update connect state if necessary
                    if relation in self.states_connect.values():
                        for state in [key for key, value in self.states_connect.items() if value == relation]:
                            shift = result_dict[relation].shape[1]
                            self.states[state] = torch.roll(self.states[state], shifts=-1, dims=1)
                            self.states[state][:, -shift:, :] = result_dict[relation]#.detach() ## TODO: detach??
                            available_keys.add(state)

        ## Update closed loop state if necessary
        for relation in self.relations.keys():
            if relation in self.states_closed_loop.values():
                for state in [key for key, value in self.states_closed_loop.items() if value == relation]:
                    shift = result_dict[relation].shape[1]
                    self.states[state] = torch.roll(self.states[state], shifts=-1, dims=1)  # shifts=-shift, dims=1)
                    self.states[state][:, -shift:, :] = result_dict[relation]  # .detach() ## TODO: detach??

        ## Return a dictionary with all the outputs final values
        output_dict = {key: result_dict[value] for key, value in self.outputs.items()}
        ## Return a dictionary with the minimization relations
        minimize_dict = {}
        for key in self.minimizers_keys:
            minimize_dict[key] = result_dict[self.outputs[key]] if key in self.outputs.keys() else result_dict[key]
                
        return output_dict, minimize_dict


    def init_states(self, state_model, connect = {}, reset_states = False):
        ## Initialize state variables
        if reset_states:
            self.reset_states()
        self.reset_connect_variables(copy.deepcopy(connect), only=False)
        self.states_connect = {}
        self.states_closed_loop = {}
        ## save the states updates
        for state, param in state_model.items():
            if 'connect' in param.keys():
                self.states_connect[state] = param['connect']
            else:
                self.states_closed_loop[state] = param['closedLoop']

    def reset_connect_variables(self, connect, values = None, only = True):
        if only == False:
            self.connect = connect
            self.connect_variables = {}
            self.initialize_connect = True
        for key in connect.keys():
            if values is not None and key in values.keys():
                self.connect_variables[key] = values[key].clone()
            elif only == False:
                batch = values[list(values)[0]].shape[0] if values is not None else 1
                window_size = self.input_n_samples[key]
                self.connect_variables[key] = torch.zeros(size=(batch, window_size, self.inputs[key]['dim']),
                                                           dtype=torch.float32, requires_grad=False)
    def reset_states(self, values = None, only = True):
        if values is None:
            for key, value in self.state_model.items():
                batch = self.states[key].shape[0] if key in self.states else 1
                window_size = self.input_n_samples[key]
                self.states[key] = torch.zeros(size=(batch, window_size, self.state_model[key]['dim']),
                                               dtype=torch.float32, requires_grad=False)
        else:
            if type(values) is set:
                for key in self.state_model.keys():
                    if key in values:
                        batch = self.states[key].shape[0] if key in self.states else 1
                        window_size = self.input_n_samples[key]
                        self.states[key] = torch.zeros(size=(batch, window_size, self.state_model[key]['dim']),
                                                   dtype=torch.float32, requires_grad=False)
            else:
                for key in self.state_model.keys():
                    if key in values.keys():
                        self.states[key] = values[key].clone()
                        self.states[key].requires_grad = False
                    elif only == False:
                        batch = values[list(values)[0]].shape[0]
                        window_size = self.input_n_samples[key]
                        self.states[key] = torch.zeros(size=(batch, window_size, self.state_model[key]['dim']),
                                                         dtype=torch.float32, requires_grad=False)

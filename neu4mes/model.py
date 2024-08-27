import torch.nn as nn
import torch

import copy

class Model(nn.Module):
    def __init__(self, model_def,  minimize_dict, input_ns_backward, input_n_samples):
        super(Model, self).__init__()
        self.inputs = model_def['Inputs']
        self.outputs = model_def['Outputs']
        self.relations = model_def['Relations']
        self.params = model_def['Parameters']
        self.sample_time = model_def['SampleTime']
        self.functions = model_def['Functions']
        self.state_model = model_def['States']
        self.minimizers = minimize_dict
        self.input_ns_backward = input_ns_backward
        self.input_n_samples = input_n_samples ## TODO: use this for all the windows
        self.minimizers_keys = [minimize_dict[key]['A'].name for key in minimize_dict] + [minimize_dict[key]['B'].name for key in minimize_dict]
        
        self.connect = {}
        self.batch_size = 1

        ## Build the network
        self.all_parameters = {}
        self.relation_forward = {}
        self.relation_inputs = {}
        self.relation_parameters = {}
        self.states = {}
        self.constants = set()
        self.connect_variables = {}

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
            window = 'tw' if 'tw' in param_data.keys() else ('sw' if 'sw' in self.params[name].keys() else None)
            aux_sample_time = self.sample_time if 'tw' == window else 1
            if window:
                sample_window = round(param_data[window] / aux_sample_time)
            else:
                sample_window = 1
            if type(param_data['dim']) is tuple:
                param_size = (sample_window,)+param_data['dim']
            else:
                param_size = (sample_window, param_data['dim'])
            if 'values' in param_data:
                self.all_parameters[name] = nn.Parameter(torch.tensor(param_data['values'], dtype=torch.float32),
                                                                 requires_grad=True)
            # TODO clean code
            elif 'init_fun' in param_data:
                exec(param_data['init_fun']['code'], globals())
                function_to_call = globals()[param_data['init_fun']['name']]
                from itertools import product
                import numpy as np
                values = np.zeros(param_size)
                for indexes in product(*(range(v) for v in param_size)):
                    if 'params' in param_data['init_fun']:
                        values[indexes] = function_to_call(indexes, param_size, param_data['init_fun']['params'])
                    else:
                        values[indexes] = function_to_call(indexes, param_size)
                self.all_parameters[name] = nn.Parameter(torch.tensor(values.tolist(), dtype=torch.float32),
                                                         requires_grad=True)
            else:
                self.all_parameters[name] = nn.Parameter(torch.rand(size=param_size, dtype=torch.float32),
                                                         requires_grad=True)

        ## Initialize state variables
        self.clear_state()
        ## save the states updates
        self.states_updates = {}
        for state, param in self.state_model.items():
            self.states_updates[state] = param['update']

        ## Create all the relations
        for relation, inputs in self.relations.items():
            ## Take the relation name
            rel_name = inputs[0]
            ## collect the inputs needed for the relation
            input_var = inputs[1]
            ## collect the constants of the model
            self.constants.update([item for item in inputs[1] if not isinstance(item, str)])
            
            ## Check shared layers
            if rel_name in ['Fir','Linear',]:
                if inputs[2] in self.relation_parameters.keys(): ## we have a shared layer
                    self.relation_forward[relation] = self.relation_forward[self.relation_parameters[inputs[2]]]
                    self.relation_inputs[relation] = input_var
                    continue

            ## Create All the Relations
            func = getattr(self,rel_name)
            if func:
                if rel_name == 'ParamFun': 
                    self.relation_forward[relation] = func(self.functions[inputs[2]])
                elif rel_name == 'Fuzzify':
                    self.relation_forward[relation] = func(self.functions[inputs[2]])
                elif rel_name == 'Fir':  
                    self.relation_forward[relation] = func(self.all_parameters[inputs[2]],inputs[3])
                elif rel_name == 'Linear':
                    if inputs[3]:
                        self.relation_forward[relation] = func(self.all_parameters[inputs[2]],self.all_parameters[inputs[3]], inputs[4])
                    else:
                        self.relation_forward[relation] = func(self.all_parameters[inputs[2]], None, inputs[4])
                elif rel_name == 'TimePart':
                    part = inputs[2]
                    offset = inputs[3] if len(inputs) > 3 else None
                    self.relation_forward[relation] = func(part, offset)
                elif rel_name == 'SamplePart':
                    part = inputs[2]
                    offset = inputs[3] if len(inputs) > 3 else None
                    self.relation_forward[relation] = func(part, offset)
                elif rel_name == 'Part':
                    part_start_idx, part_end_idx = inputs[2][0], inputs[2][1]
                    self.relation_forward[relation] = func(part_start_idx, part_end_idx)
                elif rel_name == 'Select' or rel_name == 'SampleSelect':
                    select_idx = inputs[2]
                    self.relation_forward[relation] = func(select_idx)
                else: ## All the Functions that takes no parameters
                    self.relation_forward[relation] = func()

                ## Save the inputs needed for the relative relation
                self.relation_inputs[relation] = input_var

                ## Add the shared layers
                if rel_name in ['Fir','Linear']:
                    if inputs[2] not in self.relation_parameters.keys():
                        self.relation_parameters[inputs[2]] = relation
            else:
                print(f"Key Error: [{rel_name}] Relation not defined")

        ## Add the gradient to all the relations and parameters that requires it
        self.relation_forward = nn.ParameterDict(self.relation_forward)
        self.all_parameters = nn.ParameterDict(self.all_parameters)
        ## TODO: add count number of parameters

        ## list of network outputs
        self.network_output_predictions = set(self.outputs.values())

        ## list of network minimization outputs
        self.network_output_minimizers = [] 
        for key,value in self.minimizers.items():
            self.network_output_minimizers.append(self.outputs[value['A'].name]) if value['A'].name in self.outputs.keys() else self.network_output_minimizers.append(value['A'].name)
            self.network_output_minimizers.append(self.outputs[value['B'].name]) if value['B'].name in self.outputs.keys() else self.network_output_minimizers.append(value['B'].name)
        self.network_output_minimizers = set(self.network_output_minimizers)

        ## list of all the network Outputs
        self.network_outputs = self.network_output_predictions.union(self.network_output_minimizers)

    def forward(self, kwargs):
        result_dict = {}

        ## Initially i have only the inputs from the dataset, the parameters, and the constants
        available_inputs = [key for key in self.inputs.keys() if key not in self.connect.keys()]  ## remove the connected inputs
        available_keys = set(available_inputs + list(self.all_parameters.keys()) + list(self.constants) + list(self.state_model.keys()))
        
        #batch_size = list(kwargs.values())[0].shape[0] if kwargs else 1 ## TODO: define the batch inside the init as a model variables so that the forward can work even with only states variables
        ## Initialize State variables if necessary
        for state in self.state_model.keys():
            if state in kwargs.keys(): ## the state variable must be initialized with the dataset values
                self.states[state] = kwargs[state].clone()
                self.states[state].requires_grad = False
            elif self.batch_size > self.states[state].shape[0]:
                self.states[state] = self.states[state].repeat(self.batch_size, 1, 1)
                self.states[state].requires_grad = False

        ## Forward pass through the relations
        while not self.network_outputs.issubset(available_keys): ## i need to climb the relation tree until i get all the outputs
            for relation in self.relations.keys():
                ## if i have all the variables i can calculate the relation
                if set(self.relation_inputs[relation]).issubset(available_keys) and (relation not in available_keys):
                    ## Collect all the necessary inputs for the relation
                    layer_inputs = []
                    for key in self.relation_inputs[relation]:
                        if not isinstance(key, str): ## relation that takes a constant
                            layer_inputs.append(torch.tensor(key, dtype=torch.float32))
                        elif key in self.states.keys(): ## relation that takes a state
                            layer_inputs.append(self.states[key])
                        elif key in available_inputs:  ## relation that takes inputs (self.inputs.keys())
                            layer_inputs.append(kwargs[key])
                        elif key in self.all_parameters.keys(): ## relation that takes parameters
                            layer_inputs.append(self.all_parameters[key])
                        else: ## relation than takes another relation or a connect variable
                            layer_inputs.append(result_dict[key])

                    ## Execute the current relation
                    #print('relation to execute: ', relation)
                    #print('inputs: ', layer_inputs)
                    if 'ParamFun' in relation:
                        layer_parameters = []
                        func_parameters = self.functions[self.relations[relation][2]]['parameters']
                        for func_par in func_parameters:
                            layer_parameters.append(self.all_parameters[func_par])
                        result_dict[relation] = self.relation_forward[relation](layer_inputs, layer_parameters)
                    else:
                        if len(layer_inputs) <= 1: ## i have a single forward pass
                            result_dict[relation] = self.relation_forward[relation](layer_inputs[0])
                        else:
                            result_dict[relation] = self.relation_forward[relation](layer_inputs)
                    #print('result relation: ', result_dict[relation])
                    available_keys.add(relation)

                    ## Update the connect variables if necessary
                    for connect_in, connect_out in self.connect.items():
                        if connect_in in available_keys:
                            continue
                        if relation == self.outputs[connect_out]:  ## we have to save the output
                            window_size = self.input_n_samples[connect_in]
                            relation_size = result_dict[relation].shape[1]
                            if relation_size > window_size:
                                result_dict[connect_in] = result_dict[relation][:, window_size:, :].clone()
                            elif relation_size == window_size:
                                result_dict[connect_in] = result_dict[relation].clone()
                            else: ## input window is bigger than the output window
                                if connect_in not in self.connect_variables:  ## initialization
                                    if connect_in in kwargs.keys(): ## initialize with dataset
                                        self.connect_variables[connect_in] = kwargs[connect_in]
                                    else: ## initialize with zeros
                                        self.connect_variables[connect_in] = torch.zeros(size=(self.batch_size, window_size, self.inputs[connect_in]['dim']), dtype=torch.float32, requires_grad=True)
                                    result_dict[connect_in] = self.connect_variables[connect_in].clone()
                                else: ## update connect variable
                                    result_dict[connect_in] = torch.roll(self.connect_variables[connect_in], shifts=-relation_size, dims=1)
                                result_dict[connect_in][:, -relation_size:, :] = result_dict[relation].clone() 
                                self.connect_variables[connect_in] = result_dict[connect_in].clone()
                            ## add the new input
                            available_keys.add(connect_in)

                    ## Update the state if necessary
                    if relation in self.states_updates.values():
                        for state in [key for key, value in self.states_updates.items() if value == relation]:
                            shift = result_dict[relation].shape[1]
                            self.states[state] = torch.roll(self.states[state], shifts=-shift, dims=1)
                            self.states[state][:, -shift:, :] = result_dict[relation].detach() ## TODO: detach??
                            self.states[state].requires_grad = False
                        
        ## Return a dictionary with all the outputs final values
        output_dict = {key: result_dict[value] for key, value in self.outputs.items()}

        ## Return a dictionary with the minimization relations
        minimize_dict = {}
        for key in self.minimizers_keys:
            if key in self.outputs.keys():
                minimize_dict[key] = result_dict[self.outputs[key]]
            else:
                minimize_dict[key] = result_dict[key]
                
        return output_dict, minimize_dict
        
    def clear_state(self, state=None):
        if state: ## Clear a specific state variable
            if state in self.states.keys():
                window_size = self.input_n_samples[state]
                self.states[state] = torch.zeros(size=(1, window_size, self.state_model[state]['dim']), dtype=torch.float32, requires_grad=False)
        else: ## Clear all states variables
            self.states = {}
            for key, value in self.state_model.items():
                window_size = self.input_n_samples[key]
                self.states[key] = torch.zeros(size=(1, window_size, value['dim']), dtype=torch.float32, requires_grad=False)
    
    def clear_connect_variables(self, name=None):
        if name is None:
            self.connect_variables = {}
        else:
            if name in self.connect_variables.keys():
                del self.connect_variables[name]
            else:
                raise KeyError
    

    #window_size = round(max(abs(self.state_model[state]['sw'][0]), abs(self.state_model[state]['tw'][0]//self.sample_time)) +
    #                    max(self.state_model[state]['sw'][1], self.state_model[state]['tw'][1]//self.sample_time))

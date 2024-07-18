import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, model_def, minimize_list):
        super(Model, self).__init__()
        self.inputs = model_def['Inputs']
        self.outputs = model_def['Outputs']
        self.relations = model_def['Relations']
        self.params = model_def['Parameters']
        self.sample_time = model_def['SampleTime']
        self.functions = model_def['Functions']
        self.state_model = model_def['States']
        self.minimizers = minimize_list
        self.minimizers_keys = [i[0] for i in self.minimizers] + [i[1] for i in self.minimizers]

        ## Build the network
        self.all_parameters = {}
        self.relation_forward = {}
        self.relation_inputs = {}
        self.relation_parameters = {}
        self.states = {}
        self.constants = set()

        ## Create all the parameters
        for name, param_data in self.params.items():
            window = 'tw' if 'tw' in param_data.keys() else ('sw' if 'sw' in self.params[name].keys() else None)
            aux_sample_time = self.sample_time if 'tw' == window else 1
            if window:
                sample_window = round(param_data[window] / aux_sample_time)
            else:
                sample_window = 1
            if type(param_data['dim']) is tuple:
                param_size = tuple(param_data['dim'])
            else:
                param_size = (sample_window, param_data['dim'])
            if 'values' in param_data:
                self.all_parameters[name] = nn.Parameter(torch.tensor(param_data['values'], dtype=torch.float32), requires_grad=True)
            else:
                self.all_parameters[name] = nn.Parameter(torch.rand(size=param_size, dtype=torch.float32), requires_grad=True)

        ## save the states updates
        self.states_updates = {}
        for state, param in self.state_model.items():
            self.states_updates[param['update']] = state
        print('states_updates: ', self.states_updates)

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
                    self.relation_forward[relation] = func(self.all_parameters[inputs[2]])
                elif rel_name == 'Linear':
                    self.relation_forward[relation] = func(self.all_parameters[inputs[2]],inputs[3])
                elif rel_name == 'TimePart':
                    part = inputs[2]
                    offset = inputs[3] if len(inputs) > 3 else None
                    self.relation_forward[relation] = func(part, offset, self.sample_time)
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
                if rel_name in ['Fir',]:
                    if inputs[2] not in self.relation_parameters.keys():
                        self.relation_parameters[inputs[2]] = relation
            else:
                print(f"Key Error: [{rel_name}] Relation not defined")

        ## Add the gradient to all the relations and parameters that requires it
        self.relation_forward = nn.ParameterDict(self.relation_forward)
        self.all_parameters = nn.ParameterDict(self.all_parameters)

        ## list of network outputs
        self.network_output_predictions = set(self.outputs.values())

        ## list of network minimization outputs
        self.network_output_minimizers = [] 
        for el1, el2, _ in self.minimizers:
            self.network_output_minimizers.append(self.outputs[el1]) if el1 in self.outputs.keys() else self.network_output_minimizers.append(el1)
            self.network_output_minimizers.append(self.outputs[el2]) if el2 in self.outputs.keys() else self.network_output_minimizers.append(el2)
        self.network_output_minimizers = set(self.network_output_minimizers)

        ## list of all the network Outputs
        self.network_outputs = self.network_output_predictions.union(self.network_output_minimizers)

    def forward(self, kwargs, initialize_state=False):
        result_dict = {}

        ## Initially i have only the inputs from the dataset, the parameters, and the constants
        available_keys = set(list(self.inputs.keys()) + list(self.all_parameters.keys()) + list(self.constants) + list(self.state_model.keys()))
        ## Initialize the state variables
        if initialize_state:
            print('INITIALIZE STATE')
            for state, value in self.state_model.items():
                print('state key: ', state)
                if state in self.inputs.keys(): ## the state variable must be initialized with the dataset values
                    self.states[state] = kwargs[state].clone()
                    self.states[state].requires_grad = False
                else: ## the state variable must be initialized with zeros
                    batch_size = list(kwargs.values())[0].shape[0]
                    window_size = round(max(abs(value['sw'][0]), abs(value['tw'][0]//self.sample_time)) + max(value['sw'][1], value['tw'][1]//self.sample_time))
                    self.states[state] = torch.zeros(size=(batch_size, window_size, value['dim']), dtype=torch.float32, requires_grad=False)
                print('shape: ', self.states[state].shape)
                print('state value: ', self.states[state])

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
                        elif key in self.inputs.keys():  ## relation that takes inputs
                            layer_inputs.append(kwargs[key])
                        elif key in self.all_parameters.keys(): ## relation that takes parameters
                            layer_inputs.append(self.all_parameters[key])
                        else: ## relation than takes another relation
                            layer_inputs.append(result_dict[key])
                    #print('relation: ', relation)
                    #print('layer_inputs: ', layer_inputs)
                    ## Execute the current relation
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
                    available_keys.add(relation)

                    ## Update the state if necessary
                    if relation in self.states_updates.keys():
                        print('relation to update: ', relation)
                        print('Update State..')
                        shift = result_dict[relation].shape[1]
                        self.states[self.states_updates[relation]] = torch.roll(self.states[self.states_updates[relation]], shifts=-shift, dims=1)
                        self.states[self.states_updates[relation]][:, -shift:, :] = result_dict[relation].detach().clone()
                        print('Updated state: ', self.states[self.states_updates[relation]])
                        '''
                        temp = torch.roll(self.states[self.states_updates[relation]].detach().clone(), shifts=-shift, dims=1)
                        print('temp: ', temp)
                        temp = torch.cat((temp[:, :-shift, :], result_dict[relation].detach().clone()), dim=1)
                        print('temp update: ', temp)
                        self.states[self.states_updates[relation]] = temp
                        '''

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

'''
class Model(nn.Module):
    def __init__(self, model_def, minimize_list):
        super(Model, self).__init__()
        self.inputs = model_def['Inputs']
        self.outputs = model_def['Outputs']
        self.relations = model_def['Relations']
        self.params = model_def['Parameters']
        self.sample_time = model_def['SampleTime']
        self.functions = model_def['Functions']
        self.minimizers = minimize_list
        self.minimizers_keys = [i[0] for i in self.minimizers] + [i[1] for i in self.minimizers]

        ## Build the network
        self.all_parameters = {}
        self.relation_forward = {}
        self.relation_inputs = {}
        self.relation_parameters = {}
        self.constants = set()

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
                self.all_parameters[name] = nn.Parameter(torch.tensor(param_data['values'], dtype=torch.float32), requires_grad=True)
            else:
                self.all_parameters[name] = nn.Parameter(torch.rand(size=param_size, dtype=torch.float32), requires_grad=True)

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
                    self.relation_forward[relation] = func(self.all_parameters[inputs[2]])
                elif rel_name == 'Linear':
                    if inputs[3]:
                        self.relation_forward[relation] = func(self.all_parameters[inputs[2]],self.all_parameters[inputs[3]])
                    else:
                        self.relation_forward[relation] = func(self.all_parameters[inputs[2]],None)
                elif rel_name == 'TimePart':
                    part = inputs[2]
                    offset = inputs[3] if len(inputs) > 3 else None
                    self.relation_forward[relation] = func(part, offset, self.sample_time)
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
                if rel_name in ['Fir',]:
                    if inputs[2] not in self.relation_parameters.keys():
                        self.relation_parameters[inputs[2]] = relation
            else:
                print(f"Key Error: [{rel_name}] Relation not defined")

        ## Add the gradient to all the relations and parameters that requires it
        self.relation_forward = nn.ParameterDict(self.relation_forward)
        self.all_parameters = nn.ParameterDict(self.all_parameters)

        ## list of network outputs
        self.network_output_predictions = set(self.outputs.values())

        ## list of network minimization outputs
        self.network_output_minimizers = [] 
        for el1, el2, _ in self.minimizers:
            self.network_output_minimizers.append(self.outputs[el1]) if el1 in self.outputs.keys() else self.network_output_minimizers.append(el1)
            self.network_output_minimizers.append(self.outputs[el2]) if el2 in self.outputs.keys() else self.network_output_minimizers.append(el2)
        self.network_output_minimizers = set(self.network_output_minimizers)

        ## list of all the network Outputs
        self.network_outputs = self.network_output_predictions.union(self.network_output_minimizers)

    def forward(self, kwargs):
        result_dict = {}

        ## Initially i have only the inputs from the dataset, the parameters, and the constants
        available_keys = list(self.inputs.keys()) + list(self.all_parameters.keys()) + list(self.constants)

        ## Forward pass through the relations
        while not self.network_outputs.issubset(available_keys): ## i need to climb the relation tree until i get all the outputs
            for relation in self.relations.keys():
                ## if i have all the variables i can calculate the relation
                if set(self.relation_inputs[relation]).issubset(available_keys):
                    ## Collect all the necessary inputs for the relation
                    layer_inputs = []
                    for key in self.relation_inputs[relation]:
                        if not isinstance(key, str): ## relation that takes a constant
                            layer_inputs.append(torch.tensor(key, dtype=torch.float32))
                        elif key in self.inputs.keys():  ## relation that takes inputs
                            layer_inputs.append(kwargs[key])
                        elif key in self.all_parameters.keys(): ## relation that takes parameters
                            layer_inputs.append(self.all_parameters[key])
                        else: ## relation than takes another relation
                            layer_inputs.append(result_dict[key])

                    ## Execute the current relation
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
                    available_keys.append(relation)

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
'''
        



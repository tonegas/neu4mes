import torch.nn as nn
import torch

'''
class Model(nn.Module):
    def __init__(self, model_def, relation_samples):
        super(Model, self).__init__()
        self.inputs = model_def['Inputs']
        self.outputs = model_def['Outputs']
        self.relations = model_def['Relations']
        self.params = model_def['Parameters']
        self.sample_time = model_def['SampleTime']
        self.functions = model_def['Functions']
        self.samples = relation_samples

        ## Build the network
        self.all_parameters = {}
        self.relation_forward = {}
        self.relation_inputs = {}
        self.relation_parameters = {}

        ## Create all the parameters
        for name, param_dimensions in self.params.items():
            window = 'tw' if 'tw' in self.params[name].keys() else ('sw' if 'sw' in self.params[name].keys() else None)
            aux_sample_time = self.sample_time if 'tw' == window else 1
            if window:
                sample_window = round(param_dimensions[window] / aux_sample_time)
            else:
                sample_window = 1
            if type(param_dimensions['dim']) is list:
                param_size = tuple(param_dimensions['dim'])
            else:
                param_size = (param_dimensions['dim'], sample_window)
            self.all_parameters[name] = nn.Parameter(torch.rand(size=param_size), requires_grad=True)

        ## Create all the relations
        for relation, inputs in (self.relations|self.outputs).items():
            if relation in self.relations:
                rel_name = inputs[0]
                ## collect the inputs needed for the relation
                input_var = [i[0] if type(i) is tuple else i for i in inputs[1]]
            else:
                if type(inputs) is tuple:
                    self.relation_inputs[relation] = [inputs[0]]
                else:
                    self.relation_inputs[relation] = [inputs]
                continue
            
            ## Check shared layers
            if len(inputs) >= 3 and rel_name not in ('TimePart','SamplePart'): #TODO must to be fixed
                if inputs[2] in self.relation_parameters.keys(): ## we have a shared layer
                    self.relation_forward[relation] = self.relation_forward[self.relation_parameters[inputs[2]]]
                    self.relation_inputs[relation] = input_var
                    continue

            func = getattr(self,rel_name)
            if func:
                if rel_name == 'ParamFun': 
                    self.relation_forward[relation] = func(self.functions[inputs[2]])
                elif rel_name == 'Fuzzify':
                    self.relation_forward[relation] = func(self.functions[inputs[2]])
                elif rel_name == 'Fir':  ## Linear module requires 2 inputs: input_size and output_size
                    self.relation_forward[relation] = func(self.all_parameters[inputs[2]])
                else: ## Functions that takes no parameters
                    self.relation_forward[relation] = func()
                self.relation_inputs[relation] = input_var

                ## Add the shared layers
                if len(inputs) >= 3 and rel_name not in ('TimePart','SamplePart'): #TODO must to be fixed
                    if inputs[2] not in self.relation_parameters.keys():
                        self.relation_parameters[inputs[2]] = relation   
            else:
                print("Relation not defined")
        self.relation_forward = nn.ParameterDict(self.relation_forward)
        self.all_parameters = nn.ParameterDict(self.all_parameters)

        #print('[LOG] relation forward: ', self.relation_forward)
        #print('[LOG] relation inputs: ', self.relation_inputs)
        #print('[LOG] relation parameters: ', self.relation_parameters)
        #print('[LOG] samples: ', self.samples)
    
    def forward(self, kwargs):
        available_inputs = {}
        inputs_keys = list(self.inputs.keys())
        outputs_list = set([elem[0] if type(elem) is tuple else elem for elem in self.outputs.values()])
        while not outputs_list.issubset(inputs_keys):
            for output in (self.relations|self.outputs).keys():
                ## if i have all the variables i can calculate the relation
                if (output not in inputs_keys) and (set(self.relation_inputs[output]).issubset(inputs_keys)):
                    ## Layer_inputs: Selects the portion of the window from the complete vector that needs to be used for the current layer
                    #layer_inputs = [available_inputs[key][:,self.samples[output][key]['backward']:self.samples[output][key]['forward']] for key in self.relation_inputs[output]]
                    layer_inputs = []
                    for key in self.relation_inputs[output]:
                        if key in self.inputs.keys():
                            if kwargs[key].ndim == 1:
                                temp = kwargs[key][:,self.samples[output][key]['start_idx']]
                            else:
                                temp = kwargs[key][:,self.samples[output][key]['start_idx']:self.samples[output][key]['end_idx']]
                        else:
                            temp = available_inputs[key]
                        if 'offset_idx' in self.samples[output][key]:
                            temp = temp - temp[:,self.samples[output][key]['offset_idx']-self.samples[output][key]['start_idx']-1]
                        layer_inputs.append(temp)

                        if output in self.outputs.keys():
                            if key in self.inputs.keys():
                                available_inputs[output] = layer_inputs[0]
                            else:
                                available_inputs[output] = available_inputs[key]
                    #print('[LOG] output: ', output)
                    #print('[LOG] layer_inputs: ', layer_inputs)
                    #print('[LOG] relation forward: ', self.relation_forward[output])
                    if output not in self.outputs.keys():
                        if 'ParamFun' in output:
                            layer_parameters = []
                            func_parameters = self.functions[self.relations[output][2]]['parameters']
                            for func_par in func_parameters:
                                layer_parameters.append(self.all_parameters[func_par])
                            available_inputs[output] = self.relation_forward[output](layer_inputs, layer_parameters)
                        else:
                            #print('[LOG] layer_inputs: ', layer_inputs)
                            #print('[LOG] output: ', output)
                            if len(layer_inputs) <= 1: ## i have a single forward pass
                                available_inputs[output] = self.relation_forward[output](layer_inputs[0])
                            else:
                                available_inputs[output] = self.relation_forward[output](layer_inputs)
                            #print('[LOG] available_inputs: ', available_inputs[output])
                        inputs_keys.append(output)

        ## Return a dictionary with all the outputs final values
        print('[LOG] available inputs: ', available_inputs.keys())
        result_dict = {key: available_inputs[key] for key in self.outputs.keys()}
        return result_dict
'''
## MODEL VERSION 2
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

        ## Create all the parameters
        for name, param_dimensions in self.params.items():
            window = 'tw' if 'tw' in param_dimensions.keys() else ('sw' if 'sw' in self.params[name].keys() else None)
            aux_sample_time = self.sample_time if 'tw' == window else 1
            if window:
                sample_window = round(param_dimensions[window] / aux_sample_time)
            else:
                sample_window = 1
            if type(param_dimensions['dim']) is tuple:
                param_size = tuple(param_dimensions['dim'])
            else:
                param_size = (param_dimensions['dim'], sample_window)
            self.all_parameters[name] = nn.Parameter(torch.rand(size=param_size), requires_grad=True)

        ## Create all the relations
        for relation, inputs in self.relations.items():
            ## Take the relation name
            rel_name = inputs[0]
            ## collect the inputs needed for the relation
            input_var = [i[0] if type(i) is tuple else i for i in inputs[1]]
            
            ## Check shared layers
            if rel_name in ['Fir','Linear',]:
                if inputs[2] in self.relation_parameters.keys(): ## we have a shared layer
                    self.relation_forward[relation] = self.relation_forward[self.relation_parameters[inputs[2]]]
                    self.relation_inputs[relation] = input_var
                    continue

            func = getattr(self,rel_name)
            if func:
                if rel_name == 'ParamFun': 
                    self.relation_forward[relation] = func(self.functions[inputs[2]])
                elif rel_name == 'Fuzzify':
                    self.relation_forward[relation] = func(self.functions[inputs[2]])
                elif rel_name == 'Fir':  ## Linear module requires 2 inputs: input_size and output_size
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
                else: ## Functions that takes no parameters
                    self.relation_forward[relation] = func()

                self.relation_inputs[relation] = input_var

                ## Add the shared layers
                if rel_name in ['Fir',]:
                    if inputs[2] not in self.relation_parameters.keys():
                        self.relation_parameters[inputs[2]] = relation   
            else:
                print(f"Key Error: [{rel_name}] Relation not defined")

        self.relation_forward = nn.ParameterDict(self.relation_forward)
        self.all_parameters = nn.ParameterDict(self.all_parameters)

        #print('[LOG] relation forward: ', self.relation_forward)
        #print('[LOG] all_parameters: ', self.all_parameters)
        #print('[LOG] relation inputs: ', self.relation_inputs)
        #print('[LOG] relation parameters: ', self.relation_parameters)

        self.network_output_predictions = set(self.outputs.values())
        self.network_output_minimizers = []  ## TODO: the minimize list now already has the correct relations
        for el1, el2, _ in self.minimizers:
            self.network_output_minimizers.append(self.outputs[el1]) if el1 in self.outputs.keys() else self.network_output_minimizers.append(el1)
            self.network_output_minimizers.append(self.outputs[el2]) if el2 in self.outputs.keys() else self.network_output_minimizers.append(el2)    
        self.network_output_minimizers = set(self.network_output_minimizers)
        #self.network_output_minimizers = set([i[0] for i in self.minimizers] + [i[1] for i in self.minimizers] - self.outputs.keys())
        self.network_outputs = self.network_output_predictions.union(self.network_output_minimizers)

        #print('[LOG] network_output_predictions: ', self.network_output_predictions)
        #print('[LOG] network_output_minimizers: ', self.network_output_minimizers)
        #print('[LOG] network_outputs: ', self.network_outputs)
    
    def forward(self, kwargs):
        result_dict = {}

        ## Initially i have only the inputs from the dataset
        available_keys = list(self.inputs.keys())

        while not self.network_outputs.issubset(available_keys): ## i need to climb the relation tree until i get all the outputs
            for relation in self.relations.keys():
                ## if i have all the variables i can calculate the relation
                if set(self.relation_inputs[relation]).issubset(available_keys):
                    #print('[LOG] relation: ', relation)
                    ## Collect all the necessary inputs for the relation
                    layer_inputs = []
                    for key in self.relation_inputs[relation]:
                        if key in self.inputs.keys():
                            layer_inputs.append(kwargs[key])
                        else:
                            layer_inputs.append(result_dict[key])
                    #print('[LOG] layer inputs: ', layer_inputs)
                    #print('[LOG] relation forward: ', self.relation_forward[relation])
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
                    #print('[LOG] result dict: ', result_dict[relation])
                    available_keys.append(relation)

        ## Return a dictionary with all the outputs final values
        output_dict = {key: result_dict[value] for key, value in self.outputs.items()}
        #minimize_dict = {key: result_dict[key] for key in self.minimizers_keys}
        minimize_dict = {}
        for key in self.minimizers_keys:
            if key in self.outputs.keys():
                minimize_dict[key] = result_dict[self.outputs[key]]
            else:
                minimize_dict[key] = result_dict[key]
                
        return output_dict, minimize_dict



        



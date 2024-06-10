import torch.nn as nn
import torch

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
            if len(inputs) >= 3:
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
                if len(inputs) >= 3:
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
                            temp = kwargs[key][:,self.samples[output][key]['start_idx']:self.samples[output][key]['end_idx']]
                        else:
                            temp = available_inputs[key]
                        if 'offset_idx' in self.samples[output][key]:
                            temp = temp - temp[:, self.samples[output][key]['offset_idx']:self.samples[output][key]['offset_idx']+1]
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
        result_dict = {key: available_inputs[key] for key in self.outputs.keys()}
        return result_dict

    '''
    class Model(nn.Module):
    def __init__(self, model_def, relation_samples):
        super(Model, self).__init__()
        self.inputs = model_def['Inputs']
        self.outputs = model_def['Outputs']
        self.relations = model_def['Relations']
        self.params = model_def['Parameters']
        self.sample_time = model_def['SampleTime']
        self.samples = relation_samples

        ## Build the network
        self.relation_forward = {}
        self.relation_inputs = {}
        self.relation_parameters = {}

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
            if len(inputs) >= 3:
                if inputs[2] in self.relation_parameters.keys(): ## we have a shared layer
                    self.relation_forward[relation] = self.relation_forward[self.relation_parameters[inputs[2]]]
                    self.relation_inputs[relation] = input_var
                    continue

            func = getattr(self,rel_name)
            if func:
                if rel_name == 'LocalModel': # TODO: Work in progress
                    pass
                elif rel_name == 'Fir':  ## Linear module requires 2 inputs: input_size and output_size
                    window = 'tw' if 'tw' in self.params[inputs[2]].keys() else ('sw' if 'sw' in self.params[inputs[2]].keys() else None)
                    aux_sample_time = self.sample_time if 'tw' == window else 1
                    if window:
                        dim_in =  int(self.params[inputs[2]][window] / aux_sample_time)
                        self.relation_forward[relation] = func(dim_in, self.params[inputs[2]]['dim'])
                    else:
                        self.relation_forward[relation] = func(1, self.params[inputs[2]]['dim'])
                else: ## Functions that takes no parameters
                    self.relation_forward[relation] = func()
                self.relation_inputs[relation] = input_var

                ## Add the shared layers
                if len(inputs) >= 3:
                    if inputs[2] not in self.relation_parameters.keys():
                        self.relation_parameters[inputs[2]] = relation   
            else:
                print("Relation not defined")
        self.params = nn.ParameterDict(self.relation_forward)

        print('[LOG] relation forward: ', self.relation_forward)
        print('[LOG] relation inputs: ', self.relation_inputs)
        print('[LOG] relation parameters: ', self.relation_parameters)
    
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
                            temp = kwargs[key][:,self.samples[output][key]['backward']:self.samples[output][key]['forward']]
                        else:
                            temp = available_inputs[key]
                        if 'offset' in self.samples[output][key]:
                            temp = temp - temp[:, self.samples[output][key]['offset']:self.samples[output][key]['offset']+1]
                        layer_inputs.append(temp)

                        if output in self.outputs.keys():
                            if key in self.inputs.keys():
                                available_inputs[output] = layer_inputs[0]
                            else:
                                available_inputs[output] = available_inputs[key]

                    if output not in self.outputs.keys():
                        if len(layer_inputs) <= 1: ## i have a single forward pass
                            available_inputs[output] = self.relation_forward[output](layer_inputs[0])
                        else:
                            available_inputs[output] = self.relation_forward[output](layer_inputs)
                        inputs_keys.append(output)

        ## Return a dictionary with all the outputs final values
        result_dict = {key: available_inputs[val] for key, val in self.outputs.items()}
        return result_dict
    '''
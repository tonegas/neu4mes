import torch.nn as nn

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

        for relation, inputs in self.relations.items():
            rel_name = inputs[0]

            ## collect the inputs needed for the relation
            input_var = [i[0] if type(i) is tuple else i for i in inputs[1]]
            
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
        while not set(self.outputs.values()).issubset(inputs_keys):
            for output in self.relations.keys():
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
                        if self.samples[output][key]['offset'] is not None:
                            temp = temp - temp[:, self.samples[output][key]['offset']:self.samples[output][key]['offset']+1]
                        layer_inputs.append(temp)

                    if len(layer_inputs) <= 1: ## i have a single forward pass
                        available_inputs[output] = self.relation_forward[output](layer_inputs[0])
                    else:
                        available_inputs[output] = self.relation_forward[output](layer_inputs)
                    inputs_keys.append(output)

        ## Return a dictionary with all the outputs final values
        result_dict = {key: available_inputs[key] for key in self.outputs.values()}
        return result_dict

    '''
    def forward(self, kwargs):
        available_inputs = kwargs
        while not set(self.outputs.keys()).issubset(available_inputs.keys()):
            for output in self.relations.keys():
                ## if i have all the variables i can calculate the relation
                if (output not in available_inputs.keys()) and (set(self.relation_inputs[output]).issubset(available_inputs.keys())):
                    ## Layer_inputs: Selects the portion of the window from the complete vector that needs to be used for the current layer
                    #layer_inputs = [available_inputs[key][:,self.samples[output][key]['backward']:self.samples[output][key]['forward']] for key in self.relation_inputs[output]]

                    layer_inputs = []
                    for key in self.relation_inputs[output]:
                        temp = available_inputs[key][:,self.samples[output][key]['backward']:self.samples[output][key]['forward']]
                        if self.samples[output][key]['offset'] is not None:
                            print('[LOG] temp: ', temp)
                            temp = temp - temp[self.samples[output][key]['offset']]
                            print('[LOG] temp with offset: ', temp)
                        layer_inputs.append(temp)

                    if len(layer_inputs) <= 1: ## i have a single forward pass
                        available_inputs[output] = self.relation_forward[output](layer_inputs[0])
                    else:
                        available_inputs[output] = self.relation_forward[output](layer_inputs)

        ## Return a dictionary with all the outputs final values
        result_dict = {key: available_inputs[key] for key in self.outputs.keys()}
        return result_dict
    '''
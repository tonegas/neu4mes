import copy
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
                    #self.relation_forward[relation] = func(self.n_samples[input_var[0]], len(self.inputs[input_var[1]]['Discrete']))
                    #self.n_samples[relation] = len(self.inputs[input_var[1]]['Discrete'])
                elif rel_name == 'Fir':  ## Linear module requires 2 inputs: input_size and output_size
                    if set(['dim_in', 'dim_out']).issubset(self.params[inputs[2]].keys()):
                        self.relation_forward[relation] = func(self.params[inputs[2]]['dim_in'], self.params[inputs[2]]['dim_out'])
                    elif 'tw_in' in self.params[inputs[2]].keys():
                        if type(self.params[inputs[2]]['tw_in']) is list:  ## Backward + forward
                            dim_in = int(abs(self.params[inputs[2]]['tw_in'][0]) / self.sample_time) + int(abs(self.params[inputs[2]]['tw_in'][1]) / self.sample_time)
                        else:
                            dim_in =  int(self.params[inputs[2]]['tw_in'] / self.sample_time)
                        self.relation_forward[relation] = func(dim_in, self.params[inputs[2]]['dim_out'])
                    elif 'tw_out' in self.params[inputs[2]].keys():
                        if type(self.params[inputs[2]]['tw_out']) is list:  ## Backward + forward
                            dim_out = int(abs(self.params[inputs[2]]['tw_out'][0]) / self.sample_time) + int(abs(self.params[inputs[2]]['tw_out'][1]) / self.sample_time)
                        else:
                            dim_out =  int(self.params[inputs[2]]['tw_out'] / self.sample_time)
                        self.relation_forward[relation] = func(self.params[inputs[2]]['dim_in'], dim_out)

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
        while not set(self.outputs.keys()).issubset(inputs_keys):
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
        result_dict = {key: available_inputs[key] for key in self.outputs.keys()}
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
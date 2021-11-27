from pprint import pprint 
import tensorflow.keras.layers
import tensorflow.keras.models
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import string

import neu4mes 
import random

def rand(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def rmse(y_true, y_pred):
    # Root mean squared error (rmse) for regression
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

class RNNCell(tensorflow.keras.layers.Layer):
    def __init__(self, model, input_is_state, states_size, inputs_size, **kwargs):
        super(RNNCell, self).__init__(**kwargs)
        self.model = model
        self.input_is_state = input_is_state
        self.inputs_size = inputs_size
        self.state_size = states_size

    def call(self, inputs, states):

        # print('-------------------------------------')
        # print(states)
        # print(len(states))
        # print(self.state_size)
        # print(inputs)
        # print(self.input_is_state)
        # print('-------------------------------------')
        
        network_inputs, inputs_idx, states_idx = [], 0 ,0
        for is_state in self.input_is_state:
            if is_state:
                network_inputs.append(states[states_idx])
                states_idx += 1
            else:
                network_inputs.append(inputs[inputs_idx])
                inputs_idx += 1
        
        output = self.model(network_inputs)
        if type(output) is not list:
            output = [output] 
        # print('-------------------------------------')
        # print(output)
        # print('-------------------------------------')
        
        new_states = []
        for idx in range(len(states)):
            new_states.append(tf.concat([states[idx][:,1:self.state_size[idx]], output[idx]], axis=1))

        out = K.concatenate(output)
        return out, (new_states)

    def get_config(self):
        pass

class Neu4mes:
    def __init__(self, model_def = 0, verbose = False):
        # Set verbose print inside the class
        self.verbose = verbose

        # Inizialize the model_used and the model_def
        # the model_used is a reduced model that consider only the used input and relation
        # the model_def has all the relation defined for that model 
        self.model_used =  neu4mes.NeuObj().json
        if type(model_def) is neu4mes.Output:
            self.model_def = model_def.json
        elif type(model_def) is dict:
            self.model_def = self.model_def
        else:
            self.model_def = neu4mes.NeuObj().json

        # Input, output, and model characteristics 
        self.input_tw_backward = {}         # dimensions of the time window in the past for each input
        self.input_tw_forward = {}          # dimensions of the time window in the future for each input
        self.max_samples_backward = 0       # maxmimum number of samples backward for all the inputs
        self.max_samples_forward = 0        # maxmimum number of samples forward for all the inputs
        self.max_n_samples = 0              # maxmimum number of samples for all the inputs
        self.input_ns_backward = {}         # maxmimum number of samples backward for each the input
        self.input_ns_forward = {}          # maxmimum number of samples forward for each the input
        self.input_n_samples = {}           # maxmimum number of samples for each the input 

        self.inputs = {}                    # NN element - processed network inputs
        self.inputs_for_model = {}          # NN element - clean network inputs
        self.relations = {}                 # NN element - operations 
        self.outputs = {}                   # NN element - clean network outputs

        self.output_relation = {}           # dict with the outputs  
        self.output_keys = []               # clear output signal keys (without delay info string __-z1)
        
        self.model = 0                      # NN model - Keras model

        # Dataset characteristics
        self.input_data = {}                # dict with data divided by file and symbol key: input_data[(file,key)]
        self.inout_data_time_window = {}    # dict with data divided by signal ready for network input
        self.inout_asarray = {}             # dict for network input in asarray format

        self.idx_of_rows = [0] 
        
        #training
        self.batch_size = 128
        self.inout_4train = {}
        self.inout_4validation = {}
        self.learning_rate = 0.0005
        self.num_of_epochs = 200
        #rnn
        self.rnn_model = None
        self.rnn_window = None
        self.inputs_for_rnn_model = {}
        self.inout_rnn_asarray = {}
        self.inout_rnn_data_time_window = {}
        self.rnn_inputs = {}
        self.init_rnn_state = {}
        self.learning_rate_rnn = self.learning_rate/10000
        self.num_of_epochs_rnn = 50
        self.rnn_opt = None
        self.inout_rnn_4train = {}
        self.inout_rnn_4validation = {}

    def MP(self,fun,arg):
        if self.verbose:
            fun(arg)

    def addModel(self, model_def):
        """
        Add a new model to be trained: 
        :param model_def: can be a json model definition or a Output object 
        """
        if type(model_def) is neu4mes.Output:
            self.model_def = neu4mes.merge(self.model_def, model_def.json)
        elif type(model_def) is dict:
            self.model_def = neu4mes.merge(self.model_def, model_def) 
        self.MP(pprint,self.model_def)

    def neuralizeModel(self, sample_time = 0, prediction_window = None):
        """
        Definition of the network structure through the dependency graph and sampling time. 
        If a prediction window is also specified, it means that a recurrent network is also to be defined.
        :param sample_time: the variable defines the rate of the network based on the training set
        :param prediction_window: the variable defines the prediction horizon in the future
        """
        # Prediction window is used for the recurrent network
        if prediction_window is not None:
            self.rnn_window = round(prediction_window/sample_time)

        # Sample time is used to define the number of sample for each time window
        if sample_time:
            self.model_def["SampleTime"] = sample_time
        
        # Look for all the inputs referred to each outputs 
        # Set the maximum time window for each input
        relations = self.model_def['Relations']
        for outel in self.model_def['Outputs']:
            relel = relations.get(outel)
            for reltype, relvalue in relel.items():
                self.__setInput(relvalue, outel)
        
        self.MP(pprint,{"window_backward": self.input_tw_backward, "window_forward":self.input_tw_forward})

        # Building the inputs considering the dimension of the maximum time window forward + backward
        for key,val in self.model_def['Inputs'].items():
            input_ns_backward_aux = int(self.input_tw_backward[key]/self.model_def['SampleTime'])
            input_ns_forward_aux = int(-self.input_tw_forward[key]/self.model_def['SampleTime'])

            # Find the biggest window backwards for building the dataset
            if input_ns_backward_aux > self.max_samples_backward:
                self.max_samples_backward = input_ns_backward_aux
            
            # Find the biggest horizon forwars for building the dataset
            if input_ns_forward_aux > self.max_samples_forward:
                self.max_samples_forward = input_ns_forward_aux
            
            # Find the biggest n sample for building the dataset
            if input_ns_forward_aux+input_ns_backward_aux > self.max_n_samples:
                self.max_n_samples = input_ns_forward_aux+input_ns_backward_aux            
            
            # Defining the number of sample for each signal
            if self.input_n_samples.get(key):
                if input_ns_backward_aux+input_ns_forward_aux > self.input_n_samples[key]:
                    self.input_n_samples[key] = input_ns_backward_aux+input_ns_forward_aux
                if input_ns_backward_aux > self.input_ns_backward[key]:
                    self.input_ns_backward[key] = input_ns_backward_aux
                if input_ns_forward_aux > self.input_ns_forward[key]:
                    self.input_ns_forward[key] = input_ns_forward_aux
            else:
                self.input_n_samples[key] = input_ns_backward_aux+input_ns_forward_aux
                self.input_ns_backward[key] = input_ns_backward_aux
                self.input_ns_forward[key] = input_ns_forward_aux
                            
            # Building the signal object and fill the model_used structure
            if key not in self.model_used['Inputs']:
                if 'Discrete' in val:
                    (self.inputs_for_model[key],self.inputs[key]) = self.discreteInput(key, self.input_n_samples[key], val['Discrete'])
                    (self.inputs_for_rnn_model[key],self.rnn_inputs[key]) = self.discreteInputRNN(key, self.rnn_window, self.input_n_samples[key], val['Discrete'])
                else: 
                    (self.inputs_for_model[key],self.inputs[key]) = self.input(key, self.input_n_samples[key])
                    (self.inputs_for_rnn_model[key],self.rnn_inputs[key]) = self.inputRNN(key, self.rnn_window, self.input_n_samples[key])

                self.model_used['Inputs'][key]=val

        self.MP(print,"max_n_samples:"+str(self.max_n_samples))
        self.MP(pprint,{"input_n_samples":self.input_n_samples})
        self.MP(pprint,{"input_ns_backward":self.input_ns_backward})
        self.MP(pprint,{"input_ns_forward":self.input_ns_forward})
        self.MP(pprint,{"inputs_for_model":self.inputs_for_model})
        self.MP(pprint,{"inputs":self.inputs})
        self.MP(pprint,{"inputs_for_rnn_model":self.inputs_for_rnn_model})
        self.MP(pprint,{"rnn_inputs":self.rnn_inputs})

        # Building the elements of the network
        for outel in self.model_def['Outputs']:
            relel = relations.get(outel)
            self.output_relation[outel] = []
            # Loop all the output relation and for each relation go up the tree 
            for reltype, relvalue in relel.items():
                relation = getattr(self,reltype)
                if relation:
                    # Create the element of the relation and add to the list of the output
                    self.outputs[outel] = self.__createElem(relation,relvalue,outel)
                else:
                    print("Relation not defined") 

            # Add the relation and the output elements to the model_used
            if outel not in self.model_used['Outputs']:
                self.model_used['Outputs'][outel]=self.model_def['Outputs'][outel]   
                self.model_used['Relations'][outel]=relel    

        self.MP(pprint,{"relations":self.relations}) 
        self.MP(pprint,{"outputs":self.outputs})   
        self.MP(pprint,{"output_relation":self.output_relation})    
        self.MP(pprint,self.model_used)  

        # Building the model
        self.model = tensorflow.keras.models.Model(inputs = [val for key,val in self.inputs_for_model.items()], outputs=[val for key,val in self.outputs.items()])
        print(self.model.summary())
    
    #
    # Recursive method that terminates all inputs that result in a specific relationship for an output
    # During the procedure the dimension of the time window for each input is define
    #
    def __setInput(self, relvalue, outel):
        for el in relvalue:
            if type(el) is tuple:
                if el[0] in self.model_def['Inputs']:
                    time_window = self.input_tw_backward.get(el[0])
                    if time_window is not None:
                        if type(el[1]) is tuple:
                            if self.input_tw_backward[el[0]] < el[1][0]:
                                self.input_tw_backward[el[0]] = el[1][0]
                            if self.input_tw_forward[el[0]] > el[1][1]:
                                self.input_tw_forward[el[0]] = el[1][1]                            
                        else:
                            if self.input_tw_backward[el[0]] < el[1]:
                                self.input_tw_backward[el[0]] = el[1]
                    else:
                        if type(el[1]) is tuple:
                            self.input_tw_backward[el[0]] = el[1][0]
                            self.input_tw_forward[el[0]] = el[1][1]                       
                        else:
                            self.input_tw_backward[el[0]] = el[1]
                            self.input_tw_forward[el[0]] = 0  
                else:
                    raise Exception("A window on internal signal is not supported!")
            else: 
                if el in self.model_def['Inputs']:
                    time_window = self.input_tw_backward.get(el)
                    if time_window is None:
                        self.input_tw_backward[el] = self.model_def['SampleTime']
                        self.input_tw_forward[el] = 0  
                else:
                    relel = self.model_def['Relations'].get((outel,el))
                    if relel is None:
                        relel = self.model_def['Relations'].get(el)
                        if relel is None:
                            raise Exception("Graph is not completed!")
                    for reltype, relvalue in relel.items():
                        self.__setInput(relvalue, outel)

    #
    # Recursive method to create all the relations for an output
    # relation is a callback to the relation action, it is the type of relation (linear, relu, ect..)
    # relvalue is the input of the relation 
    # outel is the output signal name
    #
    def __createElem(self, relation, relvalue, outel):
        el_idx = neu4mes.NeuObj().count
        if len(relvalue) == 1:
            # The relvalue is a single element
            el = relvalue[0]
            if type(el) is tuple:
                # The relvalue is a part of an input
                if el[0] in self.model_def['Inputs']:
                    if type(el[1]) is tuple:
                        n_backward = int(el[1][0]/self.model_def['SampleTime'])
                        n_forward = int(el[1][0]/self.model_def['SampleTime'])
                        if (el[0],n_backward,n_forward) not in self.inputs:
                            self.inputs[(el[0],n_backward,n_forward)] = self.part(el[0],self.inputs[el[0]],n_backward,n_forward)
                            input_for_relation = self.inputs[(el[0],n_backward,n_forward)]
                    else:
                        n_backward = int(el[1]/self.model_def['SampleTime'])
                        if (el[0],n_backward) not in self.inputs:
                            self.inputs[(el[0],n_backward)] = self.part(el[0],self.inputs[el[0]],n_backward)
                            input_for_relation = self.inputs[(el[0],n_backward)]
                    return relation(outel[:2]+'_'+el[0][:2]+str(el_idx), input_for_relation)                
                else:
                    print("Tuple is defined only for Input")   
            else:
                if el in self.model_def['Inputs']:
                    # The relvalue is a part of an input
                    if (el,1) not in self.inputs:
                        self.inputs[(el,1)] = self.part(el,self.inputs[el],1)
                    return relation(outel[:2]+'_'+el[:2]+str(el_idx), self.inputs[(el,1)])
                else:
                    # The relvalue is a relation
                    input = self.__createRelation(relation, el, outel)
                    return relation(outel[:2]+'_'+el[:2]+str(el_idx), input)
        else:
            # The relvalue is a vector (e.g. Sum relation use vectors)
            # Create a list of all the inputs and then it calls the relation
            inputs = []
            name = outel[:2]
            for idx, el in enumerate(relvalue):
                if type(el) is tuple:
                    # The relvalue[i] is a part of an input
                    if el[0] in self.model_def['Inputs']:
                        if type(el[1]) is tuple:
                            n_backward = int(el[1][0]/self.model_def['SampleTime'])
                            n_forward = int(el[1][0]/self.model_def['SampleTime'])
                            if (el[0],n_backward,n_forward) not in self.inputs:
                                self.inputs[(el[0],n_backward,n_forward)] = self.part(el[0],self.inputs[el[0]],n_backward,n_forward)
                                input_for_relation = self.inputs[(el[0],n_backward,n_forward)]
                        else:
                            n_backward = int(el[1]/self.model_def['SampleTime'])
                            if (el[0],n_backward) not in self.inputs:
                                self.inputs[(el[0],n_backward)] = self.part(el[0],self.inputs[el[0]],n_backward)
                                input_for_relation = self.inputs[(el[0],n_backward)]

                        inputs.append(input_for_relation)
                        name = name +'_'+el[0][:2]
                    else:
                        print("Tuple is defined only for Input")  
                else:
                    if el in self.model_def['Inputs']:
                        # The relvalue[i] is a part of an input
                        if (el[0],1) not in self.inputs:
                            self.inputs[(el[0],1)] = self.part(el,self.inputs[el],1)
                        inputs.append(self.inputs[(el[0],1)])
                        name = name +'_'+ el[:2]
                    else:
                        # The relvalue[i] is a relation
                        inputs.append(self.__createRelation(relation, el, outel))
                        name = name +'_'+ el[:2]

            # Call the realtion with all the defined inputs
            return relation(name+str(el_idx), inputs)

    #
    # Recursive method to create all the elements of a relation
    # relation is a callback to the relation action, it is the type of relation (linear, relu, ect..)
    # el is the relation name 
    # outel is the output signal name
    #
    def __createRelation(self,relation,el,outel):
        relel = self.model_def['Relations'].get((outel,el))
        if relel is None:
            relel = self.model_def['Relations'].get(el)
            if relel is None:
                raise Exception("Graph is not completed!")
        for new_reltype, new_relvalue in relel.items():
            new_relation = getattr(self,new_reltype)
            if new_relation:
                if el not in self.model_used['Relations']:
                    self.relations[el] = self.__createElem(new_relation, new_relvalue, outel)
                    self.model_used['Relations'][el]=relel
                return self.relations[el]
            else:
                print("Relation not defined")   

    """
    Loading of the data set files and generate the structure for the training considering the structure of the input and the output 
    :param format: it is a list of the variable in the csv. All the input keys must be inside this list.
    :param folder: folder of the dataset. Each file is a simulation.
    :param sample_time: number of lines to be skipped (header lines)
    :param delimiters: it is a list of the symbols used between the element of the file
    """
    def loadData(self, format, folder = './data', skiplines = 0, delimiters=['\t',';',',']):
        path, dirs, files = next(os.walk(folder))
        file_count = len(files)
        
        # Get the list of the output keys
        for idx,key in enumerate(self.output_relation.keys()):
            self.output_keys.append(key)
            elem_key = key.split('__')
            if len(elem_key) > 1 and elem_key[1]== '-z1':
                self.output_keys[idx] = elem_key[0]
            else:
                raise("Operation not implemeted yet!")

        # Create a vector of all the signals in the file + output_relation keys 
        for key in format+list(self.output_relation.keys()):
            self.inout_data_time_window[key] = []

        # RNN network
        if self.rnn_window:
            # Create a vector of all the signals in the file + output_relation keys for rnn 
            for key in format+list(self.output_relation.keys()):
                self.inout_rnn_data_time_window[key] = []

        # Read each file
        for file in files:
            for data in format: 
                self.input_data[(file,data)] = []

            # Open the file and read lines
            all_lines = open(folder+file, 'r')
            lines = all_lines.readlines()[skiplines:] # skip first lines to avoid NaNs

            # Append the data to the input_data dict
            for line in range(0, len(lines)):
                delimiter_string = '|'.join(delimiters)
                splitline = re.split(delimiter_string,lines[line].rstrip("\n"))
                for idx, key in enumerate(format):
                    try:
                        self.input_data[(file,key)].append(float(splitline[idx]))
                    except ValueError:
                        self.input_data[(file,key)].append(splitline[idx])  

            # Add one sample if input look at least one forward
            add_sample_forward = 0
            if self.max_samples_forward > 0:
                add_sample_forward = 1

            # Create inout_data_time_window dict 
            # it is a dict of signals. Each signal is a list of vector the dimensions of the vector are (tokens, input_n_samples[key]) 
            if 'time' in format:
                for i in range(0, len(self.input_data[(file,'time')])-self.max_n_samples+add_sample_forward):
                    self.inout_data_time_window['time'].append(self.input_data[(file,'time')][i+self.max_n_samples-1-self.max_samples_forward])

            for key in self.input_n_samples.keys():
                for i in range(0, len(self.input_data[(file,key)])-self.max_n_samples+add_sample_forward):
                    aux_ind = i+self.max_n_samples+self.input_ns_forward[key]-self.max_samples_forward
                    if self.input_n_samples[key] == 1:
                        self.inout_data_time_window[key].append(self.input_data[(file,key)][aux_ind-1])
                    else:
                        self.inout_data_time_window[key].append(self.input_data[(file,key)][aux_ind-self.input_n_samples[key]:aux_ind])
        
            for key in self.output_relation.keys():
                used_key = key
                elem_key = key.split('__')
                if len(elem_key) > 1 and elem_key[1]== '-z1':
                    used_key = elem_key[0]
                for i in range(0, len(self.input_data[(file,used_key)])-self.max_n_samples+add_sample_forward):
                    self.inout_data_time_window[key].append(self.input_data[(file,used_key)][i+self.max_n_samples-self.max_samples_forward])
            
            self.idx_of_rows.append(len(self.inout_data_time_window[list(self.input_n_samples.keys())[0]]))

            # RNN network
            # Create inout_rnn_data_time_window dict 
            # it is a dict of signals. Each signal is a list of vector the dimensions of the vector are (tokens, input_n_samples[key])
            # ADD TEST ON THIS PART
            if self.rnn_window:
                if 'time' in format:
                    for i in range(0, len(self.input_data[(file,'time')])-self.max_n_samples-self.rnn_window):
                        inout_rnn = []
                        for j in range(i, i+self.rnn_window):
                            inout_rnn.append(self.input_data[(file,'time')][j:j+self.max_n_samples])

                        self.inout_rnn_data_time_window['time'].append(inout_rnn)
                
                for key in self.input_n_samples.keys():
                    for i in range(0, len(self.input_data[(file,key)])-self.max_n_samples-self.rnn_window):
                        inout_rnn = []
                        for j in range(i, i+self.rnn_window):
                            if self.input_n_samples[key] == 1:
                                inout_rnn.append(self.input_data[(file,key)][j+self.max_n_samples-1])
                            else:
                                inout_rnn.append(self.input_data[(file,key)][j+self.max_n_samples-self.input_n_samples[key]:j+self.max_n_samples])
                        self.inout_rnn_data_time_window[key].append(inout_rnn)

                for idx,key in enumerate(self.output_relation.keys()):
                    for i in range(0, len(self.input_data[(file,self.output_keys[idx])])-self.max_n_samples-self.rnn_window):
                        inout_rnn = []
                        for j in range(i, i+self.rnn_window):
                            inout_rnn.append(self.input_data[(file,self.output_keys[idx])][j+self.max_n_samples])
                        self.inout_rnn_data_time_window[key].append(inout_rnn)

        # Build the asarray for numpy
        for key,data in self.inout_data_time_window.items():
            self.inout_asarray[key]  = np.asarray(data)

        # RNN network
        # Build the asarray for numpy
        if self.rnn_window:
            for key,data in self.inout_rnn_data_time_window.items():
                self.inout_rnn_asarray[key]  = np.asarray(data)


    def trainModel(self, states = None, validation_percentage = 0, show_results = False):
        # Divide train and test samples
        num_of_sample = len(list(self.inout_asarray.values())[0])
        validation = round(validation_percentage*num_of_sample/100)
        self.train_idx  = num_of_sample-validation

        if self.train_idx < self.batch_size or validation < self.batch_size:
            batch = 1
        else:
            # Samples must be multiplier of batch
            self.train_idx = int(self.train_idx/self.batch_size) * self.batch_size
            validation  = num_of_sample-self.train_idx
            validation = int(validation/self.batch_size) * self.batch_size

        for key,data in self.inout_asarray.items():
            if len(data.shape) == 1:
                self.inout_4train[key] = data[0:self.train_idx]
                self.inout_4validation[key]  = data[self.train_idx:self.train_idx+validation]
            else:
                self.inout_4train[key] = data[0:self.train_idx,:]
                self.inout_4validation[key]  = data[self.train_idx:self.train_idx+validation,:]                

        #print('Samples: ' + str(train+validation) + '/' + str(num_of_sample) + ' (' + str(train) + ' train + ' + str(validation) + ' validation)')
        #print('Batch: ' + str(self.batch_size))
        
        # Configure model for training
        self.opt = optimizers.Adam(learning_rate = self.learning_rate) #optimizers.Adam(learning_rate=l_rate) #optimizers.RMSprop(learning_rate=lrate, rho=0.4)
        self.model.compile(optimizer = self.opt, loss = 'mean_squared_error', metrics=[rmse])

        # print(self.inout_4train['x'].shape)
        # Train model
        #print('[Fitting]')
        #print(len([self.inout_4train[key] for key in self.model_def['Outputs'].keys()]))

        self.fit = self.model.fit([self.inout_4train[key] for key in self.model_def['Inputs'].keys()],
                                  [self.inout_4train[key] for key in self.model_def['Outputs'].keys()],
                                  epochs = self.num_of_epochs, batch_size = self.batch_size, verbose=1)
        
        self.net_weights = self.model.get_weights()

        # Rmse training
        rmse_generator = (rmse_i for i, rmse_i in enumerate(self.fit.history) if i in range(len(self.fit.history)-len(self.model.outputs), len(self.fit.history)))
        rmse_train = []
        for rmse_i in rmse_generator:
            rmse_train.append(self.fit.history[rmse_i][-1])
        rmse_train = np.array(rmse_train)

        # Prediction on validation samples
        #print('[Prediction]')
        self.prediction = self.model([self.inout_4validation[key] for key in self.model_def['Inputs'].keys()], training=False) #, callbacks=[NoiseCallback()])
        self.prediction = np.array(self.prediction)
        if len(self.prediction.shape) == 2:
            self.prediction = np.expand_dims(self.prediction, axis=0)

        key_out = list(self.model_def['Outputs'].keys())

        se         = np.empty([len(key_out),validation])
        mse        = np.empty([len(key_out),])
        rmse_valid = np.empty([len(key_out),])
        fvu        = np.empty([len(key_out),])
        for i in range(len(key_out)):
            # Square error validation
            se[i] = np.square(self.prediction[i].flatten() - self.inout_4validation[key_out[i]].flatten())
            # Mean square error validation
            mse[i] = np.mean(np.square(self.prediction[i].flatten() - self.inout_4validation[key_out[i]].flatten()))
            # Rmse validation
            rmse_valid[i] = np.sqrt(np.mean(np.square(self.prediction[i].flatten() - self.inout_4validation[key_out[i]].flatten())))
            # Fraction of variance unexplained (FVU) validation
            fvu[i] = np.var(self.prediction[i].flatten() - self.inout_4validation[key_out[i]].flatten()) / np.var(self.inout_4validation[key_out[i]].flatten())
        self.max_se_idxs = np.argmax(se, axis=1)        

        # Akaike’s Information Criterion (AIC) validation
        aic = validation * np.log(mse) + 2 * self.model.count_params()

        if show_results:
            # Plot
            self.fig, self.ax = plt.subplots(2*self.prediction.shape[0], 2,
                                             gridspec_kw={'width_ratios': [5, 1], 'height_ratios': [2, 1]*self.prediction.shape[0]})
            if len(self.ax.shape) == 1:
                self.ax = np.expand_dims(self.ax, axis=0)
            #plotsamples = self.prediction.shape[1]
            plotsamples = 200
            for i in range(0, self.prediction.shape[0]):
                # Zoomed validation data
                self.ax[2*i,0].plot(self.prediction[i].flatten(), linestyle='dashed')
                self.ax[2*i,0].plot(self.inout_4validation[key_out[i]].flatten())
                self.ax[2*i,0].grid('on')
                self.ax[2*i,0].set_xlim((self.max_se_idxs[i]-plotsamples, self.max_se_idxs[i]+plotsamples))
                self.ax[2*i,0].vlines(self.max_se_idxs[i], self.prediction[i][self.max_se_idxs[i]], self.inout_4validation[key_out[i]][self.max_se_idxs[i]],
                                      colors='r', linestyles='dashed')
                self.ax[2*i,0].legend(['predicted', 'validation'], prop={'family':'serif'})
                self.ax[2*i,0].set_title(key_out[i], family='serif')
                # Statitics
                self.ax[2*i,1].axis("off")
                self.ax[2*i,1].invert_yaxis()
                text = "Rmse training: {:3.6f}\nRmse validation: {:3.6f}\nAIC: {:3.6f}\nFVU: {:3.6f}".format(rmse_train[i], rmse_valid[i], aic[i], fvu[i])
                self.ax[2*i,1].text(0, 0, text, family='serif', verticalalignment='top')
                # Validation data
                self.ax[2*i+1,0].plot(self.prediction[i].flatten(), linestyle='dashed')
                self.ax[2*i+1,0].plot(self.inout_4validation[key_out[i]].flatten())
                self.ax[2*i+1,0].grid('on')
                self.ax[2*i+1,0].legend(['predicted', 'validation'], prop={'family':'serif'})
                self.ax[2*i+1,0].set_title(key_out[i], family='serif')
                # Empty
                self.ax[2*i+1,1].axis("off")
            self.fig.tight_layout()
            plt.show()
       
        if states is not None:
            self.trainRecurrentModel(states, validation_percentage, show_results)

    def trainRecurrentModel(self, states, validation_percentage = 0, show_results = False):
        state_keys = [key.signal_name for key in states if type(key) is neu4mes.Output]
        states_size = [self.input_n_samples[key] for key in self.model_def['Inputs'].keys() if key in state_keys]
        inputs_size = [self.input_n_samples[key] for key in self.model_def['Inputs'].keys() if key not in state_keys]
        state_vector = [1 if key in state_keys else 0 for key in self.model_def['Inputs'].keys()]
        rnn_cell = RNNCell(self.model, state_vector, states_size, inputs_size)
        # print(self.inputs_for_rnn_model)
        # print(state_keys)
        # print(states_size)
        # print(inputs_size)

        for key in state_keys:
            self.init_rnn_state[key] = tensorflow.keras.layers.Lambda(lambda x: x[:,0,:], name=key+'_init_state')(self.rnn_inputs[key])
        # print(self.init_rnn_state)

        initial_state_rnn = [self.init_rnn_state[key] for key in state_keys]
        # print(initial_state_rnn)
        inputs = [self.rnn_inputs[key] for key in self.model_def['Inputs'].keys() if key not in state_keys]
        # print(inputs)
        out_x_rnn = tensorflow.keras.layers.RNN(rnn_cell, return_sequences=True, stateful=False, unroll=True, name='rnn')(tuple(inputs), initial_state=initial_state_rnn)
        # splited_out = tensorflow.keras.layers.Lambda(lambda tensor: tf.split(tensor, num_or_size_splits=len(states_size), axis = 2))(out_x_rnn)
        splited_out = []
        for idx in range(len(self.model_def['Outputs'])):
            splited_out.append(out_x_rnn[:,:,idx])

        self.rnn_model = tensorflow.keras.models.Model(inputs=[val for key,val in self.inputs_for_rnn_model.items()], outputs=splited_out)
        print(self.rnn_model.summary())

        self.rnn_opt = optimizers.Adam(learning_rate = self.learning_rate_rnn)
        self.rnn_model.compile(optimizer = self.opt, loss = 'mean_squared_error', metrics=[rmse])
        self.rnn_model.set_weights(self.net_weights)

        # Divide train and test samples
        num_of_sample = len(list(self.inout_rnn_asarray.values())[0])
        validation = round(validation_percentage*num_of_sample/100)
        training  = num_of_sample-validation

        if training < self.batch_size or validation < self.batch_size:
            batch = 1
        else:
            # Samples must be multiplier of batch
            training = int(training/self.batch_size) * self.batch_size
            validation  = num_of_sample-training
            validation = int(validation/self.batch_size) * self.batch_size

        for key,data in self.inout_rnn_asarray.items():
            if len(data.shape) == 1:
                self.inout_rnn_4train[key] = data[0:training]
                self.inout_rnn_4validation[key]  = data[training:training+validation]
            else:
                self.inout_rnn_4train[key] = data[0:training,:]
                self.inout_rnn_4validation[key]  = data[training:training+validation,:]

        self.fit = self.rnn_model.fit([self.inout_rnn_4train[key] for key in self.model_def['Inputs'].keys()],
                                        [self.inout_rnn_4train[key] for key in self.model_def['Outputs'].keys()],
                                        epochs = self.num_of_epochs_rnn, batch_size = self.batch_size, verbose=1)

        # print(self.model.get_weights())             
        # print(self.rnn_model.get_weights())  
        first_idx_validation = next(x[0] for x in enumerate(self.idx_of_rows) if x[1] > self.train_idx)
        # print(first_idx_validation)
        if show_results:
            print('[Prediction]')
            self.output_rnn = []
            for i in range(first_idx_validation, len(self.idx_of_rows)-1):
                first_idx = self.idx_of_rows[i]-self.train_idx
                last_idx = self.idx_of_rows[i+1]-self.train_idx
                # print(last_idx-first_idx)
                # print(len(self.inout_4validation[key][first_idx:-1]))
                # print(first_idx)
                # print(last_idx)
                # print(self.idx_of_rows[i]-self.train_idx)
                input = [np.expand_dims(self.inout_4validation[key][first_idx],axis = 0) for key in self.model_def['Inputs'].keys()]
                # print([self.idx_of_rows[i]+1-self.train_idx,self.idx_of_rows[i+1]-self.train_idx])
                for t in range(first_idx+1,last_idx+1): 
                    # print(t)
                    self.rnn_prediction = np.array(self.model(input)) #, callbacks=[NoiseCallback()])
                    # print(self.rnn_prediction)
                    if len(self.rnn_prediction.shape) == 2:
                        self.rnn_prediction = np.expand_dims(self.rnn_prediction, axis=0)

                    if t != last_idx:
                        for idx, key in enumerate(self.model_def['Inputs'].keys()):
                            # print('------------------------------')
                            if key in state_keys:
                                idx_out = self.output_keys.index(key)
                                input[idx] = np.append(input[idx][:,1:],self.rnn_prediction[idx_out], axis = 1)
                                
                                # print(self.rnn_prediction[idx_out])
                                # print(input[idx])
                                
                            else:
                                input[idx] = np.expand_dims(self.inout_4validation[key][t], axis = 0)  
                        # print('------------------------------')                
                    
                    self.output_rnn.append(self.rnn_prediction)
            
            key = list(self.model_def['Outputs'].keys())
            # print(state_keys)
            # print(self.output_rnn)
            self.output_rnn = np.asarray(self.output_rnn)
            # print(self.output_rnn.shape)
            # print(len(self.output_keys))
            self.output_rnn = np.transpose(self.output_rnn.reshape((-1,len(self.output_keys))))
            # print(self.output_rnn.shape)
            # print(self.output_rnn[0].shape)
            #print(self.output_rnn)
            # print(self.inout_4validation[key[0]].shape)
            # print(self.inout_4validation[key[0]][self.idx_of_rows[first_idx_validation]-self.train_idx:].shape)
            # print(self.inout_4validation[key[0]][self.idx_of_rows[first_idx_validation]-self.train_idx-1:-1].flatten().shape)

            

            # Plot
            self.fig, self.ax = plt.subplots(2*len(key), 2, gridspec_kw={'width_ratios': [5, 1], 'height_ratios': [2, 1]*len(key)})
            if len(self.ax.shape) == 1:
                self.ax = np.expand_dims(self.ax, axis=0)
            #plotsamples = self.prediction.shape[1]
            plotsamples = 200
            for i in range(0, len(key)):
                # Zoomed validation data
                self.ax[2*i,0].plot(self.output_rnn[i].flatten(), linestyle='dashed')
                self.ax[2*i,0].plot(self.inout_4validation[key[i]][self.idx_of_rows[first_idx_validation]-self.train_idx:].flatten())
                self.ax[2*i,0].grid('on')
                # self.ax[2*i,0].set_xlim((self.max_se_idxs[i]-plotsamples, self.max_se_idxs[i]+plotsamples))
                # self.ax[2*i,0].vlines(self.max_se_idxs[i], self.output_rnn[i][self.max_se_idxs[i]], self.inout_4validation[key[i]][self.max_se_idxs[i]], colors='r', linestyles='dashed')
                self.ax[2*i,0].legend(['predicted', 'validation'], prop={'family':'serif'})
                self.ax[2*i,0].set_title(key[i], family='serif')
                # Statitics
                self.ax[2*i,1].axis("off")
                self.ax[2*i,1].invert_yaxis()
                # text = "Rmse: {:3.4f}".format(pred_rmse[i])
                # self.ax[2*i,1].text(0, 0, text, family='serif', verticalalignment='top')
                # Validation data
                self.ax[2*i+1,0].plot(self.output_rnn[i].flatten(), linestyle='dashed')
                self.ax[2*i+1,0].plot(self.inout_4validation[key[i]][self.idx_of_rows[first_idx_validation]-self.train_idx:-1].flatten())
                self.ax[2*i+1,0].grid('on')
                self.ax[2*i+1,0].legend(['predicted', 'validation'], prop={'family':'serif'})
                self.ax[2*i+1,0].set_title(key[i], family='serif')
                # Empty
                self.ax[2*i+1,1].axis("off")
            self.fig.tight_layout()
            plt.show()


    #def __updateSlider(val, i):
    #    pos = self.spos.val
    #    plotsamples = self.prediction.shape[1]
    #    for i in range(self.prediction.shape[0]):
    #        self.ax[i].axis([pos-plotsamples/10,pos+plotsamples/10,1])
    #    self.fig.canvas.draw_idle()

    # def controlDefinition(control):
    #     pass

    # def neuralizeControl():
    #     pass

    # def trainControl(data):
    #     pass

    # def exportModel(params):
    #     pass

    # def exportControl(params):
    #     pass



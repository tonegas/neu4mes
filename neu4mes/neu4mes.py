from pprint import pprint 
import tensorflow.keras.layers
import tensorflow.keras.models
import tensorflow as tf
#from tensorflow.python.ops.gen_math_ops import rint


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
import re
import os
import numpy as np

import random
import string

import neu4mes 
from neu4mes.visualizer import StandardVisualizer

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
    def __init__(self, model_def = 0, verbose = False, visualizer = StandardVisualizer()):
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
        self.rnn_inputs = {}                # RNN element - processed network inputs
        self.inputs_for_model = {}          # NN element - clean network inputs
        self.rnn_inputs_for_model = {}      # RNN element - clean network inputs
        self.rnn_init_state = {}            # RNN element - for the states of RNN
        self.relations = {}                 # NN element - operations 
        self.outputs = {}                   # NN element - clean network outputs

        self.output_relation = {}           # dict with the outputs  
        self.output_keys = []               # clear output signal keys (without delay info string __-z1)
        
        # Models of the framework
        self.model = None                   # NN model - Keras model
        self.rnn_model = None               # RNN model - Keras model
        self.net_weights = None             # NN weights

        # Optimizer parameters 
        self.opt = None                     # NN model - Keras Optimizer
        self.rnn_opt = None                 # RNN model - Keras Optimizer

        # Dataset characteristics
        self.input_data = {}                # dict with data divided by file and symbol key: input_data[(file,key)]
        self.inout_data_time_window = {}    # dict with data divided by signal ready for network input
        self.rnn_inout_data_time_window = {}# dict with data divided by signal ready for RNN network input
        self.inout_asarray = {}             # dict for network input in asarray format
        self.rnn_inout_asarray = {}         # dict for RNN network input in asarray format
        self.num_of_samples = None          # number of rows of the file
        self.num_of_training_sample = 0     # number of rows for training

        self.idx_of_rows = [0]              # Index identifying each file start 
        self.first_idx_test = 0             # Index identifying the first test 

        # Training params
        self.batch_size = 128                               # batch size
        self.learning_rate = 0.0005                         # learning rate for NN
        self.num_of_epochs = 300                            # number of epochs
        self.rnn_batch_size = self.batch_size               # batch size for RNN
        self.rnn_window = None                              # window of the RNN
        self.rnn_learning_rate = self.learning_rate/10000   # learning rate for RNN
        self.rnn_num_of_epochs = 50                         # number of epochs for RNN
        
        # Training dataset
        self.inout_4train = {}                              # Dataset for training NN
        self.inout_4test = {}                               # Dataset for test NN
        self.rnn_inout_4train = {}                          # Dataset for training RNN
        self.rnn_inout_4test = {}                           # Dataset for test RNN

        # Training performance
        self.performance = {}                               # Dict with performance parameters for NN

        # Visualizer
        self.visualizer = visualizer                        # Class for visualizing data

    def MP(self,fun,arg):
        if self.verbose:
            fun(arg)

    """
    Add a new model to be trained: 
    :param model_def: can be a json model definition or a Output object 
    """
    def addModel(self, model_def):
        if type(model_def) is neu4mes.Output:
            self.model_def = neu4mes.merge(self.model_def, model_def.json)
        elif type(model_def) is dict:
            self.model_def = neu4mes.merge(self.model_def, model_def) 
        self.MP(pprint,self.model_def)

    """
    Definition of the network structure through the dependency graph and sampling time. 
    If a prediction window is also specified, it means that a recurrent network is also to be defined.
    :param sample_time: the variable defines the rate of the network based on the training set
    :param prediction_window: the variable defines the prediction horizon in the future
    """
    def neuralizeModel(self, sample_time = 0, prediction_window = None):
        # Prediction window is used for the recurrent network
        if prediction_window is not None:
            self.rnn_window = round(prediction_window/sample_time)
            assert prediction_window >= sample_time

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
                    (self.rnn_inputs_for_model[key],self.rnn_inputs[key]) = self.discreteInputRNN(key, self.rnn_window, self.input_n_samples[key], val['Discrete'])
                else: 
                    (self.inputs_for_model[key],self.inputs[key]) = self.input(key, self.input_n_samples[key])
                    (self.rnn_inputs_for_model[key],self.rnn_inputs[key]) = self.inputRNN(key, self.rnn_window, self.input_n_samples[key])

                self.model_used['Inputs'][key]=val

        self.MP(print,"max_n_samples:"+str(self.max_n_samples))
        self.MP(pprint,{"input_n_samples":self.input_n_samples})
        self.MP(pprint,{"input_ns_backward":self.input_ns_backward})
        self.MP(pprint,{"input_ns_forward":self.input_ns_forward})
        self.MP(pprint,{"inputs_for_model":self.inputs_for_model})
        self.MP(pprint,{"inputs":self.inputs})
        self.MP(pprint,{"rnn_inputs_for_model":self.rnn_inputs_for_model})
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

        self.MP(pprint,{"inputs":self.inputs})
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
                    if len(el) == 3:
                        # If three element I get the offset
                        offset = int(el[2]/self.model_def['SampleTime'])
                        if type(el[1]) is tuple:
                            # The time window is backward and forward
                            n_backward = int(el[1][0]/self.model_def['SampleTime'])
                            n_forward = -int(el[1][1]/self.model_def['SampleTime'])
                            if (el[0],(n_backward,n_forward),offset) not in self.inputs:
                                self.inputs[(el[0],(n_backward,n_forward),offset)] = self.part(el[0],self.inputs[el[0]],n_backward,n_forward,offset)
                                input_for_relation = self.inputs[(el[0],(n_backward,n_forward),offset)]
                        else:
                            n_backward = int(el[1]/self.model_def['SampleTime'])
                            if (el[0],n_backward,offset) not in self.inputs:
                                self.inputs[(el[0],n_backward,offset)] = self.part(el[0],self.inputs[el[0]],n_backward,0,offset)
                                input_for_relation = self.inputs[(el[0],n_backward,offset)]               
                    elif len(el) == 2:
                        if type(el[1]) is tuple:
                            # The time window is backward and forward
                            n_backward = int(el[1][0]/self.model_def['SampleTime'])
                            n_forward = -int(el[1][1]/self.model_def['SampleTime'])
                            if (el[0],(n_backward,n_forward)) not in self.inputs:
                                self.inputs[(el[0],(n_backward,n_forward))] = self.part(el[0],self.inputs[el[0]],n_backward,n_forward)
                                input_for_relation = self.inputs[(el[0],(n_backward,n_forward))]
                        else:
                            n_backward = int(el[1]/self.model_def['SampleTime'])
                            if (el[0],n_backward) not in self.inputs:
                                self.inputs[(el[0],n_backward)] = self.part(el[0],self.inputs[el[0]],n_backward)
                                input_for_relation = self.inputs[(el[0],n_backward)]                      
                    else:
                        raise Exception("This tuple has only one element: "+str(el))

                    return relation(outel[:2]+'_'+el[0][:2]+str(el_idx), input_for_relation)                
                else:
                    raise Exception("Tuple is defined only for Input")   
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
                        if len(el) == 3:
                            # If three element I get the offset
                            offset = int(el[2]/self.model_def['SampleTime'])
                            if type(el[1]) is tuple:
                                # The time window is backward and forward
                                n_backward = int(el[1][0]/self.model_def['SampleTime'])
                                n_forward = -int(el[1][1]/self.model_def['SampleTime'])
                                if (el[0],(n_backward,n_forward),offset) not in self.inputs:
                                    self.inputs[(el[0],(n_backward,n_forward),offset)] = self.part(el[0],self.inputs[el[0]],n_backward,n_forward,offset)
                                    input_for_relation = self.inputs[(el[0],(n_backward,n_forward),offset)]
                            else:
                                n_backward = int(el[1]/self.model_def['SampleTime'])
                                if (el[0],n_backward,offset) not in self.inputs:
                                    self.inputs[(el[0],n_backward,offset)] = self.part(el[0],self.inputs[el[0]],n_backward,0,offset)
                                    input_for_relation = self.inputs[(el[0],n_backward,offset)]                          
                        elif len(el) == 2:
                            if type(el[1]) is tuple:
                                # The time window is backward and forward
                                n_backward = int(el[1][0]/self.model_def['SampleTime'])
                                n_forward = -int(el[1][1]/self.model_def['SampleTime'])
                                if (el[0],(n_backward,n_forward)) not in self.inputs:
                                    self.inputs[(el[0],(n_backward,n_forward))] = self.part(el[0],self.inputs[el[0]],n_backward,n_forward)
                                    input_for_relation = self.inputs[(el[0],(n_backward,n_forward))]
                            else:
                                n_backward = int(el[1]/self.model_def['SampleTime'])
                                if (el[0],n_backward) not in self.inputs:
                                    self.inputs[(el[0],n_backward)] = self.part(el[0],self.inputs[el[0]],n_backward)
                                    input_for_relation = self.inputs[(el[0],n_backward)]                        
                        else:
                            raise Exception("This tuple has only one element: "+str(el))

                        inputs.append(input_for_relation)
                        name = name +'_'+el[0][:2]
                    else:
                        raise Exception("Tuple is defined only for Input")  
                else:
                    if el in self.model_def['Inputs']:
                        # The relvalue[i] is a part of an input
                        if (el,1) not in self.inputs:
                            self.inputs[(el,1)] = self.part(el,self.inputs[el],1)
                        inputs.append(self.inputs[(el,1)])
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

        self.MP(print, "Total number of files: {}".format(file_count))
        
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
                self.rnn_inout_data_time_window[key] = []

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
            
            # Index identifying each file start 
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

                        self.rnn_inout_data_time_window['time'].append(inout_rnn)
                
                for key in self.input_n_samples.keys():
                    for i in range(0, len(self.input_data[(file,key)])-self.max_n_samples-self.rnn_window):
                        inout_rnn = []
                        for j in range(i, i+self.rnn_window):
                            if self.input_n_samples[key] == 1:
                                inout_rnn.append(self.input_data[(file,key)][j+self.max_n_samples-1])
                            else:
                                inout_rnn.append(self.input_data[(file,key)][j+self.max_n_samples-self.input_n_samples[key]:j+self.max_n_samples])
                        self.rnn_inout_data_time_window[key].append(inout_rnn)

                for idx,key in enumerate(self.output_relation.keys()):
                    for i in range(0, len(self.input_data[(file,self.output_keys[idx])])-self.max_n_samples-self.rnn_window):
                        inout_rnn = []
                        for j in range(i, i+self.rnn_window):
                            inout_rnn.append(self.input_data[(file,self.output_keys[idx])][j+self.max_n_samples])
                        self.rnn_inout_data_time_window[key].append(inout_rnn)

        # Build the asarray for numpy
        for key,data in self.inout_data_time_window.items():
            self.inout_asarray[key]  = np.asarray(data)
        
        # Definition of number of samples
        if self.num_of_samples == None: 
            keys = list(self.output_relation.keys())
            self.num_of_samples = len(self.inout_asarray[keys[0]])

        # RNN network
        # Build the asarray for numpy and definition of the number of samples
        if self.rnn_window:
            for key,data in self.rnn_inout_data_time_window.items():
                self.rnn_inout_asarray[key]  = np.asarray(data)
            keys = list(self.output_relation.keys())
            self.rnn_num_of_samples = len(self.rnn_inout_asarray[keys[0]])

    #
    # Function for cheking the state is an Output
    #
    def __checkStates(self, states):
        state_keys = None
        if states:
            state_keys = [key.signal_name if type(key) is neu4mes.Output else Exception("A state is not an Output object") for key in states ]
        return state_keys

    #
    # Function that get specific parameters for training 
    #
    def __getTrainParams(self, training_params):
        if bool(training_params):
            self.batch_size = (training_params['batch_size'] if 'batch_size' in training_params else self.batch_size) 
            self.learning_rate = (training_params['learning_rate'] if 'learning_rate' in training_params else self.learning_rate) 
            self.num_of_epochs = (training_params['num_of_epochs'] if 'num_of_epochs' in training_params else self.num_of_epochs) 
            self.rnn_batch_size = (training_params['rnn_batch_size'] if 'rnn_batch_size' in training_params else self.rnn_batch_size) 
            self.rnn_learning_rate = (training_params['rnn_learning_rate'] if 'rnn_learning_rate' in training_params else self.rnn_learning_rate) 
            self.rnn_num_of_epochs = (training_params['rnn_num_of_epochs'] if 'rnn_num_of_epochs' in training_params else self.rnn_num_of_epochs) 
    
    """
    Analysis of the results 
    """
    def resultAnalysis(self):
        # Rmse training
        rmse_generator = (rmse_i for i, rmse_i in enumerate(self.fit.history) if i in range(len(self.fit.history)-len(self.model.outputs), len(self.fit.history)))
        self.performance['rmse_train'] = []
        for rmse_i in rmse_generator:
            self.performance['rmse_train'].append(self.fit.history[rmse_i][-1])
        self.performance['rmse_train'] = np.array(self.performance['rmse_train'])

        # Prediction on test samples
        prediction = self.model([self.inout_4test[key] for key in self.model_def['Inputs'].keys()], training=False) #, callbacks=[NoiseCallback()])
        self.prediction = np.array(prediction)
        if len(self.prediction.shape) == 2:
            self.prediction = np.expand_dims(self.prediction, axis=0)

        # List of keys
        output_keys = list(self.model_def['Outputs'].keys())

        # Performance parameters
        self.performance['se'] = np.empty([len(output_keys),self.num_of_test_sample])
        self.performance['mse'] = np.empty([len(output_keys),])
        self.performance['rmse_test'] = np.empty([len(output_keys),])
        self.performance['fvu'] = np.empty([len(output_keys),])
        for i in range(len(output_keys)):
            # Square error test
            self.performance['se'][i] = np.square(self.prediction[i].flatten() - self.inout_4test[output_keys[i]].flatten())
            # Mean square error test
            self.performance['mse'][i] = np.mean(np.square(self.prediction[i].flatten() - self.inout_4test[output_keys[i]].flatten()))
            # Rmse test
            self.performance['rmse_test'][i] = np.sqrt(np.mean(np.square(self.prediction[i].flatten() - self.inout_4test[output_keys[i]].flatten())))
            # Fraction of variance unexplained (FVU) test
            self.performance['fvu'][i] = np.var(self.prediction[i].flatten() - self.inout_4test[output_keys[i]].flatten()) / np.var(self.inout_4test[output_keys[i]].flatten())
        
        # Index of worst results
        self.performance['max_se_idxs'] = np.argmax(self.performance['se'], axis=1)        

        # Akaike’s Information Criterion (AIC) test
        self.performance['aic'] = self.num_of_test_sample * np.log(self.performance['mse']) + 2 * self.model.count_params()

        self.visualizer.showResults(self, output_keys, performance = self.performance)

    """
    Training of the model. 
    :param states: it is a list of a states, the state must be an Output object
    :param training_params: dict that contains the parameters of training (batch_size, learning rate, etc..)
    :param test_percentage: numeric value from 0 to 100, it is the part of the dataset used for validate the performance of the network
    :param show_results: it is a boolean for enable the plot of the results
    """
    def trainModel(self, states = None, training_params = {}, test_percentage = 0,  show_results = False):
        # Check input
        state_keys = self.__checkStates(states)
        self.__getTrainParams(training_params)

        # Divide train and test samples
        self.num_of_test_sample = round(test_percentage*self.num_of_samples/100)
        self.num_of_training_sample  = self.num_of_samples-self.num_of_test_sample

        # Definition of the batch size with respect of the test dimensions
        if self.num_of_training_sample < self.batch_size or self.num_of_test_sample < self.batch_size:
            self.batch_size = 1
        else:
            # Samples must be multiplier of batch
            self.num_of_training_sample = int(self.num_of_training_sample/self.batch_size) * self.batch_size
            self.num_of_test_sample = int((self.num_of_samples-self.num_of_training_sample)/self.batch_size) * self.batch_size
        
        # Building the dataset structures training and test set
        for key,data in self.inout_asarray.items():
            if len(data.shape) == 1:
                self.inout_4train[key] = data[0:self.num_of_training_sample]
                self.inout_4test[key]  = data[self.num_of_training_sample:self.num_of_training_sample+self.num_of_test_sample]
            else:
                self.inout_4train[key] = data[0:self.num_of_training_sample,:]
                self.inout_4test[key]  = data[self.num_of_training_sample:self.num_of_training_sample+self.num_of_test_sample,:]                

        # Print information 
        self.MP(print, 'Samples: {}/{} (train size: {}, test size: {})'.format(self.num_of_training_sample+self.num_of_test_sample,self.num_of_samples,self.num_of_training_sample,self.num_of_test_sample)) 
        self.MP(print, 'Batch: {}'.format(self.batch_size))
        
        # Configure model for training
        self.opt = optimizers.Adam(learning_rate = self.learning_rate) #optimizers.Adam(learning_rate=l_rate) #optimizers.RMSprop(learning_rate=lrate, rho=0.4)
        self.model.compile(optimizer = self.opt, loss = 'mean_squared_error', metrics=[rmse])

        # Fitting of the network
        self.fit = self.model.fit([self.inout_4train[key] for key in self.model_def['Inputs'].keys()],
                                  [self.inout_4train[key] for key in self.model_def['Outputs'].keys()],
                                  epochs = self.num_of_epochs, batch_size = self.batch_size, verbose=1)
        self.net_weights = self.model.get_weights()

        # Show the analysis of the Result
        if show_results:
            self.resultAnalysis()
        
        # Recurrent training enabling
        if states is not None:
            self.trainRecurrentModel(states, test_percentage = test_percentage, show_results = show_results)

    """
    Analysis of the results for recurrent network
    :param states: it is a list of a states, the state must be an Output object
    """
    def resultRecurrentAnalysis(self, states):
        # Check input
        state_keys = self.__checkStates(states)

        # Get the first index from all the data
        self.first_idx_test = next(x[0] for x in enumerate(self.idx_of_rows) if x[1] > self.num_of_training_sample)

        # Define output for each window
        rnn_prediction = []
        for i in range(self.first_idx_test, len(self.idx_of_rows)-1):
            first_idx = self.idx_of_rows[i]-self.num_of_training_sample
            last_idx = self.idx_of_rows[i+1]-self.num_of_training_sample
            input = [np.expand_dims(self.inout_4test[key][first_idx],axis = 0) for key in self.model_def['Inputs'].keys()]

            for t in range(first_idx+1,last_idx+1): 
                rnn_output = np.array(self.model(input)) #, callbacks=[NoiseCallback()])
                if len(rnn_output.shape) == 2:
                    rnn_output = np.expand_dims(rnn_output, axis=0)

                if t != last_idx:
                    for idx, key in enumerate(self.model_def['Inputs'].keys()):
                        if key in state_keys:
                            idx_out = self.output_keys.index(key)
                            input[idx] = np.append(input[idx][:,1:],rnn_output[idx_out], axis = 1)
                        else:
                            input[idx] = np.expand_dims(self.inout_4test[key][t], axis = 0)               
                
                rnn_prediction.append(rnn_output)
        
        key = list(self.model_def['Outputs'].keys())
        rnn_prediction = np.asarray(rnn_prediction)

        # Final prediction for whole test set
        self.rnn_prediction = np.transpose(rnn_prediction.reshape((-1,len(self.output_keys))))

        # Analysis of the Result
        self.visualizer.showRecurrentResults(self, self.output_keys, performance = self.performance)

    """
    Reccurrent training of the model. 
    :param states: it is a list of a states, the state must be an Output object
    :param training_params: dict that contains the parameters of training (batch_size, learning rate, etc..)
    :param test_percentage: numeric value from 0 to 100, it is the part of the dataset used for validate the performance of the network
    :param show_results: it is a boolean for enable the plot of the results
    """
    def trainRecurrentModel(self, states, training_params = {}, test_percentage = 0, show_results = False):
        # Check input
        state_keys = self.__checkStates(states)
        self.__getTrainParams(training_params)

        # Definition of sizes
        states_size = [self.input_n_samples[key] for key in self.model_def['Inputs'].keys() if key in state_keys]
        inputs_size = [self.input_n_samples[key] for key in self.model_def['Inputs'].keys() if key not in state_keys]
        # This boolean vector representing if the input is also a state
        state_vector = [1 if key in state_keys else 0 for key in self.model_def['Inputs'].keys()]
        # Creation of the RNN cell
        rnn_cell = RNNCell(self.model, state_vector, states_size, inputs_size)
        
        self.MP(print, 'state_keys: {}'.format(state_keys))
        self.MP(print, 'inputs_size: {}'.format(inputs_size))
        self.MP(print, 'states_size: {}'.format(states_size))
        self.MP(print, 'state_vector: {}'.format(state_vector))

        # Inizialization of the initial state for the states
        for key in state_keys:
            self.rnn_init_state[key] = tensorflow.keras.layers.Lambda(lambda x: x[:,0,:], name=key+'_init_state')(self.rnn_inputs[key])
        rnn_initial_state = [self.rnn_init_state[key] for key in state_keys]

        # Definition of the input of a recurrent cell network
        inputs = [self.rnn_inputs[key] for key in self.model_def['Inputs'].keys() if key not in state_keys]

        # Creation of the RNN node
        rnn_out = tensorflow.keras.layers.RNN(rnn_cell, return_sequences=True, stateful=False, unroll=True, name='rnn')(tuple(inputs), initial_state=rnn_initial_state)

        self.MP(pprint,{"rnn_init_state":self.rnn_init_state})
        self.MP(pprint,{"rnn_initial_state":rnn_initial_state})        
        self.MP(pprint,{"inputs":inputs})

        # splited_out = tensorflow.keras.layers.Lambda(lambda tensor: tf.split(tensor, num_or_size_splits=len(states_size), axis = 2))(out_x_rnn)
        splited_out = []
        for idx in range(len(self.model_def['Outputs'])):
            splited_out.append(rnn_out[:,:,idx])

        self.MP(pprint,{"splited_out":splited_out})
        
        # Creation of the RNN model
        self.rnn_model = tensorflow.keras.models.Model(inputs=[val for key,val in self.rnn_inputs_for_model.items()], outputs=splited_out)
        print(self.rnn_model.summary())

        # Divide train and test samples
        self.rnn_num_of_test_sample = round(test_percentage*self.rnn_num_of_samples/100)
        self.rnn_num_of_training_sample = self.rnn_num_of_samples-self.rnn_num_of_test_sample

        # Definition of the batch size with respect of the test dimensions
        if self.rnn_num_of_training_sample < self.rnn_batch_size or self.rnn_num_of_test_sample < self.rnn_batch_size:
            self.rnn_batch_size = 1
        else:
            # Samples must be multiplier of batch
            self.rnn_num_of_training_sample = int(self.rnn_num_of_training_sample/self.rnn_batch_size) * self.rnn_batch_size
            self.rnn_num_of_test_sample = int((self.rnn_num_of_samples - self.rnn_num_of_training_sample)/self.rnn_batch_size) * self.rnn_batch_size

        # Building the dataset structures training and test set
        for key,data in self.rnn_inout_asarray.items():
            if len(data.shape) == 1:
                self.rnn_inout_4train[key] = data[0:self.rnn_num_of_training_sample]
                self.rnn_inout_4test[key]  = data[self.rnn_num_of_training_sample:self.rnn_num_of_training_sample+self.rnn_num_of_test_sample]
            else:
                self.rnn_inout_4train[key] = data[0:self.rnn_num_of_training_sample,:]
                self.rnn_inout_4test[key]  = data[self.rnn_num_of_training_sample:self.rnn_num_of_training_sample+self.rnn_num_of_test_sample,:]       

        # Print information 
        self.MP(print, 'RNN Samples: {}/{} (train size: {}, test size: {})'.format(self.rnn_num_of_training_sample+self.rnn_num_of_test_sample,self.rnn_num_of_samples,self.rnn_num_of_training_sample,self.rnn_num_of_test_sample)) 
        self.MP(print, 'RNN Batch: {}'.format(self.rnn_batch_size))
        
        # Configure rnn model for training
        self.rnn_opt = optimizers.Adam(learning_rate = self.rnn_learning_rate)
        self.rnn_model.compile(optimizer = self.rnn_opt, loss = 'mean_squared_error', metrics=[rmse])
        if self.net_weights:
            self.rnn_model.set_weights(self.net_weights)

        # Fitting of the network
        self.fit = self.rnn_model.fit([self.rnn_inout_4train[key] for key in self.model_def['Inputs'].keys()],
                                    [self.rnn_inout_4train[key] for key in self.model_def['Outputs'].keys()],
                                    epochs = self.rnn_num_of_epochs, batch_size = self.rnn_batch_size, verbose=1)

        # Show the analysis of the Result
        if show_results:
            self.resultRecurrentAnalysis(states)

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



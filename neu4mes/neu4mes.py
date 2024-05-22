import torch
from torch.utils.data import DataLoader

import numpy as np
import random
import os
from pprint import pprint
import re
from datetime import datetime
import matplotlib.pyplot as plt

from neu4mes.relation import NeuObj, merge
from neu4mes.visualizer import TextVisualizer
from neu4mes.dataset import Neu4MesDataset
from neu4mes.loss import CustomRMSE
from neu4mes.output import Output
from neu4mes.model import Model
from neu4mes.relation import Stream

class Neu4mes:
    def __init__(self, model_def = 0, verbose = False, visualizer = TextVisualizer()):
        # Set verbose print inside the class
        self.verbose = verbose

        # Inizialize the model definition
        # the model_def has all the relation defined for that model
        if type(model_def) is Output:
            self.model_def = model_def.json
        elif type(model_def) is dict:
            self.model_def = self.model_def
        else:
            self.model_def = NeuObj().json

        # Input, output, and model characteristics
        self.input_tw_backward = {}         # dimensions of the time window in the past for each input
        self.input_tw_forward = {}          # dimensions of the time window in the future for each input
        self.max_samples_backward = 0       # maxmimum number of samples backward for all the inputs
        self.max_samples_forward = 0        # maxmimum number of samples forward for all the inputs
        self.max_n_samples = 0              # maxmimum number of samples for all the inputs
        self.input_ns_backward = {}         # maxmimum number of samples backward for each the input
        self.input_ns_forward = {}          # maxmimum number of samples forward for each the input
        self.input_n_samples = {}           # maxmimum number of samples for each the input

        self.neuralized = False
        self.inputs = {}                    # NN element - processed network inputs
        self.rnn_inputs = {}                # RNN element - processed network inputs
        self.inputs_for_model = {}          # NN element - clean network inputs
        self.rnn_inputs_for_model = {}      # RNN element - clean network inputs
        self.rnn_init_state = {}            # RNN element - for the states of RNN
        self.relations = {}                 # NN element - operations
        self.relation_samples = {}          # N samples for each relation
        self.outputs = {}

        # List of element to be minimized
        self.minimize_list = []
        self.loss_fn = None                 # RNN model - Pytorch loss function

        self.output_relation = {}           # dict with the outputs
        self.output_keys = []               # clear output signal keys (without delay info string __-z1)

        # Models of the framework
        self.model = None                   # NN model - Pytorch model
        self.rnn_model = None               # RNN model - Pytorch model
        self.net_weights = None             # NN weights

        # Optimizer parameters
        self.optimizer = None                     # NN model - Pytorch optimizer

        # Dataset characteristics
        self.input_data = {}                # dict with data divided by file and symbol key: input_data[(file,key)]
        self.inout_data_time_window = {}    # dict with data divided by signal ready for network input
        self.rnn_inout_data_time_window = {}# dict with data divided by signal ready for RNN network input
        self.inout_asarray = {}             # dict for network input in asarray format
        self.rnn_inout_asarray = {}         # dict for RNN network input in asarray format
        self.num_of_samples = None          # number of rows of the file
        self.num_of_training_sample = 0     # number of rows for training
        self.n_samples_train = None
        self.n_samples_test = None

        self.idx_of_rows = [0]              # Index identifying each file start
        self.first_idx_test = 0             # Index identifying the first test

        # Dataloaders
        self.train_loader = None
        self.test_loader = None

        # Training params
        self.batch_size = 128                               # batch size
        self.learning_rate = 0.0005                         # learning rate for NN
        self.num_of_epochs = 100                             # number of epochs
        self.rnn_batch_size = self.batch_size               # batch size for RNN
        self.rnn_window = None                              # window of the RNN
        self.rnn_learning_rate = self.learning_rate/10000   # learning rate for RNN
        self.rnn_num_of_epochs = 50                         # number of epochs for RNN

        # Training dataset
        self.data_loaded = False
        self.inout_4train = {}                              # Dataset for training NN
        self.inout_4test = {}                               # Dataset for test NN
        self.rnn_inout_4train = {}                          # Dataset for training RNN
        self.rnn_inout_4test = {}                           # Dataset for test RNN

        # Training performance
        self.performance = {}                               # Dict with performance parameters for NN

        # Visualizer
        self.visualizer = visualizer                        # Class for visualizing data

    def __call__(self, inputs):
        print('[LOG] inputs: ', inputs)
        window_dim = min([len(val)-self.input_n_samples[key]+1 for key, val in inputs.items()])
        print('[LOG] window_dim: ', window_dim)
        assert window_dim > 0, 'Invalid Number of Inputs!'

        result_dict = {}
        for key in self.model_def['Outputs'].keys():
            result_dict[key] = []

        for i in range(window_dim):
            X = {}
            for key, val in inputs.items():
                X[key] = torch.from_numpy(np.array(val[i:i+self.input_n_samples[key]])).to(torch.float32)
                if X[key].ndim == 1: ## add the batch dimension
                    X[key] = X[key].unsqueeze(0)
            print(f'[LOG] sample {i}: {X}')
            result = self.model(X)
            print(f'[LOG] result {i}: {result}')
            for key in self.model_def['Outputs'].keys():
                result_dict[key].append(result[key].squeeze().detach().numpy().tolist())

        return result_dict
    
    def get_random_samples(self, window=1):
        if self.data_loaded:
            result_dict = {}
            for key in self.model_def['Inputs'].keys():
                result_dict[key] = []
            random_idx = random.randint(0, self.num_of_samples)
            for idx in range(window):
                for key ,data in self.inout_data_time_window.items():
                    if key in self.model_def['Inputs'].keys() and data is not None:
                        result_dict[key].append(data[random_idx+idx])
            return result_dict
        else:
            print('The Dataset must first be loaded using <loadData> function!')
            return {}

    def MP(self,fun,arg):
        if self.verbose:
            fun(arg)

    """
    Add a new model to be trained:
    :param model_def: can be a json model definition or a Output object
    """
    def addModel(self, model_def):
        if type(model_def) is Output:
            self.model_def = merge(self.model_def, model_def.json)
        elif type(model_def) is dict:
            self.model_def = merge(self.model_def, model_def)
        #self.MP(pprint,self.model_def)

    # TODO da modificare perchè adesso serve solo per far funzionare i test
    def minimizeError(self, variable_name, stream1, stream2, loss_function='mse'):
        # TODO Create un modello solo per la minimizzazione
        self.model_def = merge(self.model_def, stream1.json)
        self.model_def = merge(self.model_def, stream2.json)

        nameA = variable_name + '_' + (stream1.name[0] if type(stream1.name) is tuple else stream1.name)
        if type(stream1) is not Output:
            self.model_def['Outputs'][nameA] = stream1.name
        else:
            self.model_def['Outputs'][nameA] = self.model_def['Outputs'][stream1.name]

        nameB = variable_name + '_' + (stream2.name[0] if type(stream2.name) is tuple else stream2.name)
        if type(stream2) is not Output:
            self.model_def['Outputs'][nameB] = stream2.name
        else:
            self.model_def['Outputs'][nameB] = self.model_def['Outputs'][stream2.name]

        self.minimize_list.append((nameA, nameB, loss_function))

        ## Get output relations
        #out1 = self.model_def['Relations'][stream1.name]
        #out2 = self.model_def['Relations'][stream2.name]


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
        self.MP(pprint,self.model_def)

        for key, value in self.model_def['Inputs'].items():
            self.input_tw_backward[key] = abs(value['tw'][0])
            self.input_tw_forward[key] = value['tw'][1]
            if value['sw'] == [0,0] and value['tw'] == [0,0]:
                self.input_tw_backward[key] = sample_time
            self.input_ns_backward[key] = max(int(self.input_tw_backward[key] / sample_time),abs(value['sw'][0]))
            self.input_ns_forward[key] = max(int(self.input_tw_forward[key] / sample_time),abs(value['sw'][1]))
            self.input_n_samples[key] = self.input_ns_backward[key] + self.input_ns_forward[key]

        self.max_samples_backward = max(self.input_ns_backward.values())
        self.max_samples_forward = max(self.input_ns_forward.values())
        self.max_n_samples = self.max_samples_forward + self.max_samples_backward

        self.MP(pprint,{"window_backward": self.input_tw_backward, "window_forward":self.input_tw_forward})
        self.MP(pprint,{"samples_backward": self.input_ns_backward, "samples_forward":self.input_ns_forward})
        self.MP(pprint,{"input_n_samples": self.input_n_samples})
        self.MP(pprint,{"max_samples_backward": self.max_samples_backward, "max_samples_forward":self.max_samples_forward, "max_samples":self.max_n_samples})

        ## Get samples per relation
        for name, inputs in (self.model_def['Relations']|self.model_def['Outputs']).items():
            input_samples = {}
            inputs = inputs[1] if name in self.model_def['Relations'] else [inputs]
            for input_name in inputs:
                if type(input_name) is tuple: ## we have a window
                    window = 'tw' if 'tw' in input_name[1] else 'sw'
                    aux_sample_time = sample_time if 'tw' in input_name[1] else 1
                    if type(input_name[1][window]) is list: ## we have the forward and backward window
                        if input_name[0] in self.model_def['Inputs']:
                            backward = self.input_ns_backward[input_name[0]] - int(abs(input_name[1][window][0])/aux_sample_time)
                            forward = self.input_ns_backward[input_name[0]] + int(abs(input_name[1][window][1])/aux_sample_time)
                        else:
                            backward = int(abs(input_name[1][window][0])/aux_sample_time)
                            forward = int(abs(input_name[1][window][1])/aux_sample_time)

                        if 'offset' in input_name[1]: ## we have the offset
                            offset = int(abs(input_name[1][window][0])/aux_sample_time) + int(input_name[1]['offset'] / aux_sample_time)
                            input_samples[input_name[0]] = {'backward': backward, 'forward': forward, 'offset': offset}
                        else:
                            input_samples[input_name[0]] = {'backward':backward, 'forward': forward}

                else: ## we have no window
                    input_samples[input_name] = {'backward':0, 'forward':1}

            self.relation_samples[name] = input_samples

        self.MP(pprint,{"relation_samples": self.relation_samples})

        ## Build the network
        self.model = Model(self.model_def, self.relation_samples)
        self.MP(pprint,self.model)
        self.neuralized = True
    """
    Loading of the data set files and generate the structure for the training considering the structure of the input and the output
    :param format: it is a list of the variable in the csv. All the input keys must be inside this list.
    :param folder: folder of the dataset. Each file is a simulation.
    :param sample_time: number of lines to be skipped (header lines)
    :param delimiters: it is a list of the symbols used between the element of the file
    """
    def loadData(self, format, folder = './data', skiplines = 0, delimiters=['\t',';',',']):
        assert self.neuralized == True, "The network is not neuralized yet."
        path, dirs, files = next(os.walk(folder))
        file_count = len(files)

        self.MP(print, "Total number of files: {}".format(file_count))

        # Create a vector of all the signals in the file + output_relation keys
        output_keys = self.model_def['Outputs'].keys()
        for key in format+list(output_keys):
            self.inout_data_time_window[key] = []

        # Read each file
        for file in files:
            for data in format:
                self.input_data[(file,data)] = []

            # Open the file and read lines
            with open(os.path.join(folder,file), 'r') as all_lines:
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

                # for key in output_keys:
                #     if type(self.model_def['Outputs'][key]) is tuple:
                #         input_key = self.model_def['Outputs'][key][0]
                #         for i in range(0, len(self.input_data[(file,input_key)])-self.max_n_samples+add_sample_forward):
                #             aux_ind = i+self.max_n_samples+self.input_ns_forward[input_key]-self.max_samples_forward
                #             if self.input_n_samples[input_key] == 1:
                #                 self.inout_data_time_window[key].append(self.input_data[(file,input_key)][aux_ind-1])
                #             else:
                #                 self.inout_data_time_window[key].append(self.input_data[(file,input_key)][aux_ind-self.input_n_samples[input_key]:aux_ind])

                # for key in output_keys:
                #     used_key = key
                #     elem_key = key.split('__')
                #     if len(elem_key) > 1 and elem_key[1]== '-z1':
                #         used_key = elem_key[0]
                #     for i in range(0, len(self.input_data[(file,used_key)])-self.max_n_samples+add_sample_forward):
                #         self.inout_data_time_window[key].append(self.input_data[(file,used_key)][i+self.max_n_samples-self.max_samples_forward])

                # Index identifying each file start
                self.idx_of_rows.append(len(self.inout_data_time_window[list(self.input_n_samples.keys())[0]]))

        # Build the asarray for numpy
        for key,data in self.inout_data_time_window.items():
            self.inout_asarray[key]  = np.asarray(data)
            if data and self.num_of_samples is None:
                self.num_of_samples = len(self.inout_asarray[key])

        self.data_loaded = True

    #
    # Function that get specific parameters for training
    #
    def __getTrainParams(self, training_params, test_size):
        if bool(training_params):
            if 'batch_size' in training_params:
                if training_params['batch_size'] > round(self.num_of_samples*test_size):
                    self.batch_size = 1
                else:
                    self.batch_size = training_params['batch_size']
            #self.batch_size = (training_params['batch_size'] if 'batch_size' in training_params else self.batch_size)
            self.learning_rate = (training_params['learning_rate'] if 'learning_rate' in training_params else self.learning_rate)
            self.num_of_epochs = (training_params['num_of_epochs'] if 'num_of_epochs' in training_params else self.num_of_epochs)
            self.rnn_batch_size = (training_params['rnn_batch_size'] if 'rnn_batch_size' in training_params else self.rnn_batch_size)
            self.rnn_learning_rate = (training_params['rnn_learning_rate'] if 'rnn_learning_rate' in training_params else self.rnn_learning_rate)
            self.rnn_num_of_epochs = (training_params['rnn_num_of_epochs'] if 'rnn_num_of_epochs' in training_params else self.rnn_num_of_epochs)

    """
    Analysis of the results
    """
    def resultAnalysis(self, train_loss, test_loss, xy_train, xy_test):
        ## Plot train loss and test loss
        plt.plot(train_loss, label='train loss')
        plt.plot(test_loss, label='test loss')
        plt.legend()
        plt.show()

        # List of keys
        output_keys = list(self.model_def['Outputs'].keys())
        number_of_samples = int(self.n_samples_test*self.batch_size - self.batch_size) 

        # Performance parameters
        self.performance['se'] = np.empty([len(output_keys),number_of_samples])
        self.performance['mse'] = np.empty([len(output_keys),])
        self.performance['rmse_test'] = np.empty([len(output_keys),])
        self.performance['fvu'] = np.empty([len(output_keys),])

        # Prediction on test samples
        #test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=0, shuffle=False)
        self.prediction = np.empty((len(output_keys), number_of_samples))
        self.label = np.empty((len(output_keys), number_of_samples))

        with torch.inference_mode():
            self.model.eval()
            for idx in range(number_of_samples):
                XY = {}
                for key, val in xy_test.items():
                    XY[key] = torch.from_numpy(val[idx]).to(torch.float32).unsqueeze(dim=0)

                out = self.model(XY)
                for ind, obj in enumerate(self.minimize_list):
                    self.performance['se'][ind][idx] = self.loss_fn(out[obj[0]], out[obj[1]])

            for ind, obj in enumerate(self.minimize_list):
                # Mean Square Error
                self.performance['mse'][ind] = np.mean(self.performance['se'][ind])
                # Root Mean Square Error
                self.performance['rmse_test'][ind] = np.sqrt(np.mean(self.performance['se'][ind]))
                # Fraction of variance unexplained (FVU)
                #self.performance['fvu'][ind] = np.var(self.prediction[i] - self.label[i]) / np.var(self.label[i])

            # Index of worst results
            self.performance['max_se_idxs'] = np.argmax(self.performance['se'], axis=1)

            # Akaike’s Information Criterion (AIC) test
            ## TODO: Use log likelihood instead of MSE
            #self.performance['aic'] = - (self.num_of_test_sample * np.log(self.performance['mse'])) + 2 * self.model.count_params()

        self.visualizer.showResults(self, output_keys, performance = self.performance)

    """
    Training of the model.
    :param states: it is a list of a states, the state must be an Output object
    :param training_params: dict that contains the parameters of training (batch_size, learning rate, etc..)
    :param test_percentage: numeric value from 0 to 100, it is the part of the dataset used for validate the performance of the network
    :param show_results: it is a boolean for enable the plot of the results
    """
    def trainModel(self, test_percentage = 0, training_params = {}, show_results = False):

        # Check input
        train_size = 1 - (test_percentage / 100.0)
        test_size = 1 - train_size
        self.__getTrainParams(training_params, test_size=test_size)

        ## Split train and test
        XY_train = {}
        XY_test = {}
        for key,data in self.inout_data_time_window.items():
            if data:
                samples = np.asarray(data)
                if samples.ndim == 1:
                    samples = np.reshape(samples, (-1, 1))

                if key in self.model_def['Inputs'].keys():
                    XY_train[key] = samples[:int(len(samples)*train_size)]
                    XY_test[key] = samples[int(len(samples)*train_size):]
                    if self.n_samples_train is None:
                        self.n_samples_train = round(len(XY_train[key]) / self.batch_size)
                    if self.n_samples_test is None:
                        self.n_samples_test = round(len(XY_test[key]) / self.batch_size)
                #elif key in self.model_def['Outputs'].keys():
                #    Y_train[key] = samples[:int(len(samples)*train_size)]
                #    Y_test[key] = samples[int(len(samples)*train_size):]

        ## Build the dataset
        #train_data = Neu4MesDataset(X_train, Y_train)
        #test_data = Neu4MesDataset(X_test, Y_test)
        
        #self.train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, num_workers=0, shuffle=False)
        #self.test_loader = DataLoader(dataset=test_data, batch_size=self.batch_size, num_workers=0, shuffle=False)

        ## define optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = CustomRMSE()

        train_losses, test_losses = np.zeros(self.num_of_epochs), np.zeros(self.num_of_epochs)

        for iter in range(self.num_of_epochs):
            self.model.train()
            train_loss = []
            for i in range(self.n_samples_train):

                idx = i*self.batch_size
                XY = {}
                for key, val in XY_train.items():
                    XY[key] = torch.from_numpy(val[idx:idx+self.batch_size]).to(torch.float32)

                self.optimizer.zero_grad()
                out = self.model(XY)
                for obj in self.minimize_list:
                    loss = self.loss_fn(out[obj[0]], out[obj[1]])
                    loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
            train_loss = np.mean(train_loss)

            self.model.eval()
            test_loss = []
            for i in range(self.n_samples_test):

                idx = i*self.batch_size
                XY = {}
                for key, val in XY_test.items():
                    XY[key] = torch.from_numpy(val[idx:idx+self.batch_size]).to(torch.float32)

                out = self.model(XY)
                for obj in self.minimize_list:
                    loss = self.loss_fn(out[obj[0]], out[obj[1]])
                    #loss.backward()
                test_loss.append(loss.item())
            test_loss = np.mean(test_loss)

            train_losses[iter] = train_loss
            test_losses[iter] = test_loss

            if iter % 10 == 0:
                print(f'Epoch {iter+1}/{self.num_of_epochs}, Train Loss {train_loss:.4f}, Test Loss {test_loss:.4f}')

        # Show the analysis of the Result
        if show_results:
            self.resultAnalysis(train_losses, test_losses, XY_train, XY_test)
import torch
from torch.utils.data import DataLoader

import numpy as np
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

        self.inputs = {}                    # NN element - processed network inputs
        self.rnn_inputs = {}                # RNN element - processed network inputs
        self.inputs_for_model = {}          # NN element - clean network inputs
        self.rnn_inputs_for_model = {}      # RNN element - clean network inputs
        self.rnn_init_state = {}            # RNN element - for the states of RNN
        self.relations = {}                 # NN element - operations
        self.relation_samples = {}          # N samples for each relation
        self.outputs = {}                   # NN element - clean network outputs

        self.output_relation = {}           # dict with the outputs
        self.output_keys = []               # clear output signal keys (without delay info string __-z1)

        # Models of the framework
        self.model = None                   # NN model - Pytorch model
        self.rnn_model = None               # RNN model - Pytorch model
        self.net_weights = None             # NN weights

        # Optimizer parameters
        self.optimizer = None                     # NN model - Pytorch optimizer
        self.loss_fn = None                 # RNN model - Pytorch loss function

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
        if type(model_def) is Output:
            self.model_def = merge(self.model_def, model_def.json)
        elif type(model_def) is dict:
            self.model_def = merge(self.model_def, model_def)
        #self.MP(pprint,self.model_def)

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

        # Set the maximum time window for each input
        for name, params in self.model_def['Relations'].items():
            #if name not in self.model_def['Outputs'].keys():
            relation_params = params[1]
            for rel in relation_params:
                if type(rel) is tuple:
                    if rel[0] in self.model_def['Inputs'].keys():
                        tw = rel[1]
                        if type(tw) is list: ## backward + forward
                            assert tw[0] < tw[1], 'The first element of a time window must be less that the second (Ex: [-2, 2])'
                            if rel[0] in self.input_tw_backward.keys(): ## Update if grater
                                self.input_tw_backward[rel[0]] = max(abs(tw[0]), self.input_tw_backward[rel[0]])
                                self.input_tw_forward[rel[0]] = max(tw[1], self.input_tw_forward[rel[0]])
                            else:
                                self.input_tw_backward[rel[0]] = abs(tw[0])
                                self.input_tw_forward[rel[0]] = tw[1]
                        else: ## Only backward

                            if rel[0] in self.input_tw_backward.keys(): ## Update if grater
                                self.input_tw_backward[rel[0]] = max(tw, self.input_tw_backward[rel[0]])
                                self.input_tw_forward[rel[0]] = max(0, self.input_tw_forward[rel[0]])
                            else:
                                self.input_tw_backward[rel[0]] = tw
                                self.input_tw_forward[rel[0]] = 0

                        self.input_ns_backward[rel[0]] = int(self.input_tw_backward[rel[0]] / self.model_def['SampleTime'])
                        self.input_ns_forward[rel[0]] = int(self.input_tw_forward[rel[0]] / self.model_def['SampleTime'])
                        self.input_n_samples[rel[0]] = self.input_ns_backward[rel[0]] + self.input_ns_forward[rel[0]]
                else:
                    if rel in self.model_def['Inputs'].keys(): ## instantaneous input
                        if rel not in self.input_tw_backward.keys():
                            self.input_tw_backward[rel] = self.model_def['SampleTime']
                            self.input_tw_forward[rel] = 0

                            self.input_ns_backward[rel] = 1
                            self.input_ns_forward[rel] = 0
                            self.input_n_samples[rel] = self.input_ns_backward[rel] + self.input_ns_forward[rel]

        self.max_samples_backward = max(self.input_ns_backward.values())
        self.max_samples_forward = max(self.input_ns_forward.values())
        self.max_n_samples = self.max_samples_forward + self.max_samples_backward

        self.MP(pprint,{"window_backward": self.input_tw_backward, "window_forward":self.input_tw_forward})
        self.MP(pprint,{"samples_backward": self.input_ns_backward, "samples_forward":self.input_ns_forward})
        self.MP(pprint,{"input_n_samples": self.input_n_samples})
        self.MP(pprint,{"max_samples_backward": self.max_samples_backward, "max_samples_forward":self.max_samples_forward, "max_samples":self.max_n_samples})

        ## Get samples per relation
        for name, inputs in self.model_def['Relations'].items():
            input_samples = {}
            for input_name in inputs[1]:
                if type(input_name) is tuple: ## we have a window
                    if type(input_name[1]) is list: ## we have the forward and backward window
                        if input_name[0] in self.model_def['Inputs']:
                            backward = self.input_ns_backward[input_name[0]] - int(abs(input_name[1][0])/sample_time)
                            forward = self.input_ns_backward[input_name[0]] + int(abs(input_name[1][1])/sample_time)
                        else:
                            backward = int(abs(input_name[1][0])/sample_time)
                            forward = int(abs(input_name[1][1])/sample_time)

                        if len(input_name) == 3: ## we have the offset
                            offset = int(abs(input_name[1][0])/sample_time) + int(input_name[2] / sample_time)
                        else:
                            offset = None

                    else: ## we have only the backward window
                        if input_name[0] in self.model_def['Inputs']:
                            backward = self.input_ns_backward[input_name[0]] - int(abs(input_name[1])/sample_time)
                            forward = self.input_ns_backward[input_name[0]] + 0

                        if len(input_name) == 3: ## we have the offset
                            offset = int(abs(input_name[1])/sample_time) + int(input_name[2] / sample_time)
                        else:
                            offset = None
                    
                    input_samples[input_name[0]] = {'backward':backward, 'forward':forward, 'offset':offset}
                else: ## we have no window
                    input_samples[input_name] = {'backward':0, 'forward':1, 'offset':None}

            self.relation_samples[name] = input_samples

        self.MP(pprint,{"relation_samples": self.relation_samples})


        ## Build the network
        self.model = Model(self.model_def, self.relation_samples)
        self.MP(pprint,self.model)
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

                for key in output_keys:
                    used_key = key
                    elem_key = key.split('__')
                    if len(elem_key) > 1 and elem_key[1]== '-z1':
                        used_key = elem_key[0]
                    for i in range(0, len(self.input_data[(file,used_key)])-self.max_n_samples+add_sample_forward):
                        self.inout_data_time_window[key].append(self.input_data[(file,used_key)][i+self.max_n_samples-self.max_samples_forward])

                # Index identifying each file start
                self.idx_of_rows.append(len(self.inout_data_time_window[list(self.input_n_samples.keys())[0]]))

        # Build the asarray for numpy
        for key,data in self.inout_data_time_window.items():
            self.inout_asarray[key]  = np.asarray(data)
            if data and self.num_of_samples is None:
                self.num_of_samples = len(self.inout_asarray[key])

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
    def resultAnalysis(self, train_loss, test_loss, x_test, y_test):
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
                X, Y = {}, {}
                for key, val in x_test.items():
                    #item[key] = torch.tensor(val[index], dtype=torch.float32)
                    X[key] = torch.from_numpy(val[idx]).to(torch.float32).unsqueeze(dim=0)
                for key, val in y_test.items():
                    Y[key] = torch.tensor(val[idx], dtype=torch.float32)

                pred = self.model(X)
                for i, key in enumerate(output_keys):
                    self.prediction[i][idx] = pred[key].item() 
                    self.label[i][idx] = Y[key].item() 
                    self.performance['se'][i][idx] = np.square(pred[key].item() - Y[key].item())

            for i, key in enumerate(output_keys):
                # Mean Square Error 
                self.performance['mse'][i] = np.mean(self.performance['se'][i])
                # Root Mean Square Error
                self.performance['rmse_test'][i] = np.sqrt(np.mean(self.performance['se'][i]))
                # Fraction of variance unexplained (FVU) 
                self.performance['fvu'][i] = np.var(self.prediction[i] - self.label[i]) / np.var(self.label[i])

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
        X_train, Y_train = {}, {}
        X_test, Y_test = {}, {}
        for key,data in self.inout_data_time_window.items():
            if data:
                samples = np.asarray(data)
                if samples.ndim == 1:
                    samples = np.reshape(samples, (-1, 1))

                if key in self.model_def['Inputs'].keys():
                    X_train[key] = samples[:int(len(samples)*train_size)]
                    X_test[key] = samples[int(len(samples)*train_size):]
                    if self.n_samples_train is None:
                        self.n_samples_train = round(len(X_train[key]) / self.batch_size)
                    if self.n_samples_test is None:
                        self.n_samples_test = round(len(X_test[key]) / self.batch_size)
                elif key in self.model_def['Outputs'].keys():
                    Y_train[key] = samples[:int(len(samples)*train_size)]
                    Y_test[key] = samples[int(len(samples)*train_size):]

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
                X, Y = {}, {}
                for key, val in X_train.items():
                    #item[key] = torch.tensor(val[index], dtype=torch.float32)
                    X[key] = torch.from_numpy(val[idx:idx+self.batch_size]).to(torch.float32)
                for key, val in Y_train.items():
                    Y[key] = torch.tensor(val[idx:idx+self.batch_size], dtype=torch.float32)

                #inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()
                out = self.model(X)
                loss = self.loss_fn(out, Y)
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
            train_loss = np.mean(train_loss)

            self.model.eval()
            test_loss = []
            for i in range(self.n_samples_test):

                idx = i*self.batch_size
                X, Y = {}, {}
                for key, val in X_test.items():
                    #item[key] = torch.tensor(val[index], dtype=torch.float32)
                    X[key] = torch.from_numpy(val[idx:idx+self.batch_size]).to(torch.float32)
                for key, val in Y_test.items():
                    Y[key] = torch.tensor(val[idx:idx+self.batch_size], dtype=torch.float32)

                #inputs, labels = inputs.to(device), labels.to(device)
                out = self.model(X)
                loss = self.loss_fn(out, Y)
                test_loss.append(loss.item())
            test_loss = np.mean(test_loss)

            train_losses[iter] = train_loss
            test_losses[iter] = test_loss

            if iter % 10 == 0:
                print(f'Epoch {iter+1}/{self.num_of_epochs}, Train Loss {train_loss:.4f}, Test Loss {test_loss:.4f}')

        # Show the analysis of the Result
        if show_results:
            self.resultAnalysis(train_losses, test_losses, X_test, Y_test)
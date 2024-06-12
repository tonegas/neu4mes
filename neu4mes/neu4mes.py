import copy

import torch
from torch.utils.data import DataLoader

import numpy as np
from scipy.stats import norm
import random
import os
from pprint import pprint
from pprint import pformat
import re
from datetime import datetime
import matplotlib.pyplot as plt

from neu4mes.relation import NeuObj, merge
from neu4mes.visualizer import TextVisualizer, Visualizer
from neu4mes.dataset import Neu4MesDataset
from neu4mes.loss import CustomRMSE
from neu4mes.output import Output
from neu4mes.model import Model
from neu4mes.utilis import check, argmax_max, argmin_min


from neu4mes import LOG_LEVEL
from neu4mes.logger import logging
log = logging.getLogger(__name__)
log.setLevel(max(logging.ERROR, LOG_LEVEL))

class Neu4mes:
    name = None
    def __init__(self, model_def = 0, visualizer = 'Standard'):
        # Visualizer
        if visualizer == 'Standard':
            self.visualizer = TextVisualizer(1)
        elif visualizer != None:
            self.visualizer = visualizer
        else:
            self.visualizer = Visualizer()
        self.visualizer.set_n4m(self)

        # Inizialize the model definition
        # the model_def has all the relation defined for that model

        self.model_def = NeuObj().json
        self.addModel(model_def)
        # if type(model_def) is Output:
        #     self.model_def = model_def.json
        # elif type(model_def) is dict:
        #     self.model_def = self.model_def
        # else:
        #     self.model_def = NeuObj().json

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
        self.minimize_dict = {}
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
        #self.num_of_training_sample = 0     # number of rows for training
        self.n_samples_train = None
        self.n_samples_test = None

        self.idx_of_rows = [0]              # Index identifying each file start
        self.first_idx_test = 0             # Index identifying the first test

        # Dataloaders
        self.train_loader = None
        self.test_loader = None

        # Training params
        #self.batch_size = 128                               # batch size
        self.train_batch_size = 128                          # train batch size
        self.test_batch_size = 128                         # test batch size
        self.learning_rate = 0.0005                         # learning rate for NN
        self.num_of_epochs = 100                             # number of epochs
        #self.rnn_batch_size = self.batch_size               # batch size for RNN
        self.rnn_window = None                              # window of the RNN
        #self.rnn_learning_rate = self.learning_rate/10000   # learning rate for RNN
        #self.rnn_num_of_epochs = 50                         # number of epochs for RNN

        # Training dataset
        self.data_loaded = False
        self.inout_4train = {}                              # Dataset for training NN
        self.inout_4test = {}                               # Dataset for test NN
        self.rnn_inout_4train = {}                          # Dataset for training RNN
        self.rnn_inout_4test = {}                           # Dataset for test RNN

        # Training performance
        self.performance = {}                               # Dict with performance parameters for NN
        self.prediction = {}

    def __call__(self, inputs, sampled=False):
        model_inputs = self.model_def['Inputs'].keys()

        ## Check if there are some missing inputs
        check(set(model_inputs).issubset(set(inputs.keys())),
              KeyError,
              f'The complete model inputs are {list(model_inputs)}, the provided input are {list(inputs.keys())}')
        #assert set(model_inputs).issubset(set(inputs.keys())), 'Missing Input!'

        ## Make all Scalar input inside a list
        #for key, val in inputs.items():
        #    if type(val) is not list:
        #        inputs[key] = [val]

        ## Determine the Maximal number of samples that can be created
        if sampled:
            min_dim_ind, min_dim  = argmin_min([len(inputs[key]) for key in model_inputs])
            max_dim_ind, max_dim = argmax_max([len(inputs[key]) for key in model_inputs])
        else:
            min_dim_ind, min_dim = argmin_min([len(inputs[key])-self.input_n_samples[key]+1 for key in model_inputs])
            max_dim_ind, max_dim  = argmax_max([len(inputs[key])-self.input_n_samples[key]+1 for key in model_inputs])
        window_dim = min_dim
        check(window_dim > 0, StopIteration, 'Invalid Number of Inputs!')

        ## warning the users about different time windows between samples
        if min_dim != max_dim:
            self.visualizer.warning(f'Different number of samples between inputs [MAX {list(model_inputs)[max_dim_ind]} = {max_dim}; MIN {list(model_inputs)[min_dim_ind]} = {min_dim}]')

        result_dict = {} ## initialize the resulting dictionary
        for key in self.model_def['Outputs'].keys():
            result_dict[key] = []

        for i in range(window_dim):
            X = {}
            for key, val in inputs.items():
                if key in model_inputs:
                    if sampled:
                        X[key] = torch.from_numpy(np.array(val[i])).to(torch.float32)
                    else:
                        X[key] = torch.from_numpy(np.array(val[i:i+self.input_n_samples[key]])).to(torch.float32)
                        
                    if X[key].ndim == 0: ## add the batch dimension
                        X[key] = X[key].unsqueeze(0)
                    if X[key].ndim == 1: ## add the input dimension
                        X[key] = X[key].unsqueeze(0)
            result = self.model(X)
            for key in self.model_def['Outputs'].keys():
                result_dict[key].append(result[key].squeeze().detach().numpy().tolist())

        return result_dict
    
    def get_random_samples(self, window=1):
        if self.data_loaded:
            result_dict = {}
            for key in self.model_def['Inputs'].keys():
                result_dict[key] = []
            random_idx = random.randint(0, self.num_of_samples - window)
            for idx in range(window):
                for key ,data in self.inout_data_time_window.items():
                    if key in self.model_def['Inputs'].keys() and data is not None:
                        result_dict[key].append(data[random_idx+idx])
            return result_dict
        else:
            print('The Dataset must first be loaded using <loadData> function!')
            return {}


    """
    Add a new model to be trained:
    :param model_def: can be a json model definition or a Output object
    """
    def addModel(self, model_def):
        if type(model_def) is Output:
            self.model_def = merge(self.model_def, model_def.json)
        elif type(model_def) is dict:
            self.model_def = merge(self.model_def, model_def)
        elif type(model_def) is list:
            for item in model_def:
                self.addModel(item)
    def minimizeError(self, variable_name, streamA, streamB, loss_function='mse'):
        self.model_def = merge(self.model_def, streamA.json)
        self.model_def = merge(self.model_def, streamB.json)
        A = (streamA.name[0] if type(streamA.name) is tuple else streamA.name)
        nameA = variable_name + '_' + A
        if type(streamA) is not Output:
            self.model_def['Outputs'][nameA] = streamA.name
        else:
            self.model_def['Outputs'][nameA] = self.model_def['Outputs'][streamA.name]

        B = (streamB.name[0] if type(streamB.name) is tuple else streamB.name)
        nameB = variable_name + '_' + B
        if type(streamB) is not Output:
            self.model_def['Outputs'][nameB] = streamB.name
        else:
            self.model_def['Outputs'][nameB] = self.model_def['Outputs'][streamB.name]
        self.minimize_list.append((nameA, nameB, loss_function))
        self.minimize_dict[variable_name]={'A':(nameA, copy.deepcopy(streamA)), 'B':(nameB, copy.deepcopy(streamB)), 'loss':loss_function}
        self.visualizer.showMinimizeError(variable_name)

    """
    Definition of the network structure through the dependency graph and sampling time.
    If a prediction window is also specified, it means that a recurrent network is also to be defined.
    :param sample_time: the variable defines the rate of the network based on the training set
    :param prediction_window: the variable defines the prediction horizon in the future
    """
    ## TODO sample time can be 0
    def neuralizeModel(self, sample_time = 1, prediction_window = None):
        # Prediction window is used for the recurrent network
        if prediction_window is not None:
            self.rnn_window = round(prediction_window/sample_time)
            assert prediction_window >= sample_time

        # Sample time is used to define the number of sample for each time window

        assert sample_time != 0, 'sample_time cannot be zero!'
        self.model_def["SampleTime"] = sample_time
        self.visualizer.showModel()


        for key, value in self.model_def['Inputs'].items():
            self.input_tw_backward[key] = -value['tw'][0]
            self.input_tw_forward[key] = value['tw'][1]
            if value['sw'] == [0,0] and value['tw'] == [0,0]:
                self.input_tw_backward[key] = sample_time
            if value['sw'] == [0,0] :
                self.input_ns_backward[key] = round(self.input_tw_backward[key] / sample_time)
                self.input_ns_forward[key] = round(self.input_tw_forward[key] / sample_time)
            else:
                self.input_ns_backward[key] = max(round(self.input_tw_backward[key] / sample_time),-value['sw'][0])
                self.input_ns_forward[key] = max(round(self.input_tw_forward[key] / sample_time),value['sw'][1])
            self.input_n_samples[key] = self.input_ns_backward[key] + self.input_ns_forward[key]

        self.max_samples_backward = max(self.input_ns_backward.values())
        self.max_samples_forward = max(self.input_ns_forward.values())
        if self.max_samples_backward < 0:
            self.visualizer.warning(f"The input is only in the far past the max_samples_backward is: {self.max_samples_backward}")
        if self.max_samples_forward < 0:
            self.visualizer.warning(f"The input is only in the far future the max_sample_forward is: {self.max_samples_forward}")
        self.max_n_samples = self.max_samples_forward + self.max_samples_backward

        self.visualizer.showModelInputWindow()

        ## Get samples per relation
        for name, inputs in (self.model_def['Relations']|self.model_def['Outputs']).items():
            input_samples = {}
            inputs = inputs[1] if name in self.model_def['Relations'] else [inputs]
            for input_name in inputs:
                if type(input_name) is tuple: ## we have a window
                    window = 'tw' if 'tw' in input_name[1] else 'sw'
                    aux_sample_time = sample_time if 'tw' in input_name[1] else 1
                    if type(input_name[1][window]) is list: ## we have the forward and backward window
                        assert input_name[0] in self.model_def['Inputs'], "Why isn't the input in model_def['Inputs']?"
                        # if input_name[0] in self.model_def['Inputs']:
                        backward = self.input_ns_backward[input_name[0]] + round(input_name[1][window][0]/aux_sample_time)
                        forward = self.input_ns_backward[input_name[0]] + round(input_name[1][window][1]/aux_sample_time)
                        # else:
                        #     backward = round(abs(input_name[1][window][0])/aux_sample_time)
                        #     forward = round(abs(input_name[1][window][1])/aux_sample_time)

                        if 'offset' in input_name[1]: ## we have the offset
                            offset = self.input_ns_backward[input_name[0]] + round(input_name[1]['offset'] / aux_sample_time)
                            input_samples[input_name[0]] = {'start_idx': backward, 'end_idx': forward, 'offset_idx': offset}
                        else:
                            input_samples[input_name[0]] = {'start_idx':backward, 'end_idx': forward}

                else: ## we have no window
                    input_samples[input_name] = {'start_idx':0, 'end_idx':1}

            self.relation_samples[name] = input_samples

        self.visualizer.showModelRelationSamples()


        ## Build the network
        self.model = Model(self.model_def, self.relation_samples)
        self.visualizer.showBuiltModel()
        self.neuralized = True
    """
    Loading of the data set files and generate the structure for the training considering the structure of the input and the output
    :param format: it is a list of the variable in the csv. All the input keys must be inside this list.
    :param folder: folder of the dataset. Each file is a simulation.
    :param sample_time: number of lines to be skipped (header lines)
    :param delimiters: it is a list of the symbols used between the element of the file
    """
    ## TODO: can load data from dictionaries
    def loadData(self, source, format=None, skiplines = 0, delimiters=['\t',';',',']):
        assert self.neuralized == True, "The network is not neuralized yet."

        if type(source) is str: ## we have a file path
            path, dirs, files = next(os.walk(source))
            self.file_count = len(files)

            # Create a vector of all the signals in the file + output_relation keys
            output_keys = self.model_def['Outputs'].keys()
            for key in format+list(output_keys):
                self.inout_data_time_window[key] = []

            # Read each file
            for file in files:
                for data in format:
                    self.input_data[(file,data)] = []

                # Open the file and read lines
                with open(os.path.join(source, file), 'r') as all_lines:
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

                    # Index identifying each file start
                    #self.idx_of_rows.append(len(self.inout_data_time_window[list(self.input_n_samples.keys())[0]]))

        elif type(source) is dict:  ## we have a crafted dataset
            self.file_count = 1
            ## Check if the inputs are correct
            model_inputs = self.model_def['Inputs'].keys()
            assert set(model_inputs).issubset(source.keys()), 'The dataset is missing some inputs.'

            for key in model_inputs:
                self.inout_data_time_window[key] = []  ## Initialize the dataset
                for idx in range(len(source[key]) - self.max_n_samples + 1):
                    self.inout_data_time_window[key].append(source[key][idx + (self.max_samples_backward - self.input_ns_backward[key]):idx + (self.max_samples_backward + self.input_ns_forward[key])].tolist())

        # Build the asarray for numpy
        for key,data in self.inout_data_time_window.items():
            self.inout_asarray[key]  = np.asarray(data)
            if data and self.num_of_samples is None:
                self.num_of_samples = len(self.inout_asarray[key])

        self.visualizer.showDataset()
        ## Set the Loaded flag to True
        self.data_loaded = True

    #
    # Function that get specific parameters for training
    #
    def __getTrainParams(self, training_params, train_size, test_size):
        if bool(training_params):
            if 'batch_size' in training_params:
                if training_params['batch_size'] > round(self.num_of_samples * test_size):
                    self.batch_size = 1
                else:
                    self.batch_size = training_params['batch_size']
            #self.batch_size = (training_params['batch_size'] if 'batch_size' in training_params else self.batch_size)

            self.learning_rate = (training_params['learning_rate'] if 'learning_rate' in training_params else self.learning_rate)
            self.num_of_epochs = (training_params['num_of_epochs'] if 'num_of_epochs' in training_params else self.num_of_epochs)
            self.train_batch_size = (training_params['train_batch_size'] if 'train_batch_size' in training_params else self.train_batch_size)
            self.test_batch_size = (training_params['test_batch_size'] if 'test_batch_size' in training_params else self.test_batch_size)

            ## Check if the batch_size can be used for the current dataset, otherwise set the batch_size to 1
            if self.train_batch_size > round(self.num_of_samples*train_size):
                self.train_batch_size = 1
            if self.test_batch_size > round(self.num_of_samples*test_size):
                self.test_batch_size = 1
            #if 'batch_size' in training_params:
            #    if training_params['batch_size'] > round(self.num_of_samples*test_size):
            #        self.batch_size = 1
            #    else:
            #        self.batch_size = training_params['batch_size']
            #self.batch_size = (training_params['batch_size'] if 'batch_size' in training_params else self.batch_size)
            #self.learning_rate = (training_params['learning_rate'] if 'learning_rate' in training_params else self.learning_rate)
            #self.num_of_epochs = (training_params['num_of_epochs'] if 'num_of_epochs' in training_params else self.num_of_epochs)
            #self.rnn_batch_size = (training_params['rnn_batch_size'] if 'rnn_batch_size' in training_params else self.rnn_batch_size)
            #self.rnn_learning_rate = (training_params['rnn_learning_rate'] if 'rnn_learning_rate' in training_params else self.rnn_learning_rate)
            #self.rnn_num_of_epochs = (training_params['rnn_num_of_epochs'] if 'rnn_num_of_epochs' in training_params else self.rnn_num_of_epochs)

    """
    Analysis of the results
    """
    def resultAnalysis(self, train_losses, test_losses, XY_train, XY_test):
        with torch.inference_mode():
            self.model.eval()
            key = list(XY_test.keys())[0]
            samples = len(XY_test[key])
            #samples = self.n_samples_test
            #samples = 1

            #batch_size = int(len(XY_test['x']) / samples)
            aux_test_losses = np.zeros([len(self.minimize_dict), samples])
            A = torch.zeros(len(self.minimize_dict), samples)
            B = torch.zeros(len(self.minimize_dict), samples)
            for i in range(samples):

                XY = {}
                for key, val in XY_test.items():
                    XY[key] = torch.from_numpy(val[i]).to(torch.float32).unsqueeze(dim=0)

                # idx = i * batch_size
                # XY = {}
                # for key, val in XY_test.items():
                #     XY[key] = torch.from_numpy(val[idx:idx + batch_size]).to(torch.float32)
                out = self.model(XY)
                for ind, (key, value) in enumerate(self.minimize_dict.items()):
                    A[ind][i] = out[value['A'][0]]
                    B[ind][i] = out[value['B'][0]]
                    loss = self.loss_fn(A[ind][i], B[ind][i])
                    aux_test_losses[ind][i] = loss.detach().numpy()


            for ind, (key, value) in enumerate(self.minimize_dict.items()):
                A_np = A[ind].detach().numpy()
                B_np = B[ind].detach().numpy()
                self.performance[key] = {}
                self.performance[key][value['loss']] = {'epoch_test': test_losses[key], 'epoch_train': train_losses[key], 'test': np.mean(aux_test_losses[ind])}
                self.performance[key]['fvu'] = {}
                # Compute FVU
                residual = A_np - B_np
                error_var = np.var(residual)
                error_mean = np.mean(residual)
                #error_var_manual = np.sum((residual-error_mean) ** 2) / (len(self.prediction['B'][ind]) - 0)
                #print(f"{key} var np:{new_error_var} and var manual:{error_var_manual}")
                self.performance[key]['fvu']['A'] = (error_var / np.var(A_np)).item()
                self.performance[key]['fvu']['B'] = (error_var / np.var(B_np)).item()
                self.performance[key]['fvu']['total'] = np.mean([self.performance[key]['fvu']['A'],self.performance[key]['fvu']['B']]).item()
                # Compute AIC
                #normal_dist = norm(0, error_var ** 0.5)
                #probability_of_residual = normal_dist.pdf(residual)
                #log_likelihood_first = sum(np.log(probability_of_residual))
                p1 = -len(residual)/2.0*np.log(2*np.pi)
                p2 = -len(residual)/2.0*np.log(error_var)
                p3 = -1/(2.0*error_var)*np.sum(residual**2)
                log_likelihood = p1+p2+p3
                #print(f"{key} log likelihood second mode:{log_likelihood} = {p1}+{p2}+{p3} first mode: {log_likelihood_first}")
                total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) #TODO to be check the number is doubled
                #print(f"{key} total_params:{total_params}")
                aic = - 2 * log_likelihood + 2 * total_params
                #print(f"{key} aic:{aic}")
                self.performance[key]['aic'] = {'value':aic,'total_params':total_params,'log_likelihood':log_likelihood}
                # Prediction and target
                self.prediction[key] = {}
                self.prediction[key]['A'] = A_np.tolist()
                self.prediction[key]['B'] = B_np.tolist()

            self.performance['total'] = {}
            self.performance['total']['mean_error'] = {'test': np.mean(aux_test_losses)}
            self.performance['total']['fvu'] = np.mean([self.performance[key]['fvu']['total'] for key in self.minimize_dict.keys()])
            self.performance['total']['aic'] = np.mean([self.performance[key]['aic']['value']for key in self.minimize_dict.keys()])

        self.visualizer.showResults()


    """
    Training of the model.
    :param training_params: dict that contains the parameters of training (batch_size, learning rate, etc..)
    :param test_percentage: numeric value from 0 to 100, it is the part of the dataset used for validate the performance of the network
    :param show_results: it is a boolean for enable the plot of the results
    """
    def trainModel(self, test_percentage = 0, training_params = {}):
        import time
        start = time.time()

        # Check input
        train_size = 1 - (test_percentage / 100.0)
        test_size = 1 - train_size
        self.__getTrainParams(training_params, train_size=train_size, test_size=test_size)

        ## Split train and test
        XY_train = {}
        XY_test = {}
        for key,data in self.inout_data_time_window.items():
            if data:
                samples = np.asarray(data)
                if samples.ndim == 1:
                    samples = np.reshape(samples, (-1, 1))

                if key in self.model_def['Inputs'].keys():
                    if test_percentage == 0:
                        XY_train[key] = samples
                    else:
                        XY_train[key] = samples[:round(len(samples)*train_size)]
                        XY_test[key] = samples[round(len(samples)*train_size):]
                        if self.n_samples_test is None:
                            self.n_samples_test = round(len(XY_test[key]) / self.test_batch_size)
                    if self.n_samples_train is None:
                        self.n_samples_train = round(len(XY_train[key]) / self.train_batch_size)

        ## define optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = CustomRMSE()

        train_losses, test_losses = {}, {}
        for key in self.minimize_dict.keys():
            train_losses[key] = []
            test_losses[key] = []

        for epoch in range(self.num_of_epochs):
            self.model.train()
            aux_train_losses = torch.zeros([len(self.minimize_dict),self.n_samples_train])
            for i in range(self.n_samples_train):
                idx = i*self.train_batch_size
                XY = {}
                for key, val in XY_train.items():
                    XY[key] = torch.from_numpy(val[idx:idx+self.train_batch_size]).to(torch.float32)

                self.optimizer.zero_grad()
                out = self.model(XY)
                for ind, (key, value) in enumerate(self.minimize_dict.items()):
                    loss = self.loss_fn(out[value['A'][0]], out[value['B'][0]])
                    loss.backward(retain_graph=True)
                    aux_train_losses[ind][i]= loss.item()

                self.optimizer.step()

            for ind, key in enumerate(self.minimize_dict.keys()):
                train_losses[key].append(torch.mean(aux_train_losses[ind]).tolist())

            if test_percentage != 0:
                self.model.eval()
                aux_test_losses = torch.zeros(len(self.minimize_dict), self.n_samples_test)
                for i in range(self.n_samples_test):

                    idx = i * self.test_batch_size
                    XY = {}
                    for key, val in XY_test.items():
                        XY[key] = torch.from_numpy(val[idx:idx + self.test_batch_size]).to(torch.float32)

                    out = self.model(XY)
                    for ind, (key, value) in enumerate(self.minimize_dict.items()):
                        loss = self.loss_fn(out[value['A'][0]], out[value['B'][0]])
                        aux_test_losses[ind][i] = loss.item()

                for ind, key in enumerate(self.minimize_dict.keys()):
                    test_losses[key].append(torch.mean(aux_test_losses[ind]).tolist())

            self.visualizer.showTraining(epoch, train_losses, test_losses)

        end = time.time()
        self.visualizer.showTrainingTime(end-start)
        self.resultAnalysis(train_losses, test_losses, XY_train, XY_test)


    """
    Recurrent Training of the model.
    :param close_loop: dict that contains the variables that will be used into the recurrent loop
    :param prediction_horizon: size of the prediction window
    :param step: how much to skip between different windows
    :param training_params: dict that contains the parameters of training (batch_size, learning rate, etc..)
    :param test_percentage: numeric value from 0 to 100, it is the part of the dataset used for validate the performance of the network
    """
    def trainRecurrentModel(self, close_loop, prediction_horizon=None, step=1, test_percentage = 0, training_params = {}):
        import time
        start = time.time()

        sample_time = self.model_def['SampleTime']
        if prediction_horizon is None:
            prediction_horizon = sample_time

        # Initialize input
        prediction_samples = round(prediction_horizon / sample_time)
        train_size = 1 - (test_percentage / 100.0)
        test_size = 1 - train_size
        self.__getTrainParams(training_params, train_size=train_size, test_size=test_size)

        ## Split train and test
        XY_train = {}
        XY_test = {}
        for key,data in self.inout_data_time_window.items():
            if data:
                samples = np.asarray(data)
                if samples.ndim == 1:
                    samples = np.reshape(samples, (-1, 1))
                if key in self.model_def['Inputs'].keys():
                    if test_percentage == 0:
                        XY_train[key] = samples
                    else:
                        XY_train[key] = samples[:round(len(samples)*train_size)]
                        XY_test[key] = samples[round(len(samples)*train_size):]
                        if self.n_samples_test is None:
                            self.n_samples_test = round(len(XY_test[key]) / self.test_batch_size)
                    if self.n_samples_train is None:
                        self.n_samples_train = round(len(XY_train[key]) / self.train_batch_size)

        ## Check input
        assert self.n_samples_train > prediction_samples and self.n_samples_test > prediction_samples, f'Error: The Prediction window is set to large (Max {(min(self.n_samples_test,self.n_samples_train)-1)*sample_time})'

        ## define optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = CustomRMSE()

        train_losses, test_losses = {}, {}
        for key in self.minimize_dict.keys():
            train_losses[key] = np.zeros(self.num_of_epochs)
            test_losses[key] = np.zeros(self.num_of_epochs)

        #print('[LOG] n_samples_train: ', self.n_samples_train)
        #print('[LOG] prediction_samples: ', prediction_samples)
        for epoch in range(self.num_of_epochs):
            self.model.train()
            train_loss = []
            for i in range(0, (self.n_samples_train - prediction_samples), step):

                idx = i*self.train_batch_size
                XY = {}
                XY_horizon = {}
                for key, val in XY_train.items():
                    XY[key] = torch.from_numpy(val[idx:idx+self.train_batch_size]).to(torch.float32)
                    ## This dictionary contains the horizon labels
                    XY_horizon[key] = torch.from_numpy(val[idx+1:idx+self.train_batch_size+prediction_samples+1]).to(torch.float32)

                self.optimizer.zero_grad()  
                losses = []
                #print('[LOG] prediction samples: ', prediction_samples)
                for horizon_idx in range(prediction_samples):
                    #print('[LOG] XY: ', XY)
                    #print('[LOG] XY_horizon: ', XY_horizon)
                    out = self.model(XY)
                    #print('[LOG] out: ', out)
                    for key, value in self.minimize_dict.items():
                        #print('[LOG] valueA: ', out[value['A'][0]])
                        #print('[LOG] ValueB: ', out[value['B'][0]])
                        loss = self.loss_fn(out[value['A'][0]], out[value['B'][0]]) 
                        #print('[LOG] loss: ', loss.item()) 
                        losses.append(loss)

                    ## Update the input with the recurrent prediction
                    for key in XY.keys():
                        XY[key] = torch.roll(XY[key], shifts=-1, dims=1)
                        XY[key][:, -1] = XY_horizon[key][horizon_idx:horizon_idx+self.train_batch_size, -1]
                        if key in close_loop.keys():
                            XY[key][:, -2] = out[close_loop[key]].squeeze()

                #loss = sum(losses) / prediction_samples
                loss = sum(losses)
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
            train_loss = np.mean(train_loss)
            train_losses[epoch] = train_loss

            if test_percentage != 0:
                self.model.eval()
                test_loss = []
                for i in range(0, (self.n_samples_test - prediction_samples), step):

                    idx = i*self.test_batch_size
                    XY = {}
                    XY_horizon = {}
                    for key, val in XY_test.items():
                        XY[key] = torch.from_numpy(val[idx:idx+self.test_batch_size]).to(torch.float32)
                        XY_horizon[key] = torch.from_numpy(val[idx+1:idx+self.test_batch_size+prediction_samples+1]).to(torch.float32)

                    losses = []
                    for horizon_idx in range(prediction_samples):
                        out = self.model(XY)
                        for key, value in self.minimize_dict.items():
                            loss = self.loss_fn(out[value['A'][0]], out[value['B'][0]])  
                            losses.append(loss)

                        ## Update the input with the recurrent prediction
                        for key in XY.keys():
                            XY[key] = torch.roll(XY[key], shifts=-1, dims=1)
                            XY[key][:, -1] = XY_horizon[key][horizon_idx:horizon_idx+self.test_batch_size, -1]
                            if key in close_loop.keys():
                                XY[key][:, -2] = out[close_loop[key]].squeeze()

                    loss = sum(losses) / prediction_samples
                    test_loss.append(loss.item())
                test_loss = np.mean(test_loss)
                test_losses[epoch] = test_loss

            self.visualizer.showTraining(epoch, train_losses, test_losses)

        end = time.time()
        self.visualizer.showTrainingTime(end - start)

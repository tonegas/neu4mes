import copy

import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
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
from neu4mes.loss import CustomLoss
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
        self.model_def = NeuObj().json
        self.addModel(model_def)

        ## Variables
        self.model = None

        self.minimize_list = []
        self.minimize_dict = {}

        self.input_tw_backward, self.input_tw_forward = {}, {}
        self.input_ns_backward, self.input_ns_forward = {}, {}
        self.input_n_samples = {}
        self.max_samples_backward, self.max_samples_forward = 0, 0
        self.max_n_samples = 0

        self.neuralized = False
        self.data_loaded = False

        self.file_count = 0
        #self.inout_data_time_window = {}  ## TODO: remove in new loadData
        #self.input_data = {}  ## TODO: remove in new loadData
        #self.inout_asarray = {}  ## TODO: remove in new loadData
        self.num_of_samples = None
        self.data = {}

        self.learning_rate = 0.01
        self.num_of_epochs = 100
        self.train_batch_size = 1
        self.test_batch_size = 1
        self.n_samples_train, self.n_samples_test = None, None
        self.optimizer = None
        self.losses = {}


    def __call__(self, inputs, sampled=False):
        model_inputs = list(self.model_def['Inputs'].keys())
        provided_inputs = list(inputs.keys())
        missing_inputs = list(set(model_inputs) - set(provided_inputs))
        extra_inputs = list(set(provided_inputs) - set(model_inputs))

        ## Ignoring extra inputs if not necessary
        if not set(provided_inputs).issubset(set(model_inputs)):
            self.visualizer.warning(f'The complete model inputs are {model_inputs}, the provided input are {provided_inputs}. Ignoring {extra_inputs}...')
            for key in extra_inputs:
                del inputs[key]
            provided_inputs = list(inputs.keys())

        ## Determine the Maximal number of samples that can be created
        if sampled:
            min_dim_ind, min_dim  = argmin_min([len(inputs[key]) for key in provided_inputs])
            max_dim_ind, max_dim = argmax_max([len(inputs[key]) for key in provided_inputs])
        else:
            min_dim_ind, min_dim = argmin_min([len(inputs[key])-self.input_n_samples[key]+1 for key in provided_inputs])
            max_dim_ind, max_dim  = argmax_max([len(inputs[key])-self.input_n_samples[key]+1 for key in provided_inputs])
        window_dim = min_dim
        check(window_dim > 0, StopIteration, f'Missing {abs(min_dim)+1} samples in the input window')

        ## warning the users about different time windows between samples
        if min_dim != max_dim:
            self.visualizer.warning(f'Different number of samples between inputs [MAX {list(provided_inputs)[max_dim_ind]} = {max_dim}; MIN {list(provided_inputs)[min_dim_ind]} = {min_dim}]')

        ## Autofill the missing inputs
        if missing_inputs:
            self.visualizer.warning(f'Inputs not provided: {missing_inputs}. Autofilling with zeros..')
            for key in missing_inputs:
                inputs[key] = np.zeros(shape=(window_dim, self.model_def['Inputs'][key]['dim']), dtype=np.float32)

        result_dict = {} ## initialize the resulting dictionary
        for key in self.model_def['Outputs'].keys():
            result_dict[key] = []

        ## Cycle through all the samples provided
        for i in range(window_dim):
            X = {}
            for key, val in inputs.items():
                if key in model_inputs:
                    if sampled:
                        X[key] = torch.from_numpy(np.array(val[i])).to(torch.float32)
                    else:
                        X[key] = torch.from_numpy(np.array(val[i:i+self.input_n_samples[key]])).to(torch.float32)
                        
                    input_dim = self.model_def['Inputs'][key]['dim']
                    if X[key].ndim >= 2:
                        check(X[key].shape[1] == input_dim, ValueError, 'The second dimension must be equal to the input dimension')

                    if input_dim == 1: ## add the input dimension
                        X[key] = X[key].unsqueeze(-1)
                    if X[key].ndim <= 2: ## add the batch dimension
                        X[key] = X[key].unsqueeze(0)

            ## Model Predict            
            result, _  = self.model(X)

            ## Append the prediction of the current sample to the result dictionary
            for key in self.model_def['Outputs'].keys():
                if result[key].shape[-1] == 1:
                    result[key] = result[key].squeeze(-1)
                    if result[key].shape[-1] == 1: 
                        result[key] = result[key].squeeze(-1)
                result_dict[key].append(result[key].detach().squeeze(dim=0).tolist())

        return result_dict

    def get_random_samples(self, window=1):
        if self.data_loaded:
            result_dict = {}
            for key in self.model_def['Inputs'].keys():
                result_dict[key] = []
            random_idx = random.randint(0, self.num_of_samples - window)
            for idx in range(window):
                for key ,samples in self.data.items():
                    if key in self.model_def['Inputs'].keys():
                        result_dict[key].append(samples[random_idx+idx])
            return result_dict
        else:
            print('The Dataset must first be loaded using <loadData> function!')
            return {}


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
        B = (streamB.name[0] if type(streamB.name) is tuple else streamB.name)
        self.minimize_list.append((A, B, loss_function))
        self.minimize_dict[variable_name]={'A':(A, copy.deepcopy(streamA)), 'B':(B, copy.deepcopy(streamB)), 'loss':loss_function}
        self.visualizer.showMinimizeError(variable_name)


    def neuralizeModel(self, sample_time = 1):

        check(sample_time > 0, RuntimeError, 'Sample time must be strictly positive!')
        self.model_def["SampleTime"] = sample_time
        self.visualizer.showModel()

        check(self.model_def['Inputs'] != {}, RuntimeError, "No model is defined!")

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

        ## Adjust with the correct slicing
        for _, items in self.model_def['Relations'].items():
            if items[0] == 'SamplePart':
                if items[1][0] in self.model_def['Inputs'].keys():
                    items[2][0] = self.input_ns_backward[items[1][0]] + items[2][0]
                    items[2][1] = self.input_ns_backward[items[1][0]] + items[2][1]
                    if len(items) > 3: ## Offset
                        items[3] = self.input_ns_backward[items[1][0]] + items[3]
            if items[0] == 'TimePart':
                if items[1][0] in self.model_def['Inputs'].keys():
                    items[2][0] = self.input_ns_backward[items[1][0]] + round(items[2][0]/sample_time)
                    items[2][1] = self.input_ns_backward[items[1][0]] + round(items[2][1]/sample_time)
                    if len(items) > 3: ## Offset
                        items[3] = self.input_ns_backward[items[1][0]] + round(items[3]/sample_time)
                else:
                    items[2][0] = round(items[2][0]/sample_time)
                    items[2][1] = round(items[2][1]/sample_time)
                    if len(items) > 3: ## Offset
                        items[3] = round(items[3]/sample_time)

        ## Build the network
        self.model = Model(self.model_def, self.minimize_list)
        self.visualizer.showBuiltModel()
        self.neuralized = True


    def loadData(self, source, format=None, skiplines=0, delimiter=',', header='infer'):
        assert self.neuralized == True, "The network is not neuralized yet."
        check(delimiter in ['\t', '\n', ';', ',', ' '], ValueError, 'delimiter not valid!')

        model_inputs = list(self.model_def['Inputs'].keys())
        ## Initialize the dictionary containing the data
        self.data = {}

        if type(source) is str: ## we have a directory path containing the files
            ## Initialize each input key
            for key in format:
                if key in model_inputs:
                    self.data[key] = []

            ## obtain the file names
            _,_,files = next(os.walk(source))
            self.file_count = len(files)

            ## Cycle through all the files
            for file in files:
                ## read the csv
                df = pd.read_csv(os.path.join(source,file), skiprows=skiplines, delimiter=delimiter, header=header)
                ## Cycle through all the windows
                start_cols = 0
                for key in format:
                    if key not in model_inputs:
                        start_cols += 1
                        continue
                    key_cols = self.model_def['Inputs'][key]['dim']
                    back, forw = self.input_ns_backward[key], self.input_ns_forward[key]
                    for i in range(self.max_samples_backward, len(df)-self.max_samples_forward+1):
                        ## Save as torch tensors the data
                        self.data[key].append(df.iloc[i-back:i+forw , start_cols:start_cols+key_cols].to_numpy())
                    start_cols += key_cols

            ## Stack the files
            self.num_of_samples = None
            for key in format:
                if key in model_inputs:
                    self.data[key] = np.stack(self.data[key])
                    if self.num_of_samples is None:
                        self.num_of_samples = self.data[key].shape[0]

        elif type(source) is dict:  ## we have a crafted dataset
            self.file_count = 1

            ## Check if the inputs are correct
            assert set(model_inputs).issubset(source.keys()), f'The dataset is missing some inputs. Inputs needed for the model: {model_inputs}'

            for key in model_inputs:
                self.data[key] = []  ## Initialize the dataset

                back, forw = self.input_ns_backward[key], self.input_ns_forward[key]
                for idx in range(len(source[key]) - self.max_n_samples+1):
                    self.data[key].append(source[key][idx + (self.max_samples_backward - back):idx + (self.max_samples_backward + forw)])

            ## Stack the files
            self.num_of_samples = None
            for key in model_inputs:
                self.data[key] = np.stack(self.data[key])
                if self.data[key].ndim == 2: ## Add the sample dimension
                    self.data[key] = np.expand_dims(self.data[key], axis=-1)
                if self.data[key].ndim > 3:
                    self.data[key] = np.squeeze(self.data[key], axis=1)
                if self.num_of_samples is None:
                    self.num_of_samples = self.data[key].shape[0]

        self.visualizer.showDataset()
        ## Set the Loaded flag to True
        self.data_loaded = True

    '''
    ## TODO: Rebuild the function to make it work with vectorial data files
    def loadData(self, source, format=None, skiplines = 0, delimiters=['\t',';',',']):
        assert self.neuralized == True, "The network is not neuralized yet."

        if type(source) is str: ## we have a file path
            _, _, files = next(os.walk(source))
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
    '''

    def __getTrainParams(self, training_params, train_size, test_size):
        if bool(training_params):
            self.learning_rate = (training_params['learning_rate'] if 'learning_rate' in training_params else self.learning_rate)
            self.num_of_epochs = (training_params['num_of_epochs'] if 'num_of_epochs' in training_params else self.num_of_epochs)
            self.train_batch_size = (training_params['train_batch_size'] if 'train_batch_size' in training_params else self.train_batch_size)
            self.test_batch_size = (training_params['test_batch_size'] if 'test_batch_size' in training_params else self.test_batch_size)

            ## Check if the batch_size can be used for the current dataset, otherwise set the batch_size to 1
            if self.train_batch_size > round(self.num_of_samples*train_size):
                self.train_batch_size = 1
            if self.test_batch_size > round(self.num_of_samples*test_size):
                self.test_batch_size = 1
    
    ## TODO: Adjust the Plotting function
    def resultAnalysis(self, train_losses, test_losses, XY_train, XY_test):
        with torch.inference_mode():

            self.model.eval()
            A = torch.zeros(len(self.minimize_dict), self.n_samples_test)
            B = torch.zeros(len(self.minimize_dict), self.n_samples_test)
            aux_test_losses = np.zeros([len(self.minimize_dict), self.n_samples_test])
            for i in range(self.n_samples_test):

                idx = i * self.test_batch_size
                XY = {}
                for key, val in XY_test.items():
                    XY[key] = torch.from_numpy(val[idx:idx + self.test_batch_size]).to(torch.float32)
                    if XY[key].ndim == 2:
                        XY[key] = XY[key].unsqueeze(-1)

                _, minimize_out = self.model(XY)
                for ind, (name, items) in enumerate(self.minimize_dict.items()):
                    A[ind][i] = minimize_out[items['A'][0]]
                    B[ind][i] =  minimize_out[items['B'][0]]
                    loss = self.losses[name](minimize_out[items['A'][0]], minimize_out[items['B'][0]])
                    aux_test_losses[ind][i] = loss.detach().numpy()


            # self.model.eval()
            # key = list(XY_test.keys())[0]
            # samples = len(XY_test[key])
            # #samples = self.n_samples_test
            # #samples = 1
            #
            # #batch_size = int(len(XY_test['x']) / samples)
            # aux_test_losses = np.zeros([len(self.minimize_dict), samples])
            # A = torch.zeros(len(self.minimize_dict), samples)
            # B = torch.zeros(len(self.minimize_dict), samples)
            # for i in range(samples):
            #
            #     XY = {}
            #     for key, val in XY_test.items():
            #         XY[key] = torch.from_numpy(val[i]).to(torch.float32).unsqueeze(dim=0)
            #
            #     # idx = i * batch_size
            #     # XY = {}
            #     # for key, val in XY_test.items():
            #     #     XY[key] = torch.from_numpy(val[idx:idx + batch_size]).to(torch.float32)
            #     out = self.model(XY)
            #     for ind, (key, value) in enumerate(self.minimize_dict.items()):
            #         A[ind][i] = out[value['A'][0]]
            #         B[ind][i] = out[value['B'][0]]
            #         loss = self.loss_fn(A[ind][i], B[ind][i])
            #         aux_test_losses[ind][i] = loss.detach().numpy()


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


    def trainModel(self, test_percentage = 0, training_params = {}):
        if not self.data_loaded:
            print('There is no data loaded! The Training will stop.')
            return
        if not list(self.model.parameters()):
            print('There are no modules with learnable parameters! The Training will stop.')
            return
         
        import time

        # Check input
        train_size = 1 - (test_percentage / 100.0)
        test_size = 1 - train_size
        self.__getTrainParams(training_params, train_size=train_size, test_size=test_size)

        ## Split train and test
        XY_train, XY_test = {}, {}
        self.n_samples_test, self.n_samples_train = None, None
        for key,samples in self.data.items():
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

        ## define optimizer and loss functions
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for name, values in self.minimize_dict.items():
            self.losses[name] = CustomLoss(values['loss'])

        ## Create the train and test loss dictionaries
        train_losses, test_losses = {}, {}
        for key in self.minimize_dict.keys():
            train_losses[key] = []
            test_losses[key] = []

        ## start the train timer
        start = time.time()

        for epoch in range(self.num_of_epochs):
            ## TRAIN
            self.model.train()
            aux_train_losses = torch.zeros([len(self.minimize_dict),self.n_samples_train])
            for i in range(self.n_samples_train):
                idx = i*self.train_batch_size
                XY = {}
                for key, val in XY_train.items():
                    XY[key] = torch.from_numpy(val[idx:idx+self.train_batch_size]).to(torch.float32)

                ## Reset gradient
                self.optimizer.zero_grad()
                ## Model Forward
                _, minimize_out = self.model(XY)
                ## Loss Calculation
                for ind, (name, items) in enumerate(self.minimize_dict.items()):
                    loss = self.losses[name](minimize_out[items['A'][0]], minimize_out[items['B'][0]])
                    loss.backward(retain_graph=True)
                    aux_train_losses[ind][i]= loss.item()
                ## Gradient step
                self.optimizer.step()

            for ind, key in enumerate(self.minimize_dict.keys()):
                train_losses[key].append(torch.mean(aux_train_losses[ind]).tolist())

            if test_percentage != 0:
                ## TEST
                self.model.eval()
                aux_test_losses = torch.zeros(len(self.minimize_dict), self.n_samples_test)
                for i in range(self.n_samples_test):

                    idx = i * self.test_batch_size
                    XY = {}
                    for key, val in XY_test.items():
                        XY[key] = torch.from_numpy(val[idx:idx + self.test_batch_size]).to(torch.float32)
                    ## Model Forward
                    _, minimize_out = self.model(XY)
                    ## Test Loss
                    for ind, (name, items) in enumerate(self.minimize_dict.items()):
                        loss = self.losses[name](minimize_out[items['A'][0]], minimize_out[items['B'][0]])
                        aux_test_losses[ind][i]= loss.item()

                for ind, key in enumerate(self.minimize_dict.keys()):
                    test_losses[key].append(torch.mean(aux_test_losses[ind]).tolist())
            self.visualizer.showTraining(epoch, train_losses, test_losses)
        end = time.time()

        self.visualizer.showTrainingTime(end-start)
        #self.resultAnalysis(train_losses, test_losses, XY_train, XY_test)

    '''
    def trainModel(self, test_percentage = 0, training_params = {}):
        if not list(self.model.parameters()):
            print('There are no modules with learnable parameters! The Training will stop.')
            return
         
        import time

        # Check input
        train_size = 1 - (test_percentage / 100.0)
        test_size = 1 - train_size
        self.__getTrainParams(training_params, train_size=train_size, test_size=test_size)

        ## Split train and test
        XY_train, XY_test = {}, {}
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

        ## define optimizer and loss functions
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for name, values in self.minimize_dict.items():
            self.losses[name] = CustomLoss(values['loss'])

        ## Create the train and test loss dictionaries
        train_losses, test_losses = {}, {}
        for key in self.minimize_dict.keys():
            train_losses[key] = []
            test_losses[key] = []

        ## start the train timer
        start = time.time()

        for epoch in range(self.num_of_epochs):
            ## TRAIN
            self.model.train()
            aux_train_losses = torch.zeros([len(self.minimize_dict),self.n_samples_train])
            for i in range(self.n_samples_train):
                idx = i*self.train_batch_size
                XY = {}
                for key, val in XY_train.items():
                    XY[key] = torch.from_numpy(val[idx:idx+self.train_batch_size]).to(torch.float32)
                    if XY[key].ndim == 2:
                        XY[key] = XY[key].unsqueeze(-1)
                self.optimizer.zero_grad()
                
                ## Model Forward
                _, minimize_out = self.model(XY)

                for ind, (name, items) in enumerate(self.minimize_dict.items()):
                    loss = self.losses[name](minimize_out[items['A'][0]], minimize_out[items['B'][0]])
                    loss.backward(retain_graph=True)
                    aux_train_losses[ind][i]= loss.item()

                self.optimizer.step()

            for ind, key in enumerate(self.minimize_dict.keys()):
                train_losses[key].append(torch.mean(aux_train_losses[ind]).tolist())

            if test_percentage != 0:
                ## TEST
                self.model.eval()
                aux_test_losses = torch.zeros(len(self.minimize_dict), self.n_samples_test)
                for i in range(self.n_samples_test):

                    idx = i * self.test_batch_size
                    XY = {}
                    for key, val in XY_test.items():
                        XY[key] = torch.from_numpy(val[idx:idx + self.test_batch_size]).to(torch.float32)
                        if XY[key].ndim == 2:
                            XY[key] = XY[key].unsqueeze(-1)

                    ## Model Forward
                    _, minimize_out = self.model(XY)

                    for ind, (name, items) in enumerate(self.minimize_dict.items()):
                        loss = self.losses[name](minimize_out[items['A'][0]], minimize_out[items['B'][0]])
                        aux_test_losses[ind][i]= loss.item()

                for ind, key in enumerate(self.minimize_dict.keys()):
                    test_losses[key].append(torch.mean(aux_test_losses[ind]).tolist())
            self.visualizer.showTraining(epoch, train_losses, test_losses)
        end = time.time()

        self.visualizer.showTrainingTime(end-start)
        #self.resultAnalysis(train_losses, test_losses, XY_train, XY_test)
    '''
    def trainRecurrentModel(self, close_loop, prediction_horizon=None, step=1, test_percentage = 0, training_params = {}):
        if not self.data_loaded:
            print('There is no data loaded! The Training will stop.')
            return
        if not list(self.model.parameters()):
            print('There are no modules with learnable parameters! The Training will stop.')
            return
        
        import time

        ## Calculate the Prediction Horizon
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
        self.n_samples_test, self.n_samples_train = None, None
        for key,samples in self.data.items():
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

        ## define optimizer and loss functions
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for name, values in self.minimize_dict.items():
            self.losses[name] = CustomLoss(values['loss'])

        ## initialize the train and test loss dictionaries
        train_losses, test_losses = {}, {}
        for key in self.minimize_dict.keys():
            train_losses[key] = np.zeros(self.num_of_epochs)
            test_losses[key] = np.zeros(self.num_of_epochs)

        ## start the training timer
        start = time.time()

        for epoch in range(self.num_of_epochs):
            ## TRAIN
            self.model.train()
            train_loss = []
            for i in range(0, (self.n_samples_train - prediction_samples), step):
                idx = i*self.train_batch_size
                XY = {}
                XY_horizon = {}
                for key, val in XY_train.items():
                    XY[key] = torch.from_numpy(val[idx:idx+self.train_batch_size]).to(torch.float32)
                    ## collect the horizon labels
                    XY_horizon[key] = torch.from_numpy(val[idx+1:idx+self.train_batch_size+prediction_samples+1]).to(torch.float32)

                self.optimizer.zero_grad()  
                losses = []
                ## Recurrent Training
                for horizon_idx in range(prediction_samples):
                    ## Model Forward
                    out, minimize_out = self.model(XY)
                    
                    for name, items in self.minimize_dict.items():
                        loss = self.losses[name](minimize_out[items['A'][0]], minimize_out[items['B'][0]])
                        losses.append(loss)
                    
                    ## Update the input with the recurrent prediction
                    for key in XY.keys():
                        XY[key] = torch.roll(XY[key], shifts=-1, dims=1)
                        if key in close_loop.keys():
                            XY[key][:, -1, :] = out[close_loop[key]][:, -1, :]
                        else:
                            XY[key][:, -1, :] = XY_horizon[key][horizon_idx:horizon_idx+self.train_batch_size, -1, :]

                loss = sum(losses)
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
            train_loss = np.mean(train_loss)
            train_losses[epoch] = train_loss

            if test_percentage != 0:
                ## TEST
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
                    ## Recurrent Training
                    for horizon_idx in range(prediction_samples):
                        ## Model Forward
                        out, minimize_out = self.model(XY)
                        
                        for name, items in self.minimize_dict.items():
                            loss = self.losses[name](minimize_out[items['A'][0]], minimize_out[items['B'][0]])
                            losses.append(loss)
                        
                        ## Update the input with the recurrent prediction
                        for key in XY.keys():
                            XY[key] = torch.roll(XY[key], shifts=-1, dims=1)
                            if key in close_loop.keys():
                                XY[key][:, -1, :] = out[close_loop[key]][:, -1, :]
                            else:
                                XY[key][:, -1, :] = XY_horizon[key][horizon_idx:horizon_idx+self.test_batch_size, -1, :]

                    loss = sum(losses) / prediction_samples
                    test_loss.append(loss.item())
                test_loss = np.mean(test_loss)
                test_losses[epoch] = test_loss

            self.visualizer.showTraining(epoch, train_losses, test_losses)

        end = time.time()
        self.visualizer.showTrainingTime(end - start)
        #self.resultAnalysis(train_losses=train_loss, test_losses=test_loss, XY_train=XY_train, XY_test=XY_test)
    '''
    def trainRecurrentModel(self, close_loop, prediction_horizon=None, step=1, test_percentage = 0, training_params = {}):
        if not list(self.model.parameters()):
            print('There are no modules with learnable parameters! The Training will stop.')
            return
        
        import time

        ## Calculate the Prediction Horizon
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

        ## define optimizer and loss functions
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for name, values in self.minimize_dict.items():
            self.losses[name] = CustomLoss(values['loss'])

        ## initialize the train and test loss dictionaries
        train_losses, test_losses = {}, {}
        for key in self.minimize_dict.keys():
            train_losses[key] = np.zeros(self.num_of_epochs)
            test_losses[key] = np.zeros(self.num_of_epochs)

        ## start the training timer
        start = time.time()

        for epoch in range(self.num_of_epochs):
            ## TRAIN
            self.model.train()
            train_loss = []
            for i in range(0, (self.n_samples_train - prediction_samples), step):
                idx = i*self.train_batch_size
                XY = {}
                XY_horizon = {}
                for key, val in XY_train.items():
                    XY[key] = torch.from_numpy(val[idx:idx+self.train_batch_size]).to(torch.float32)
                    ## collect the horizon labels
                    XY_horizon[key] = torch.from_numpy(val[idx+1:idx+self.train_batch_size+prediction_samples+1]).to(torch.float32)
                    if XY[key].ndim == 2:
                        XY[key] = XY[key].unsqueeze(-1)
                        XY_horizon[key] = XY_horizon[key].unsqueeze(-1)

                self.optimizer.zero_grad()  
                losses = []
                ## Recurrent Training
                for horizon_idx in range(prediction_samples):
                    ## Model Forward
                    out, minimize_out = self.model(XY)
                    
                    for name, items in self.minimize_dict.items():
                        loss = self.losses[name](minimize_out[items['A'][0]], minimize_out[items['B'][0]])
                        losses.append(loss)
                    
                    ## Update the input with the recurrent prediction
                    for key in XY.keys():
                        XY[key] = torch.roll(XY[key], shifts=-1, dims=1)
                        if key in close_loop.keys():
                            XY[key][:, -1, :] = out[close_loop[key]][:, -1, :]
                        else:
                            XY[key][:, -1, :] = XY_horizon[key][horizon_idx:horizon_idx+self.train_batch_size, -1, :]

                loss = sum(losses)
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
            train_loss = np.mean(train_loss)
            train_losses[epoch] = train_loss

            if test_percentage != 0:
                ## TEST
                self.model.eval()
                test_loss = []
                for i in range(0, (self.n_samples_test - prediction_samples), step):
                    idx = i*self.test_batch_size
                    XY = {}
                    XY_horizon = {}
                    for key, val in XY_test.items():
                        XY[key] = torch.from_numpy(val[idx:idx+self.test_batch_size]).to(torch.float32)
                        XY_horizon[key] = torch.from_numpy(val[idx+1:idx+self.test_batch_size+prediction_samples+1]).to(torch.float32)
                        if XY[key].ndim == 2:
                            XY[key] = XY[key].unsqueeze(-1)
                            XY_horizon[key] = XY_horizon[key].unsqueeze(-1)

                    losses = []
                    ## Recurrent Training
                    for horizon_idx in range(prediction_samples):
                        ## Model Forward
                        out, minimize_out = self.model(XY)
                        
                        for name, items in self.minimize_dict.items():
                            loss = self.losses[name](minimize_out[items['A'][0]], minimize_out[items['B'][0]])
                            losses.append(loss)
                        
                        ## Update the input with the recurrent prediction
                        for key in XY.keys():
                            XY[key] = torch.roll(XY[key], shifts=-1, dims=1)
                            if key in close_loop.keys():
                                XY[key][:, -1, :] = out[close_loop[key]][:, -1, :]
                            else:
                                XY[key][:, -1, :] = XY_horizon[key][horizon_idx:horizon_idx+self.test_batch_size, -1, :]

                    loss = sum(losses) / prediction_samples
                    test_loss.append(loss.item())
                test_loss = np.mean(test_loss)
                test_losses[epoch] = test_loss

            self.visualizer.showTraining(epoch, train_losses, test_losses)

        end = time.time()
        self.visualizer.showTrainingTime(end - start)
        #self.resultAnalysis(train_losses=train_loss, test_losses=test_loss, XY_train=XY_train, XY_test=XY_test)
    '''
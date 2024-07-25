import copy

import torch

import numpy as np
import pandas as pd
from scipy.stats import norm
import random
import os
from pprint import pprint
from pprint import pformat
import re
import matplotlib.pyplot as plt

from neu4mes.relation import NeuObj, merge
from neu4mes.visualizer import TextVisualizer, Visualizer
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

        # Network Parametrs
        self.minimize_list = []
        self.minimize_dict = {}
        self.input_tw_backward, self.input_tw_forward = {}, {}
        self.input_ns_backward, self.input_ns_forward = {}, {}
        self.input_n_samples = {}
        self.max_samples_backward, self.max_samples_forward = 0, 0
        self.max_n_samples = 0
        self.neuralized = False
        self.model = None

        # Dataaset Parameters
        self.data_loaded = False
        self.file_count = 0
        self.num_of_samples = {}
        self.data = {}
        self.n_datasets = 0
        self.datasets_loaded = set()

        # Training Parameters
        self.learning_rate = 0.01
        self.num_of_epochs = 100
        self.train_batch_size, self.val_batch_size, self.test_batch_size = 1, 1, 1
        self.n_samples_train, self.n_samples_val, self.n_samples_test = None, None, None
        self.n_samples_horizon = None
        self.optimizer = None
        self.losses = {}

        # Validation Parameters
        self.performance = {}
        self.prediction = {}


    def __call__(self, inputs, sampled=False):
        check(self.neuralized, ValueError, "The network is not neuralized.")
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
                    if input_dim > 1:
                        check(len(X[key].shape) == 2, ValueError,
                              f'The input {key} must have two dimensions')
                        check(X[key].shape[1] == input_dim, ValueError,
                              f'The second dimension of the input "{key}" must be equal to {input_dim}')

                    if input_dim == 1 and X[key].shape[-1] != 1: ## add the input dimension
                        X[key] = X[key].unsqueeze(-1)
                    if X[key].ndim <= 1: ## add the batch dimension
                        X[key] = X[key].unsqueeze(0)
                    if X[key].ndim <= 2: ## add the time dimension
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

    def get_random_samples(self, dataset, window=1):
        if self.data_loaded:
            result_dict = {}
            for key in self.model_def['Inputs'].keys():
                result_dict[key] = []
            random_idx = random.randint(0, self.num_of_samples[dataset] - window)
            for idx in range(window):
                for key ,samples in self.data[dataset].items():
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
        #model_def_final = copy.deepcopy(self.model_def)
        self.visualizer.showModel()

        check(self.model_def['Inputs'] != {}, RuntimeError, "No model is defined!")
        json_inputs = self.model_def['Inputs'] | self.model_def['States']

        for key,value in self.model_def['States'].items():
            check('update' in self.model_def['States'][key], RuntimeError, f'Update function is missing for state {key}. Call X.update({key}) on a Stream X.')

        for key, value in json_inputs.items():
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

        #self.visualizer.showModel()
        ## Build the network
        self.model = Model(copy.deepcopy(self.model_def), self.minimize_list, self.input_ns_backward)
        self.visualizer.showBuiltModel()
        self.neuralized = True


    def loadData(self, name, source, format=None, skiplines=0, delimiter=',', header=None):
        assert self.neuralized == True, "The network is not neuralized yet."
        check(delimiter in ['\t', '\n', ';', ',', ' '], ValueError, 'delimiter not valid!')

        json_inputs = self.model_def['Inputs'] | self.model_def['States']
        model_inputs = list(json_inputs.keys())
        ## Initialize the dictionary containing the data
        if name in list(self.data.keys()):
            self.visualizer.warning(f'Dataset named {name} already loaded! overriding the existing one..')
        self.data[name] = {}

        if type(source) is str: ## we have a directory path containing the files
            ## collect column indexes
            format_idx = {}
            idx = 0
            for item in format:
                if isinstance(item, tuple):
                    for key in item:
                        if key not in model_inputs:
                            idx += 1
                            break
                        n_cols = json_inputs[key]['dim']
                        format_idx[key] = (idx, idx+n_cols)
                    idx += n_cols
                else:
                    if item not in model_inputs:
                        idx += 1
                        continue
                    n_cols = json_inputs[item]['dim']
                    format_idx[item] = (idx, idx+n_cols)
                    idx += n_cols

            ## Initialize each input key
            for key in format_idx.keys():
                self.data[name][key] = []

            ## obtain the file names
            try:
                _,_,files = next(os.walk(source))
            except StopIteration as e:
                print(f'ERROR: The path "{source}" does not exist!')
                return
            self.file_count = len(files)

            ## Cycle through all the files
            for file in files:
                try:
                    ## read the csv
                    df = pd.read_csv(os.path.join(source,file), skiprows=skiplines, delimiter=delimiter, header=header)
                except:
                    self.visualizer.warning(f'Cannot read file {os.path.join(source,file)}')
                    continue
                ## Cycle through all the windows
                for key, idxs in format_idx.items():
                    back, forw = self.input_ns_backward[key], self.input_ns_forward[key]
                    ## Save as numpy array the data
                    data = df.iloc[:, idxs[0]:idxs[1]].to_numpy()
                    self.data[name][key] += [data[i-back:i+forw] for i in range(self.max_samples_backward, len(df)-self.max_samples_forward+1)]

            ## Stack the files
            self.num_of_samples[name] = None
            for key in format_idx.keys():
                self.data[name][key] = np.stack(self.data[name][key])

            ## save the number of samples
            if self.num_of_samples[name] is None:
                self.num_of_samples[name] = list(self.data[name].values())[0].shape[0]

        elif type(source) is dict:  ## we have a crafted dataset
            self.file_count = 1

            ## Check if the inputs are correct
            assert set(model_inputs).issubset(source.keys()), f'The dataset is missing some inputs. Inputs needed for the model: {model_inputs}'

            for key in model_inputs:
                self.data[name][key] = []  ## Initialize the dataset

                back, forw = self.input_ns_backward[key], self.input_ns_forward[key]
                for idx in range(len(source[key]) - self.max_n_samples+1):
                    self.data[name][key].append(source[key][idx + (self.max_samples_backward - back):idx + (self.max_samples_backward + forw)])

            ## Stack the files
            self.num_of_samples[name] = None
            for key in model_inputs:
                self.data[name][key] = np.stack(self.data[name][key])
                if self.data[name][key].ndim == 2: ## Add the sample dimension
                    self.data[name][key] = np.expand_dims(self.data[name][key], axis=-1)
                if self.data[name][key].ndim > 3:
                    self.data[name][key] = np.squeeze(self.data[name][key], axis=1)
                if self.num_of_samples[name] is None:
                    self.num_of_samples[name] = self.data[name][key].shape[0]

        ## Set the Loaded flag to True
        self.data_loaded = True
        ## Update the number of datasets loaded
        self.n_datasets = len(self.data.keys())
        self.datasets_loaded.add(name)
        ## Show the dataset
        self.visualizer.showDataset(name=name)

    def __getTrainParams(self, training_params):
        # Set all parameters in training_params
        for key,value in training_params.items():
            try:
                getattr(self, key)
                setattr(self, key, value)
            except:
                raise KeyError(f"The training_params contains a wrong key: {key}.")

        if self.train_batch_size > self.n_samples_train:
            self.train_batch_size = 1
        if self.val_batch_size > self.n_samples_val:
            self.val_batch_size = 1
        if self.test_batch_size > self.n_samples_test:
            self.test_batch_size = 1

    # def resultAnalysis(self, name_data, losses, XY_data, n_samples, batch_size):
    #     with torch.inference_mode():
    #
    #         self.model.eval()
    #         A = {}
    #         B = {}
    #         aux_losses = {}
    #         for (name, items) in self.minimize_dict.items():
    #             window = 'tw' if 'tw' in items['A'][1].dim else ('sw' if 'sw' in items['A'][1].dim else None)
    #             A[name] = torch.zeros([n_samples, batch_size, items['A'][1].dim[window], items['A'][1].dim['dim']])
    #             B[name] = torch.zeros([n_samples, batch_size, items['B'][1].dim[window], items['B'][1].dim['dim']])
    #             aux_losses[name] = np.zeros(
    #                 [n_samples, batch_size, items['A'][1].dim[window], items['A'][1].dim['dim']])
    #
    #         for i in range(n_samples):
    #
    #             idx = i * batch_size
    #             XY = {}
    #             for key, val in XY_data.items():
    #                 XY[key] = val[idx:idx + batch_size]
    #                 # if XY[key].ndim == 2:
    #                 #    XY[key] = XY[key].unsqueeze(-1)
    #
    #             _, minimize_out = self.model(XY)
    #             for ind, (name, items) in enumerate(self.minimize_dict.items()):
    #                 A[name][i] = minimize_out[items['A'][0]]
    #                 B[name][i] = minimize_out[items['B'][0]]
    #                 loss = self.losses[name](minimize_out[items['A'][0]], minimize_out[items['B'][0]])
    #                 aux_losses[name][i] = loss.detach().numpy()
    #
    #         for ind, (key, value) in enumerate(self.minimize_dict.items()):
    #             A_np = A[key].detach().numpy()
    #             B_np = B[key].detach().numpy()
    #             self.performance[key] = {}
    #             self.performance[key][value['loss']] = {'epoch' + name_data: losses[key],
    #                                                     name_data: np.mean(aux_losses[key])}
    #             self.performance[key]['fvu'] = {}
    #             # Compute FVU
    #             residual = A_np - B_np
    #             error_var = np.var(residual)
    #             error_mean = np.mean(residual)
    #             # error_var_manual = np.sum((residual-error_mean) ** 2) / (len(self.prediction['B'][ind]) - 0)
    #             # print(f"{key} var np:{new_error_var} and var manual:{error_var_manual}")
    #             self.performance[key]['fvu']['A'] = (error_var / np.var(A_np)).item()
    #             self.performance[key]['fvu']['B'] = (error_var / np.var(B_np)).item()
    #             self.performance[key]['fvu']['total'] = np.mean(
    #                 [self.performance[key]['fvu']['A'], self.performance[key]['fvu']['B']]).item()
    #             # Compute AIC
    #             # normal_dist = norm(0, error_var ** 0.5)
    #             # probability_of_residual = normal_dist.pdf(residual)
    #             # log_likelihood_first = sum(np.log(probability_of_residual))
    #             p1 = -len(residual) / 2.0 * np.log(2 * np.pi)
    #             p2 = -len(residual) / 2.0 * np.log(error_var)
    #             p3 = -1 / (2.0 * error_var) * np.sum(residual ** 2)
    #             log_likelihood = p1 + p2 + p3
    #             # print(f"{key} log likelihood second mode:{log_likelihood} = {p1}+{p2}+{p3} first mode: {log_likelihood_first}")
    #             total_params = sum(p.numel() for p in self.model.parameters() if
    #                                p.requires_grad)  # TODO to be check the number is doubled
    #             # print(f"{key} total_params:{total_params}")
    #             aic = - 2 * log_likelihood + 2 * total_params
    #             # print(f"{key} aic:{aic}")
    #             self.performance[key]['aic'] = {'value': aic, 'total_params': total_params,
    #                                             'log_likelihood': log_likelihood}
    #             # Prediction and target
    #             self.prediction[key] = {}
    #             self.prediction[key]['A'] = A_np.tolist()
    #             self.prediction[key]['B'] = B_np.tolist()
    #
    #         self.performance['total'] = {}
    #         self.performance['total']['mean_error'] = {name_data: np.mean([value for key, value in aux_losses.items()])}
    #         self.performance['total']['fvu'] = np.mean(
    #             [self.performance[key]['fvu']['total'] for key in self.minimize_dict.keys()])
    #         self.performance['total']['aic'] = np.mean(
    #             [self.performance[key]['aic']['value'] for key in self.minimize_dict.keys()])
    #
    #     self.visualizer.showResults(name_data)

    def resultAnalysis(self, name_data, losses, XY_data):
        with torch.inference_mode():

            self.model.eval()
            A = {}
            B = {}
            aux_losses = {}
            for (name, items) in self.minimize_dict.items():
                window = 'tw' if 'tw' in items['A'][1].dim else ('sw' if 'sw' in items['A'][1].dim else None)
                A[name] = torch.zeros([XY_data[list(XY_data.keys())[0]].shape[0],items['A'][1].dim[window],items['A'][1].dim['dim']])
                B[name] = torch.zeros([XY_data[list(XY_data.keys())[0]].shape[0],items['B'][1].dim[window],items['B'][1].dim['dim']])
                aux_losses[name] = np.zeros([XY_data[list(XY_data.keys())[0]].shape[0],items['A'][1].dim[window],items['A'][1].dim['dim']])

            _, minimize_out = self.model(XY_data)
            for ind, (name, items) in enumerate(self.minimize_dict.items()):
                A[name] = minimize_out[items['A'][0]]
                B[name] = minimize_out[items['B'][0]]
                loss = self.losses[name](minimize_out[items['A'][0]], minimize_out[items['B'][0]])
                aux_losses[name] = loss.detach().numpy()

            for ind, (key, value) in enumerate(self.minimize_dict.items()):
                A_np = A[key].detach().numpy()
                B_np = B[key].detach().numpy()
                self.performance[key] = {}
                self.performance[key][value['loss']] = {'epoch'+name_data: losses[key], name_data: np.mean(aux_losses[key]).item()}
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
            self.performance['total']['mean_error'] = {name_data: np.mean([value for key,value in aux_losses.items()])}
            self.performance['total']['fvu'] = np.mean([self.performance[key]['fvu']['total'] for key in self.minimize_dict.keys()])
            self.performance['total']['aic'] = np.mean([self.performance[key]['aic']['value']for key in self.minimize_dict.keys()])

        self.visualizer.showResults(name_data)

    def trainModel(self, train_dataset=None, validation_dataset=None, test_dataset=None, splits=[70,20,10], prediction_horizon=0, shuffle_data=True, training_params = {}):
        if not self.data_loaded:
            print('There is no data loaded! The Training will stop.')
            return
        if not list(self.model.parameters()):
            print('There are no modules with learnable parameters! The Training will stop.')
            return

        if self.n_datasets == 1: ## If we use 1 dataset with the splits
            check(len(splits)==3, ValueError, '3 elements must be inserted for the dataset split in training, validation and test')
            check(sum(splits)==100, ValueError, 'Training, Validation and Test splits must sum up to 100.')
            check(splits[0] > 0, ValueError, 'The training split cannot be zero.')

            ## Get the dataset name
            dataset = list(self.data.keys())[0] ## take the dataset name
            self.visualizer.warning(f'Only {self.n_datasets} Dataset loaded ({dataset}). The training will continue using \n{splits[0]}% of data as training set \n{splits[1]}% of data as validation set \n{splits[2]}% of data as test set')

            # Collect the split sizes
            train_size = splits[0] / 100.0
            val_size = splits[1] / 100.0
            test_size = 1 - (train_size + val_size)
            self.n_samples_train = round(self.num_of_samples[dataset]*train_size)
            self.n_samples_val = round(self.num_of_samples[dataset]*val_size)
            self.n_samples_test = round(self.num_of_samples[dataset]*test_size)

            ## Split into train, validation and test
            XY_train, XY_val, XY_test = {}, {}, {}
            for key, samples in self.data[dataset].items():
                if val_size == 0.0 and test_size == 0.0: ## we have only training set
                    XY_train[key] = torch.from_numpy(samples).to(torch.float32)
                elif val_size == 0.0 and test_size != 0.0: ## we have only training and test set
                    XY_train[key] = torch.from_numpy(samples[:round(len(samples)*train_size)]).to(torch.float32)
                    XY_test[key] = torch.from_numpy(samples[round(len(samples)*train_size):]).to(torch.float32)
                elif val_size != 0.0 and test_size == 0.0: ## we have only training and validation set
                    XY_train[key] = torch.from_numpy(samples[:round(len(samples)*train_size)]).to(torch.float32)
                    XY_val[key] = torch.from_numpy(samples[round(len(samples)*train_size):]).to(torch.float32)
                else: ## we have training, validation and test set
                    XY_train[key] = torch.from_numpy(samples[:round(len(samples)*train_size)]).to(torch.float32)
                    XY_val[key] = torch.from_numpy(samples[round(len(samples)*train_size):-round(len(samples)*test_size)]).to(torch.float32)
                    XY_test[key] = torch.from_numpy(samples[-round(len(samples)*test_size):]).to(torch.float32)
        else: ## Multi-Dataset
            datasets = list(self.data.keys())

            check(train_dataset in datasets, KeyError, f'{train_dataset} Not Loaded!')
            if validation_dataset not in datasets:
                self.visualizer.warning(f'Validation Dataset [{validation_dataset}] Not Loaded. The training will continue without validation')
            if test_dataset not in datasets:
                self.visualizer.warning(f'Test Dataset [{test_dataset}] Not Loaded. The training will continue without test')

            ## Collect the number of samples for each dataset
            self.n_samples_train, self.n_samples_val, self.n_samples_test = 0, 0, 0
            ## Split into train, validation and test
            self.n_samples_train = self.num_of_samples[train_dataset]
            XY_train = {key: torch.from_numpy(val).to(torch.float32) for key, val in self.data[train_dataset].items()}
            if validation_dataset in datasets:
                self.n_samples_val = self.num_of_samples[validation_dataset]
                XY_val = {key: torch.from_numpy(val).to(torch.float32) for key, val in self.data[validation_dataset].items()}
            if test_dataset in datasets:
                self.n_samples_test = self.num_of_samples[test_dataset]
                XY_test = {key: torch.from_numpy(val).to(torch.float32) for key, val in self.data[test_dataset].items()}

        ## TRAIN MODEL
        ## Check parameters
        self.__getTrainParams(training_params)
        self.n_samples_train = self.n_samples_train//self.train_batch_size
        self.n_samples_val = self.n_samples_val//self.val_batch_size
        self.n_samples_test = self.n_samples_test//self.test_batch_size
        assert self.n_samples_train > 0, f'There are {self.n_samples_train} samples for training.'
        self.n_samples_horizon = round(prediction_horizon // self.model_def['SampleTime'])
        self.visualizer.showTrainParams()


        ## define optimizer and loss functions
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for name, values in self.minimize_dict.items():
            self.losses[name] = CustomLoss(values['loss'])

        ## Create the train, validation and test loss dictionaries
        train_losses, val_losses, test_losses = {}, {}, {}
        for key in self.minimize_dict.keys():
            train_losses[key] = []
            if self.n_samples_val > 0:
                val_losses[key] = []

        import time
        ## start the train timer
        start = time.time()

        for epoch in range(self.num_of_epochs):
            ## TRAIN
            self.model.train()
            ## Sample Shuffle
            if shuffle_data:
                XY_train = {key: val[torch.randperm(val.size(0))] for key, val in XY_train.items()}
            ## Initialize the train losses vector
            aux_train_losses = torch.zeros([len(self.minimize_dict),self.n_samples_train])
            for i in range(self.n_samples_train):
                idx = i*self.train_batch_size
                ## Build the input tensor
                XY = {key: val[idx:idx+self.train_batch_size] for key, val in XY_train.items()}
                ## Reset gradient
                self.optimizer.zero_grad()
                ## Model Forward
                if (self.n_samples_horizon == 0) or (self.n_samples_horizon != 0 and ((i+self.n_samples_horizon)%self.n_samples_horizon == 0)):
                    _, minimize_out = self.model(XY, initialize_state=True)  ## Recurrent Training with state variables
                else:
                    _, minimize_out = self.model(XY, initialize_state=False)  ## Normal Training
                ## Loss Calculation
                for ind, (name, items) in enumerate(self.minimize_dict.items()):
                    loss = self.losses[name](minimize_out[items['A'][0]], minimize_out[items['B'][0]])
                    loss.backward(retain_graph=True)
                    aux_train_losses[ind][i]= loss.item()
                ## Gradient step
                self.optimizer.step()
            ## save the losses
            for ind, key in enumerate(self.minimize_dict.keys()):
                train_losses[key].append(torch.mean(aux_train_losses[ind]).tolist())
            if self.n_samples_val > 0: 
                ## VALIDATION
                self.model.eval()
                aux_val_losses = torch.zeros(len(self.minimize_dict), self.n_samples_val)
                for i in range(self.n_samples_val):
                    idx = i * self.val_batch_size
                    ## Build the input tensor
                    XY = {key: val[idx:idx + self.val_batch_size] for key, val in XY_val.items()}
                    ## Model Forward
                    if (self.n_samples_horizon == 0) or (self.n_samples_horizon != 0 and ((i+self.n_samples_horizon)%self.n_samples_horizon == 0)):
                        _, minimize_out = self.model(XY, initialize_state=True)  ## Recurrent Training with state variables
                    else:
                        _, minimize_out = self.model(XY)
                    ## Validation Loss
                    for ind, (name, items) in enumerate(self.minimize_dict.items()):
                        loss = self.losses[name](minimize_out[items['A'][0]], minimize_out[items['B'][0]])
                        aux_val_losses[ind][i]= loss.item()
                ## save the losses
                for ind, key in enumerate(self.minimize_dict.keys()):
                    val_losses[key].append(torch.mean(aux_val_losses[ind]).tolist())

            ## visualize the training...
            self.visualizer.showTraining(epoch, train_losses, val_losses)

        ## save the training time
        end = time.time()
        ## visualize the training time
        self.visualizer.showTrainingTime(end-start)

        ## Test the model ##TODO adjust the test visualizer
        if self.n_samples_test > 0: 
            ## TEST
            self.model.eval()
            aux_test_losses = torch.zeros(len(self.minimize_dict), self.n_samples_test)
            for i in range(self.n_samples_test):
                idx = i * self.test_batch_size
                ## Build the input tensor
                XY = {key: val[idx:idx + self.test_batch_size] for key, val in XY_test.items()}
                ## Model Forward
                if (self.n_samples_horizon == 0) or (self.n_samples_horizon != 0 and ((i+self.n_samples_horizon)%self.n_samples_horizon == 0)):
                    _, minimize_out = self.model(XY, initialize_state=True)  ## Recurrent Training with state variables
                else:
                    _, minimize_out = self.model(XY)
                ## Test Loss
                for ind, (name, items) in enumerate(self.minimize_dict.items()):
                    loss = self.losses[name](minimize_out[items['A'][0]], minimize_out[items['B'][0]])
                    aux_test_losses[ind][i]= loss.item()
            ## save the losses
            for ind, key in enumerate(self.minimize_dict.keys()):
                test_losses[key] = torch.mean(aux_test_losses[ind]).tolist()

        if self.n_samples_train > 0:
            self.resultAnalysis('Training', train_losses, XY_train)
        if self.n_samples_val > 0:
            self.resultAnalysis('Validation', val_losses, XY_val)
        if self.n_samples_test > 0:
            self.resultAnalysis('Test', test_losses, XY_test)

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
import copy

import torch
from torch.export import export

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
from neu4mes.relation import Stream
from neu4mes.model import Model
from neu4mes.utilis import check, argmax_max, argmin_min


from neu4mes import LOG_LEVEL
from neu4mes.logger import logging
log = logging.getLogger(__name__)
log.setLevel(max(logging.ERROR, LOG_LEVEL))

class Neu4mes:
    name = None
    def __init__(self, visualizer = 'Standard', seed=None):

        # Visualizer
        if visualizer == 'Standard':
            self.visualizer = TextVisualizer(1)
        elif visualizer != None:
            self.visualizer = visualizer
        else:
            self.visualizer = Visualizer()
        self.visualizer.set_n4m(self)

        ## Set the random seed for reproducibility
        if seed:
            torch.manual_seed(seed=seed) ## set the pytorch seed
            random.seed(seed) ## set the random module seed
            np.random.seed(seed) ## set the numpy seed

        # Inizialize the model definition
        self.stream_dict = {}
        self.minimize_dict = {}
        self.model_def = NeuObj().json

        # Network Parametrs
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
        self.weight_decay = 0.0
        self.train_batch_size, self.val_batch_size, self.test_batch_size = 1, 1, 1
        self.n_samples_train, self.n_samples_test, self.n_samples_val = None, None, None
        self.optimizer = None
        self.losses = {}
        self.close_loop = None
        self.prediction_samples = 1

        # Validation Parameters
        self.performance = {}
        self.prediction = {}


    def __call__(self, inputs={}, sampled=False, close_loop={}, connect={}, prediction_samples=1):
        check(self.neuralized, ValueError, "The network is not neuralized.")

        close_loop_windows = {}
        for close_in, close_out in close_loop.items():
            check(close_in in self.model_def['Inputs'], ValueError, f'the tag {close_in} is not an input variable.')
            check(close_out in self.model_def['Outputs'], ValueError, f'the tag {close_out} is not an output of the network')
            if close_in in inputs.keys():
                close_loop_windows[close_in] = len(inputs[close_in]) if sampled else len(inputs[close_in])-self.input_n_samples[close_in]+1
            else:
                close_loop_windows[close_in] = 1

        model_inputs = list(self.model_def['Inputs'].keys())
        model_states = list(self.model_def['States'].keys())
        provided_inputs = list(inputs.keys())
        missing_inputs = list(set(model_inputs) - set(provided_inputs) - set(connect.keys()))
        extra_inputs = list(set(provided_inputs) - set(model_inputs))

        for key in model_states:
            if key in inputs.keys():
                close_loop_windows[key] = len(inputs[key]) if sampled else len(inputs[key])-self.input_n_samples[key]+1
            else:
                close_loop_windows[key] = 1

        ## Ignoring extra inputs if not necessary
        if not set(provided_inputs).issubset(set(model_inputs) | set(model_states)):
            self.visualizer.warning(f'The complete model inputs are {model_inputs}, the provided input are {provided_inputs}. Ignoring {extra_inputs}...')
            for key in extra_inputs:
                del inputs[key]
            provided_inputs = list(inputs.keys())
        non_recurrent_inputs = list(set(provided_inputs) - set(close_loop.keys()) - set(model_states) - set(connect.keys()))

        ## Determine the Maximal number of samples that can be created
        if non_recurrent_inputs:
            if sampled:
                min_dim_ind, min_dim  = argmin_min([len(inputs[key]) for key in non_recurrent_inputs])
                max_dim_ind, max_dim = argmax_max([len(inputs[key]) for key in non_recurrent_inputs])
            else:
                min_dim_ind, min_dim = argmin_min([len(inputs[key])-self.input_n_samples[key]+1 for key in non_recurrent_inputs])
                max_dim_ind, max_dim  = argmax_max([len(inputs[key])-self.input_n_samples[key]+1 for key in non_recurrent_inputs])
        else:
            if provided_inputs:
                min_dim_ind, min_dim  = argmin_min([close_loop_windows[key]+prediction_samples-1 for key in provided_inputs])
                max_dim_ind, max_dim = argmax_max([close_loop_windows[key]+prediction_samples-1 for key in provided_inputs])
            else:
                min_dim = max_dim = prediction_samples

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
        with torch.inference_mode():
            X = {}
            for i in range(window_dim):
                for key, val in inputs.items():
                    if key in close_loop.keys() or key in model_states:
                        if i >= close_loop_windows[key]:
                            if key in model_states and key in X.keys():
                                del X[key]
                            continue

                    ## Collect the inputs
                    X[key] = torch.from_numpy(np.array(val[i])).to(torch.float32) if sampled else torch.from_numpy(np.array(val[i:i+self.input_n_samples[key]])).to(torch.float32)

                    if key in model_inputs:
                        input_dim = self.model_def['Inputs'][key]['dim']
                    elif key in model_states:
                        input_dim = self.model_def['States'][key]['dim']

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
                if connect:
                    if i==0 or i%prediction_samples == 0:
                        self.model.clear_connect_variables()
                result, _ = self.model(X, connect)

                ## Update the recurrent variable
                for close_in, close_out in close_loop.items():
                    if i >= close_loop_windows[close_in]-1:
                        dim = result[close_out].shape[1]  ## take the output time dimension
                        X[close_in] = torch.roll(X[close_in], shifts=-dim, dims=1) ## Roll the time window
                        X[close_in][:, -dim:, :] = result[close_out] ## substitute with the predicted value

                ## Append the prediction of the current sample to the result dictionary
                for key in self.model_def['Outputs'].keys():
                    if result[key].shape[-1] == 1:
                        result[key] = result[key].squeeze(-1)
                        if result[key].shape[-1] == 1:
                            result[key] = result[key].squeeze(-1)
                    result_dict[key].append(result[key].detach().squeeze(dim=0).tolist())

        return result_dict

    def get_samples(self, dataset, index, window=1):
        if self.data_loaded:
            result_dict = {}
            for key in self.model_def['Inputs'].keys():
                result_dict[key] = []
            for idx in range(window):
                for key ,samples in self.data[dataset].items():
                    if key in self.model_def['Inputs'].keys():
                        result_dict[key].append(samples[index+idx])
            return result_dict
        else:
            print('The Dataset must first be loaded using <loadData> function!')
            return {}

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


    def addModel(self, name, stream_list):
        if type(stream_list) is Output:
            stream_list = [stream_list]
        if type(stream_list) is list:
            self.stream_dict[name] = copy.deepcopy(stream_list)
        else:
            raise TypeError(f'json_model is type {type(stream_list)} but must be an Output or list of Output!')
        self.__update_model()

    def removeModel(self, name_list):
        if type(name_list) is str:
            name_list = [name_list]
        if type(name_list) is list:
            for name in name_list:
                check(name in self.stream_dict, IndexError, f"The name {name} is not part of the available models")
                del self.stream_dict[name]
        self.__update_model()

    def addMinimize(self, name, streamA, streamB, loss_function='mse'):
        check(isinstance(streamA, (Output, Stream)), TypeError, 'streamA must be an instance of Output or Stream')
        check(isinstance(streamB, (Output, Stream)), TypeError, 'streamA must be an instance of Output or Stream')
        self.minimize_dict[name]={'A':copy.deepcopy(streamA), 'B': copy.deepcopy(streamB), 'loss':loss_function}
        self.__update_model()
        self.visualizer.showaddMinimize(name)

    def removeMinimize(self, name_list):
        if type(name_list) is str:
            name_list = [name_list]
        if type(name_list) is list:
            for name in name_list:
                check(name in self.minimize_dict, IndexError, f"The name {name} is not part of the available minimuzes")
                del self.minimize_dict[name]
        self.__update_model()
        self.visualizer.showaddMinimize(name)

    def __update_model(self):
        self.model_def = copy.deepcopy(NeuObj().json)
        for key, stream_list in self.stream_dict.items():
            for stream in stream_list:
                self.model_def = merge(self.model_def, stream.json)
        for key, minimize in self.minimize_dict.items():
            self.model_def = merge(self.model_def, minimize['A'].json)
            self.model_def = merge(self.model_def, minimize['B'].json)


    def neuralizeModel(self, sample_time = 1):

        check(sample_time > 0, RuntimeError, 'Sample time must be strictly positive!')
        self.model_def["SampleTime"] = sample_time
        #model_def_final = copy.deepcopy(self.model_def)
        self.visualizer.showModel()

        check(self.model_def['Inputs'] | self.model_def['States'] != {}, RuntimeError, "No model is defined!")
        json_inputs = self.model_def['Inputs'] | self.model_def['States']

        for key,value in self.model_def['States'].items():
            check('closedLoop' in self.model_def['States'][key], RuntimeError, f'Update function is missing for state {key}. Call X.update({key}) on a Stream X.')

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
        self.model = Model(copy.deepcopy(self.model_def), self.minimize_dict, self.input_ns_backward, self.input_n_samples)
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
            #assert set(model_inputs).issubset(source.keys()), f'The dataset is missing some inputs. Inputs needed for the model: {model_inputs}'

            for key in model_inputs:
                if key not in source.keys():
                    continue

                self.data[name][key] = []  ## Initialize the dataset

                back, forw = self.input_ns_backward[key], self.input_ns_forward[key]
                for idx in range(len(source[key]) - self.max_n_samples+1):
                    self.data[name][key].append(source[key][idx + (self.max_samples_backward - back):idx + (self.max_samples_backward + forw)])

            ## Stack the files
            self.num_of_samples[name] = None
            for key in model_inputs:
                if key not in source.keys():
                    continue
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

    def filterData(self, filter_function, dataset_name = None):
        idx_to_remove = []
        if dataset_name is None:
            for name in self.data.keys():
                dataset = self.data[name]
                n_samples = len(dataset[list(dataset.keys())[0]])

                data_for_filter = []
                for i in range(n_samples):
                    new_sample = {key: val[i] for key, val in dataset.items()}
                    data_for_filter.append(new_sample)

                for idx, sample in enumerate(data_for_filter):
                    if not filter_function(sample):
                        idx_to_remove.append(idx)

                for key in self.data[name].keys():
                    self.data[name][key] = np.delete(self.data[name][key], idx_to_remove, axis=0)
                    self.num_of_samples[name] = self.data[name][key].shape[0]
                self.visualizer.showDataset(name=name)

        else:
            dataset = self.data[dataset_name]
            n_samples = len(dataset[list(dataset.keys())[0]])

            data_for_filter = []
            for i in range(n_samples):
                new_sample = {key: val[i] for key, val in dataset.items()}
                data_for_filter.append(new_sample)

            for idx, sample in enumerate(data_for_filter):
                if not filter_function(sample):
                    idx_to_remove.append(idx)

            for key in self.data[dataset_name].keys():
                self.data[dataset_name][key] = np.delete(self.data[dataset_name][key], idx_to_remove, axis=0)
                self.num_of_samples[dataset_name] = self.data[dataset_name][key].shape[0]
            self.visualizer.showDataset(name=dataset_name)

    def __getTrainParams(self, training_params):
        # Set all parameters in training_params
        for key,value in training_params.items():
            try:
                getattr(self, key)
                setattr(self, key, value)
            except:
                raise KeyError(f"The training_params contains a wrong key: {key}.")

            ## Check if the batch_size can be used for the current dataset, otherwise set the batch_size to 1
            if self.train_batch_size > self.n_samples_train:
                self.train_batch_size = 1
            if self.val_batch_size > self.n_samples_val:
                self.val_batch_size = 1
            if self.test_batch_size > self.n_samples_test:
                self.test_batch_size = 1


    def resultAnalysis(self, name_data, XY_data, connect):
        with torch.inference_mode():
            self.performance[name_data] = {}
            self.prediction[name_data] = {}

            self.model.eval()
            A = {}
            B = {}
            aux_losses = {}
            for (name, items) in self.minimize_dict.items():
                window = 'tw' if 'tw' in items['A'].dim else ('sw' if 'sw' in items['A'].dim else None)
                A[name] = torch.zeros([XY_data[list(XY_data.keys())[0]].shape[0],items['A'].dim[window],items['A'].dim['dim']])
                B[name] = torch.zeros([XY_data[list(XY_data.keys())[0]].shape[0],items['B'].dim[window],items['B'].dim['dim']])
                aux_losses[name] = np.zeros([XY_data[list(XY_data.keys())[0]].shape[0],items['A'].dim[window],items['A'].dim['dim']])

            _, minimize_out = self.model(XY_data, connect)
            for ind, (key, value) in enumerate(self.minimize_dict.items()):
                A[key] = minimize_out[value['A'].name]
                B[key] = minimize_out[value['B'].name]
                loss = self.losses[key](minimize_out[value['A'].name], minimize_out[value['B'].name])
                aux_losses[key] = loss.detach().numpy()

            for ind, (key, value) in enumerate(self.minimize_dict.items()):
                A_np = A[key].detach().numpy()
                B_np = B[key].detach().numpy()
                self.performance[name_data][key] = {}
                self.performance[name_data][key][value['loss']] = np.mean(aux_losses[key]).item()
                self.performance[name_data][key]['fvu'] = {}
                # Compute FVU
                residual = A_np - B_np
                error_var = np.var(residual)
                error_mean = np.mean(residual)
                #error_var_manual = np.sum((residual-error_mean) ** 2) / (len(self.prediction['B'][ind]) - 0)
                #print(f"{key} var np:{new_error_var} and var manual:{error_var_manual}")
                self.performance[name_data][key]['fvu']['A'] = (error_var / np.var(A_np)).item()
                self.performance[name_data][key]['fvu']['B'] = (error_var / np.var(B_np)).item()
                self.performance[name_data][key]['fvu']['total'] = np.mean([self.performance[name_data][key]['fvu']['A'],self.performance[name_data][key]['fvu']['B']]).item()
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
                self.performance[name_data][key]['aic'] = {'value':aic,'total_params':total_params,'log_likelihood':log_likelihood}
                # Prediction and target
                self.prediction[name_data][key] = {}
                self.prediction[name_data][key]['A'] = A_np.tolist()
                self.prediction[name_data][key]['B'] = B_np.tolist()

            self.performance[name_data]['total'] = {}
            self.performance[name_data]['total']['mean_error'] = np.mean([value for key,value in aux_losses.items()])
            self.performance[name_data]['total']['fvu'] = np.mean([self.performance[name_data][key]['fvu']['total'] for key in self.minimize_dict.keys()])
            self.performance[name_data]['total']['aic'] = np.mean([self.performance[name_data][key]['aic']['value']for key in self.minimize_dict.keys()])

    def trainModel(self, models=None,
                    train_dataset=None, validation_dataset=None, test_dataset=None, splits=[70,20,10],
                    close_loop=None, step=1, prediction_samples=0,
                    shuffle_data=True, early_stopping=None,
                    lr_gain={}, minimize_gain={}, connect={},
                    training_params = {}):

        if not self.data_loaded:
            print('There is no data loaded! The Training will stop.')
            return
        if not list(self.model.parameters()):
            print('There are no modules with learnable parameters! The Training will stop.')
            return

        check(prediction_samples >= 0, KeyError, 'The sample horizon must be positive!')
        self.close_loop = close_loop
        if self.close_loop:
            for input, output in self.close_loop.items():
                check(input in self.model_def['Inputs'], ValueError, f'the tag {input} is not an input variable.')
                check(output in self.model_def['Outputs'], ValueError, f'the tag {output} is not an output of the network')
            self.visualizer.warning(f'Recurrent train: closing the loop for {prediction_samples} samples')
            recurrent_train = True
        elif self.model_def['States']: ## if we have state variables we have to do the recurrent train
            self.visualizer.warning(f'Recurrent train: Update States variables for {prediction_samples} time steps')
            recurrent_train = True
        elif connect:
            for connect_in, connect_out in connect.items():
                check(connect_in in self.model_def['Inputs'], ValueError, f'the tag {connect_in} is not an input variable.')
                check(connect_out in self.model_def['Outputs'], ValueError, f'the tag {connect_out} is not an output of the network')
            self.visualizer.warning(f'Recurrent train: closing the loop for {prediction_samples} samples')
            recurrent_train = True
        else:
            recurrent_train = False

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

            ## Set name for resultsAnalysis
            train_dataset = "train"
            validation_dataset = "validation"
            test_dataset = "test"
        else: ## Multi-Dataset
            datasets = list(self.data.keys())

            check(train_dataset in datasets, KeyError, f'{train_dataset} Not Loaded!')
            if validation_dataset is not None and validation_dataset not in datasets:
                self.visualizer.warning(f'Validation Dataset [{validation_dataset}] Not Loaded. The training will continue without validation')
            if test_dataset is not None and test_dataset not in datasets:
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
        assert self.n_samples_train > 0, f'There are {self.n_samples_train} samples for training.'
        self.prediction_samples = prediction_samples if (recurrent_train and prediction_samples != 0) else 1

        ## define optimizer
        freezed_model_parameters = set()
        if models:
            if isinstance(models, str):
                models = [models]
            for model_name, model_params in self.stream_dict.items():
                if model_name not in models:
                    freezed_model_parameters = freezed_model_parameters.union(set(model_params[0].json['Parameters'].keys()))
        freezed_model_parameters = freezed_model_parameters - set(lr_gain.keys())
        #print('freezed model parameters: ', freezed_model_parameters)
        learned_model_parameters = set(self.model_def['Parameters'].keys()) - freezed_model_parameters
        #print('learned model parameters: ', learned_model_parameters)
        model_parameters = []
        for param_name, param_value in self.model.all_parameters.items():
            if param_name in lr_gain.keys():  ## if the parameter is specified it has top priority
                model_parameters.append({'params':param_value, 'lr':self.learning_rate*lr_gain[param_name]})
            elif param_name in freezed_model_parameters: ## if the parameter is not in the training model, it's freezed
                model_parameters.append({'params':param_value, 'lr':0.0})
            elif param_name in learned_model_parameters: ## if the parameter is in the training model, it's learned with the default learning rate
                model_parameters.append({'params':param_value, 'lr':self.learning_rate})
        #print('model parameters: ', model_parameters)
        self.optimizer = torch.optim.Adam(model_parameters, weight_decay=self.weight_decay, lr=self.learning_rate)
        #self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate, weight_decay=self.weight_decay)

        ## Define the loss functions
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
            if recurrent_train:
                losses = self.__recurrentTrain(XY_train, self.n_samples_train, self.train_batch_size, minimize_gain, self.prediction_samples, close_loop, step, connect, shuffle=shuffle_data, train=True)
            else:
                losses = self.__Train(XY_train, self.n_samples_train, self.train_batch_size, minimize_gain, shuffle_data, train=True)
            ## save the losses
            for ind, key in enumerate(self.minimize_dict.keys()):
                train_losses[key].append(torch.mean(losses[ind]).tolist())

            if self.n_samples_val > 0: 
                ## VALIDATION
                self.model.eval()
                if recurrent_train:
                    losses = self.__recurrentTrain(XY_val, self.n_samples_val, self.val_batch_size, minimize_gain, self.prediction_samples, close_loop, step, connect, shuffle=False, train=False)
                else:
                    losses = self.__Train(XY_val, self.n_samples_val, self.val_batch_size, minimize_gain, shuffle=False, train=False)
                ## save the losses
                for ind, key in enumerate(self.minimize_dict.keys()):
                    val_losses[key].append(torch.mean(losses[ind]).tolist())

            ## Early-stopping
            if early_stopping:
                if early_stopping(train_losses, val_losses):
                    self.visualizer.warning('Stopping the training..')
                    break

            ## visualize the training...
            self.visualizer.showTraining(epoch, train_losses, val_losses)

        ## save the training time
        end = time.time()
        ## visualize the training time
        self.visualizer.showTrainingTime(end-start)

        ## Test the model
        if self.n_samples_test > 0: 
            ## TEST
            self.model.eval()
            if recurrent_train:
                losses = self.__recurrentTrain(XY_test, self.n_samples_test, self.test_batch_size, minimize_gain, self.prediction_samples, close_loop, step, connect, shuffle=False, train=False)
            else:
                losses = self.__Train(XY_test, self.n_samples_test, self.test_batch_size, minimize_gain, shuffle=False, train=False)
            ## save the losses
            for ind, key in enumerate(self.minimize_dict.keys()):
                test_losses[key] = torch.mean(losses[ind]).tolist()

        '''
        # TODO: adjust the result analysis with states variables
        self.resultAnalysis(train_dataset, XY_train, connect)
        if self.n_samples_val > 0:
            self.resultAnalysis(validation_dataset, XY_val, connect)
        if self.n_samples_test > 0:
            self.resultAnalysis(test_dataset, XY_test, connect)
        '''

        self.visualizer.showResults()
        return train_losses, val_losses, test_losses
        

    def __recurrentTrain(self, data, n_samples, batch_size, loss_gains, prediction_samples, close_loop, step, connect, shuffle=True, train=True):
        ## Sample Shuffle
        initial_value = random.randint(0, step) if shuffle else 0
        ## Initialize the train losses vector
        aux_losses = torch.zeros([len(self.minimize_dict), n_samples//batch_size])
        for idx in range(initial_value, (n_samples - batch_size - prediction_samples + 1), (batch_size + step - 1)):
            ## Build the input tensor
            XY = {key: val[idx:idx+batch_size] for key, val in data.items()}
            ## collect the horizon labels
            XY_horizon = {key: val[idx+1:idx+batch_size+prediction_samples] for key, val in data.items()}
            horizon_losses = {ind: [] for ind in range(len(self.minimize_dict))}
            for horizon_idx in range(prediction_samples):
                ## Model Forward
                if connect and horizon_idx==0:
                    self.model.clear_connect_variables()
                out, minimize_out = self.model(XY, connect)  ## Forward pass
                ## Loss Calculation
                for ind, (key, value) in enumerate(self.minimize_dict.items()):
                    loss = self.losses[key](minimize_out[value['A'].name], minimize_out[value['B'].name])
                    loss = loss * loss_gains[key] if key in loss_gains.keys() else loss  ## Multiply by the gain if necessary
                    horizon_losses[ind].append(loss)

                ## remove the states variables from the data
                if prediction_samples > 1:
                    for state_key in self.model_def['States'].keys():
                        if state_key in XY.keys():
                            del XY[state_key]

                if close_loop or connect:
                    ## Update the input with the recurrent prediction
                    if horizon_idx != prediction_samples-1:
                        for key in XY.keys():
                            if close_loop:
                                if key in close_loop.keys(): ## the variable is recurrent
                                    dim = out[close_loop[key]].shape[1]  ## take the output time dimension
                                    XY[key] = torch.roll(XY[key], shifts=-dim, dims=1) ## Roll the time window
                                    XY[key][:, self.input_ns_backward[key]-dim:self.input_ns_backward[key], :] = out[close_loop[key]] ## substitute with the predicted value
                                    XY[key][:, self.input_ns_backward[key]:, :] = XY_horizon[key][horizon_idx:horizon_idx+batch_size, self.input_ns_backward[key]:, :]  ## fill the remaining values from the dataset
                                else: ## the variable is not recurrent
                                    XY[key] = torch.roll(XY[key], shifts=-1, dims=0)  ## Roll the sample window
                                    XY[key][-1] = XY_horizon[key][batch_size+horizon_idx]  ## take the next sample from the dataset
                            else: ## the variable is not recurrent
                                XY[key] = torch.roll(XY[key], shifts=-1, dims=0)  ## Roll the sample window
                                XY[key][-1] = XY_horizon[key][batch_size+horizon_idx]  ## take the next sample from the dataset
            if train:
                self.optimizer.zero_grad() ## Reset the gradient
            ## Calculate the total loss
            for ind in range(len(self.minimize_dict)):
                total_loss = sum(horizon_losses[ind])
                if train:
                    total_loss.backward(retain_graph=True) ## Backpropagate the error
                aux_losses[ind][idx//batch_size] = total_loss.item()
            ## Gradient Step
            if train:
                self.optimizer.step()
        ## return the losses
        return aux_losses
    
    def __Train(self, data, n_samples, batch_size, loss_gains, shuffle=True, train=True):
        if shuffle:
            randomize = torch.randperm(n_samples)
            data = {key: val[randomize] for key, val in data.items()}
        ## Initialize the train losses vector
        aux_losses = torch.zeros([len(self.minimize_dict),n_samples//batch_size])
        for idx in range(0, (n_samples - batch_size + 1), batch_size):
            ## Build the input tensor
            XY = {key: val[idx:idx+batch_size] for key, val in data.items()}
            ## Reset gradient
            if train:
                self.optimizer.zero_grad()
            ## Model Forward
            _, minimize_out = self.model(XY)  ## Forward pass
            ## Loss Calculation
            for ind, (key, value) in enumerate(self.minimize_dict.items()):
                loss = self.losses[key](minimize_out[value['A'].name], minimize_out[value['B'].name])
                loss = loss * loss_gains[key] if key in loss_gains.keys() else loss  ## Multiply by the gain if necessary
                if train:
                    loss.backward(retain_graph=True)
                aux_losses[ind][idx//batch_size]= loss.item()
            ## Gradient step
            if train:
                self.optimizer.step()
        return aux_losses

    def clear_state(self, state=None):
        check(self.neuralized, ValueError, "The network is not neuralized yet.")
        if self.model_def['States']:
            if state:
                self.model.clear_state(state=state)
            else:
                self.model.clear_state()
        else:
            self.visualizer.warning('The model does not have state variables!')


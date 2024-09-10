import copy

import torch
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm
import random
import os
from pprint import pprint
from pprint import pformat
import re
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
import json
from torch.fx import symbolic_trace

from neu4mes.input import closedloop_name, connect_name
from neu4mes.relation import NeuObj, MAIN_JSON
from neu4mes.visualizer import TextVisualizer, Visualizer
from neu4mes.loss import CustomLoss
from neu4mes.output import Output
from neu4mes.relation import Stream
from neu4mes.model import Model
from neu4mes.utilis import check, argmax_max, argmin_min, merge
from neu4mes.export import plot_fuzzify, generate_training_report, model_to_python, model_to_onnx, model_to_python_onnx
from neu4mes.optimizer import Optimizer, SGD, Adam

from neu4mes import LOG_LEVEL
from neu4mes.logger import logging
log = logging.getLogger(__name__)
log.setLevel(max(logging.ERROR, LOG_LEVEL))

class Neu4mes:
    name = None
    def __init__(self, visualizer = 'Standard', seed=None, workspace=None):

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
        self.model_dict = {}
        self.minimize_dict = {}
        self.update_state_dict = {}

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
        self.standard_train_parameters = {
            'models' : None,
            'train_dataset' : None, 'validation_dataset' : None, 'test_dataset' : None, 'splits' : [70, 20, 10],
            'closed_loop' : {}, 'connect' : {}, 'step' : 1, 'prediction_samples' : 0,
            'shuffle_data' : True, 'early_stopping' : None,
            'minimize_gain' : {},
            'num_of_epochs': 100,
            'train_batch_size' : 128, 'val_batch_size' : 1, 'test_batch_size' : 1,
            'optimizer' : 'Adam',
            'lr' : 0.001, 'lr_param' : {},
            'optimizer_params' : [], 'add_optimizer_params' : [],
            'optimizer_defaults' : {}, 'add_optimizer_defaults' : {}
        }

        # Optimizer
        self.optimizer = None

        # Training Losses
        self.losses = {}

        # Validation Parameters
        self.performance = {}
        self.prediction = {}

        # Export parameters
        if workspace is not None:
            self.workspace = workspace
            os.makedirs(self.workspace, exist_ok=True)
            self.folder = 'neu4mes_'+datetime.now().strftime("%Y_%m_%d_%H_%M")
            self.folder_path = os.path.join(self.workspace, self.folder)
            os.makedirs(self.folder_path, exist_ok=True)


    def __call__(self, inputs={}, sampled=False, closed_loop={}, connect={}, prediction_samples = None):
        inputs = copy.deepcopy(inputs)
        closed_loop = copy.deepcopy(closed_loop)
        connect = copy.deepcopy(connect)

        check(self.neuralized, ValueError, "The network is not neuralized.")

        closed_loop_windows = {}
        for close_in, close_out in closed_loop.items():
            check(close_in in self.model_def['Inputs'], ValueError, f'the tag {close_in} is not an input variable.')
            check(close_out in self.model_def['Outputs'], ValueError, f'the tag {close_out} is not an output of the network')
            if close_in in inputs.keys():
                closed_loop_windows[close_in] = len(inputs[close_in]) if sampled else len(inputs[close_in])-self.input_n_samples[close_in]+1
            else:
                closed_loop_windows[close_in] = 1

        for connect_in, connect_out in connect.items():
            check(connect_in in self.model_def['Inputs'], ValueError, f'the tag {connect_in} is not an input variable.')
            check(connect_out in self.model_def['Outputs'], ValueError, f'the tag {connect_out} is not an output of the network')

        model_inputs = list(self.model_def['Inputs'].keys())
        model_states = list(self.model_def['States'].keys())
        provided_inputs = list(inputs.keys())
        missing_inputs = list(set(model_inputs) - set(provided_inputs) - set(connect.keys()))
        extra_inputs = list(set(provided_inputs) - set(model_inputs) - set(model_states))

        for key in model_states:
            if key in inputs.keys():
                closed_loop_windows[key] = len(inputs[key]) if sampled else len(inputs[key])-self.input_n_samples[key]+1
            else:
                closed_loop_windows[key] = 1

        ## Ignoring extra inputs if not necessary
        if not set(provided_inputs).issubset(set(model_inputs) | set(model_states)):
            self.visualizer.warning(f'The complete model inputs are {model_inputs}, the provided input are {provided_inputs}. Ignoring {extra_inputs}...')
            for key in extra_inputs:
                del inputs[key]
            provided_inputs = list(inputs.keys())
        non_recurrent_inputs = list(set(provided_inputs) - set(closed_loop.keys()) - set(model_states) - set(connect.keys()))

        ## Determine the Maximal number of samples that can be created
        if non_recurrent_inputs:
            if sampled:
                min_dim_ind, min_dim  = argmin_min([len(inputs[key]) for key in non_recurrent_inputs])
                max_dim_ind, max_dim = argmax_max([len(inputs[key]) for key in non_recurrent_inputs])
            else:
                min_dim_ind, min_dim = argmin_min([len(inputs[key])-self.input_n_samples[key]+1 for key in non_recurrent_inputs])
                max_dim_ind, max_dim  = argmax_max([len(inputs[key])-self.input_n_samples[key]+1 for key in non_recurrent_inputs])
        else:
            ps = 0 if prediction_samples is None else prediction_samples
            if provided_inputs:
                min_dim_ind, min_dim  = argmin_min([closed_loop_windows[key]+ps for key in provided_inputs])
                max_dim_ind, max_dim = argmax_max([closed_loop_windows[key]+ps for key in provided_inputs])
            else:
                min_dim = max_dim = ps + 1

        window_dim = min_dim
        if prediction_samples == None:
            prediction_samples = window_dim
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

        ## Initialize the batch_size
        self.model.batch_size = 1
        ## Initialize the connect variables
        self.model.connect = connect
        ## Cycle through all the samples provided
        with torch.inference_mode():
            X = {}
            for i in range(window_dim):
                for key, val in inputs.items():
                    if key in closed_loop.keys() or key in model_states:
                        if i >= closed_loop_windows[key]:
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
                if self.model.connect:
                    if i==0 or i%(prediction_samples+1) == 0:
                        self.model.clear_connect_variables()
                result, _ = self.model(X)

                ## Update the recurrent variable
                for close_in, close_out in closed_loop.items():
                    if i >= closed_loop_windows[close_in]-1:
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

    def addConnect(self, stream_out, state_list_in):
        from neu4mes.input import Connect
        self.__update_state(stream_out, state_list_in, Connect)
        self.__update_model()

    def addClosedLoop(self, stream_out, state_list_in):
        from neu4mes.input import ClosedLoop
        self.__update_state(stream_out, state_list_in, ClosedLoop)
        self.__update_model()

    def __update_state(self, stream_out, state_list_in, UpdateState):
        from neu4mes.input import  State
        if type(state_list_in) is not list:
            state_list_in = [state_list_in]
        for state_in in state_list_in:
            check(isinstance(stream_out, (Output, Stream)), TypeError,
                  f"The {stream_out} must be a Stream or Output and not a {type(stream_out)}.")
            check(type(state_in) is State, TypeError,
                  f"The {state_in} must be a State and not a {type(state_in)}.")
            check(stream_out.dim['dim'] == state_in.dim['dim'], ValueError,
                  f"The dimension of {stream_out.name} is not equal to the dimension of {state_in.name} ({stream_out.dim['dim']}!={state_in.dim['dim']}).")
            if type(stream_out) is Output:
                stream_name = self.model_def['Outputs'][stream_out.name]
                stream_out = Stream(stream_name,stream_out.json,stream_out.dim, 0)
            self.update_state_dict[state_in.name] = UpdateState(stream_out, state_in)

    def addModel(self, name, stream_list):
        if isinstance(stream_list, (Output,Stream)):
            stream_list = [stream_list]
        if type(stream_list) is list:
            self.model_dict[name] = copy.deepcopy(stream_list)
        else:
            raise TypeError(f'stream_list is type {type(stream_list)} but must be an Output or Stream or a list of them')
        self.__update_model()

    def removeModel(self, name_list):
        if type(name_list) is str:
            name_list = [name_list]
        if type(name_list) is list:
            for name in name_list:
                check(name in self.model_dict, IndexError, f"The name {name} is not part of the available models")
                del self.model_dict[name]
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
        self.model_def = copy.deepcopy(MAIN_JSON)
        for key, stream_list in self.model_dict.items():
            for stream in stream_list:
                self.model_def = merge(self.model_def, stream.json)
        for key, minimize in self.minimize_dict.items():
            self.model_def = merge(self.model_def, minimize['A'].json)
            self.model_def = merge(self.model_def, minimize['B'].json)
        for key, update_state in self.update_state_dict.items():
            self.model_def = merge(self.model_def, update_state.json)


    def neuralizeModel(self, sample_time = 1, clear_model = False):
        check(sample_time > 0, RuntimeError, 'Sample time must be strictly positive!')
        self.model_def["SampleTime"] = sample_time

        if self.model is not None and clear_model == False:
            self.model_def_trained = copy.deepcopy(self.model_def)
            for key,param in self.model.all_parameters.items():
                self.model_def_trained['Parameters'][key]['values'] = param.tolist()
                if 'init_fun' in self.model_def_trained['Parameters'][key]:
                    del self.model_def_trained['Parameters'][key]['init_fun']
            model_def = copy.deepcopy(self.model_def_trained)
        else:
            model_def = copy.deepcopy(self.model_def)
        self.visualizer.showModel(model_def)

        check(model_def['Inputs'] | model_def['States'] != {}, RuntimeError, "No model is defined!")
        json_inputs = model_def['Inputs'] | self.model_def['States']

        for key,value in model_def['States'].items():
            check(closedloop_name in self.model_def['States'][key] or connect_name in self.model_def['States'][key],
                  KeyError, f'Update function is missing for state {key}. Use Connect or ClosedLoop to update the state.')

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
        self.model = Model(model_def, self.minimize_dict, self.input_ns_backward, self.input_n_samples)
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

        num_of_samples = []
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
            for key in format_idx.keys():
                self.data[name][key] = np.stack(self.data[name][key])
                num_of_samples.append(self.data[name][key].shape[0])

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
            for key in model_inputs:
                if key not in source.keys():
                    continue
                self.data[name][key] = np.stack(self.data[name][key])
                if self.data[name][key].ndim == 2: ## Add the sample dimension
                    self.data[name][key] = np.expand_dims(self.data[name][key], axis=-1)
                if self.data[name][key].ndim > 3:
                    self.data[name][key] = np.squeeze(self.data[name][key], axis=1)
                num_of_samples.append(self.data[name][key].shape[0])

        # Check dim of the samples
        check(len(set(num_of_samples)) == 1, ValueError,
              f"The number of the sample of the dataset {name} are not the same for all input in the dataset")
        self.num_of_samples[name] = num_of_samples[0]

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


    def resultAnalysis(self, name_data, XY_data):
        import warnings
        self.model.batch_size = XY_data[list(XY_data.keys())[0]].shape[0]
        self.clear_state()
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

            _, minimize_out = self.model(XY_data)
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
                with warnings.catch_warnings(record=True) as w:
                    self.performance[name_data][key]['fvu']['A'] = (error_var / np.var(A_np)).item()
                    self.performance[name_data][key]['fvu']['B'] = (error_var / np.var(B_np)).item()
                    if w and np.var(A_np) == 0.0 and  np.var(B_np) == 0.0:
                        self.performance[name_data][key]['fvu']['A'] = np.nan
                        self.performance[name_data][key]['fvu']['B'] = np.nan
                self.performance[name_data][key]['fvu']['total'] = np.mean([self.performance[name_data][key]['fvu']['A'],self.performance[name_data][key]['fvu']['B']]).item()
                # Compute AIC
                #normal_dist = norm(0, error_var ** 0.5)
                #probability_of_residual = normal_dist.pdf(residual)
                #log_likelihood_first = sum(np.log(probability_of_residual))
                p1 = -len(residual)/2.0*np.log(2*np.pi)
                with warnings.catch_warnings(record=True) as w:
                    p2 = -len(residual)/2.0*np.log(error_var)
                    p3 = -1 / (2.0 * error_var) * np.sum(residual ** 2)
                    if w and p2 == np.float32(np.inf) and p3 == np.float32(-np.inf):
                        p2 = p3 = 0.0
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


    def __get_train_parameters(self, training_params):
        run_train_parameters = copy.deepcopy(self.standard_train_parameters)
        if training_params is None:
            return run_train_parameters
        for key, value in training_params.items():
            check(key in run_train_parameters, KeyError, f"The param {key} is not exist as standard parameters")
            run_train_parameters[key] = value
        return run_train_parameters

    def __get_parameter(self, **parameter):
        assert len(parameter) == 1
        name = list(parameter.keys())[0]
        self.run_training_params[name] =  parameter[name] if parameter[name] is not None else self.run_training_params[name]
        return self.run_training_params[name]

    def __get_batch_sizes(self, train_batch_size, val_batch_size, test_batch_size):
        ## Check if the batch_size can be used for the current dataset, otherwise set the batch_size to the maximum value
        self.__get_parameter(train_batch_size = train_batch_size)
        self.__get_parameter(val_batch_size = val_batch_size)
        self.__get_parameter(test_batch_size = test_batch_size)
        if self.run_training_params['train_batch_size'] > self.run_training_params['n_samples_train']:
            self.run_training_params['train_batch_size'] = self.run_training_params['n_samples_train']
        if  self.run_training_params['val_batch_size'] > self.run_training_params['n_samples_val']:
            self.run_training_params['val_batch_size'] = self.run_training_params['n_samples_val']
        if self.run_training_params['test_batch_size'] > self.run_training_params['n_samples_test']:
            self.run_training_params['test_batch_size'] = self.run_training_params['n_samples_test']
        return self.run_training_params['train_batch_size'], self.run_training_params['val_batch_size'], self.run_training_params['test_batch_size']

    def __inizilize_optimizer(self, optimizer, optimizer_params, optimizer_defaults, add_optimizer_params, add_optimizer_defaults, models, lr, lr_param):
        # Get optimizer and initialization parameters
        optimizer = copy.deepcopy(self.__get_parameter(optimizer=optimizer))
        optimizer_params = copy.deepcopy(self.__get_parameter(optimizer_params=optimizer_params))
        optimizer_defaults = copy.deepcopy(self.__get_parameter(optimizer_defaults=optimizer_defaults))
        add_optimizer_params = copy.deepcopy(self.__get_parameter(add_optimizer_params=add_optimizer_params))
        add_optimizer_defaults = copy.deepcopy(self.__get_parameter(add_optimizer_defaults=add_optimizer_defaults))

        ## Get params to train
        models = self.__get_parameter(models=models)
        all_parameters = self.model.all_parameters
        params_to_train = set()
        if models:
            if isinstance(models, str):
                models = [models]
            for model_name, model_params in self.model_dict.items():
                if model_name in models:
                    params_to_train = params_to_train.union(set(model_params[0].json['Parameters'].keys()))
        else:
            self.__get_parameter(models=list(self.model_dict.keys()))
            params_to_train = all_parameters.keys()

        # Get the optimizer
        if type(optimizer) is str:
            if optimizer == 'SGD':
                optimizer = SGD({},[])
            elif optimizer == 'Adam':
                optimizer = Adam({},[])
        else:
            check(issubclass(type(optimizer), Optimizer), TypeError,
                  "The optimizer must be an Optimizer or str")

        optimizer.set_params_to_train(all_parameters, params_to_train)

        optimizer.add_defaults('lr', self.run_training_params['lr'])
        optimizer.add_option_to_params('lr', self.run_training_params['lr_param'])

        if optimizer_defaults != {}:
            optimizer.set_defaults(optimizer_defaults)
        if optimizer_params != []:
            optimizer.set_params(optimizer_params)

        for key, value in add_optimizer_defaults.items():
            optimizer.add_defaults(key, value)

        add_optimizer_params = optimizer.unfold(add_optimizer_params)
        for param in add_optimizer_params:
            par = param['params']
            del param['params']
            for key, value in param.items():
                optimizer.add_option_to_params(key, {par:value})

        # Modify the parameter
        optimizer.add_defaults('lr', lr)
        optimizer.add_option_to_params('lr', lr_param)

        return optimizer


    def trainModel(self,
                    models=None,
                    train_dataset = None, validation_dataset = None, test_dataset = None, splits = None,
                    closed_loop = None, connect = None, step = None, prediction_samples = None,
                    shuffle_data = None, early_stopping=  None,
                    minimize_gain = None,
                    num_of_epochs = None,
                    train_batch_size = None, val_batch_size = None, test_batch_size = None,
                    optimizer = None,
                    lr = None, lr_param = None, #weight_decay = None, weight_decay_param = None,
                    optimizer_params = None, optimizer_defaults = None,
                    training_params = None,
                    add_optimizer_params = None, add_optimizer_defaults = None
                   ):
        # def trainModel(self, train_parameters = None, optimizer_parameters = None, **kwargs):
        check(self.data_loaded, RuntimeError, 'There is no data loaded! The Training will stop.')
        check(list(self.model.parameters()), RuntimeError, 'There are no modules with learnable parameters! The Training will stop.')

        # Get running parameter from dict
        self.run_training_params = copy.deepcopy(self.__get_train_parameters(training_params))

        # Get connect and closed_loop
        prediction_samples = self.__get_parameter(prediction_samples = prediction_samples)
        check(prediction_samples >= 0, KeyError, 'The sample horizon must be positive!')

        step = self.__get_parameter(step = step)
        closed_loop = self.__get_parameter(closed_loop = closed_loop)
        connect = self.__get_parameter(connect = connect)
        recurrent_train = True
        if closed_loop:
            for input, output in closed_loop.items():
                check(input in self.model_def['Inputs'], ValueError, f'the tag {input} is not an input variable.')
                check(output in self.model_def['Outputs'], ValueError, f'the tag {output} is not an output of the network')
                self.visualizer.warning(f'Recurrent train: closing the loop between the the input ports {input} and the output ports {output} for {prediction_samples} samples')
        elif connect:
            for connect_in, connect_out in connect.items():
                check(connect_in in self.model_def['Inputs'], ValueError, f'the tag {connect_in} is not an input variable.')
                check(connect_out in self.model_def['Outputs'], ValueError, f'the tag {connect_out} is not an output of the network')
                self.visualizer.warning(f'Recurrent train: connecting the input ports {connect_in} with output ports {connect_out} for {prediction_samples} samples')
        elif self.model_def['States']: ## if we have state variables we have to do the recurrent train
            self.visualizer.warning(f"Recurrent train: update States variables {list(self.model_def['States'].keys())} for {prediction_samples} samples")
        else:
            if prediction_samples != 0:
                self.visualizer.warning(
                    f"The value of the prediction_samples={prediction_samples} is not used in not recursive network.")
            recurrent_train = False
        self.run_training_params['recurrent_train'] = recurrent_train

        ## Get early stopping
        early_stopping = self.__get_parameter(early_stopping = early_stopping)

        # Get dataset for training
        shuffle_data = self.__get_parameter(shuffle_data = shuffle_data)

        ## Get the dataset name
        train_dataset = self.__get_parameter(train_dataset = train_dataset)
        #TODO manage multiple datasets
        if train_dataset is None: ## If we use all datasets with the splits
            splits = self.__get_parameter(splits = splits)
            check(len(splits)==3, ValueError, '3 elements must be inserted for the dataset split in training, validation and test')
            check(sum(splits)==100, ValueError, 'Training, Validation and Test splits must sum up to 100.')
            check(splits[0] > 0, ValueError, 'The training split cannot be zero.')

            ## Get the dataset name
            dataset = list(self.data.keys())[0] ## take the dataset name

            # Collect the split sizes
            train_size = splits[0] / 100.0
            val_size = splits[1] / 100.0
            test_size = 1 - (train_size + val_size)
            num_of_samples = self.num_of_samples[dataset]
            n_samples_train = round(num_of_samples*train_size)
            n_samples_val = round(num_of_samples*val_size)
            n_samples_test = round(num_of_samples*test_size)

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
            train_dataset = self.__get_parameter(train_dataset = f"train_{dataset}_{train_size:0.2f}")
            validation_dataset = self.__get_parameter(validation_dataset =f"validation_{dataset}_{val_size:0.2f}")
            test_dataset = self.__get_parameter(test_dataset = f"test_{dataset}_{test_size:0.2f}")
        else: ## Multi-Dataset
            ## Get the names of the datasets
            datasets = list(self.data.keys())
            validation_dataset = self.__get_parameter(validation_dataset=validation_dataset)
            test_dataset = self.__get_parameter(test_dataset=test_dataset)

            ## Collect the number of samples for each dataset
            n_samples_train, n_samples_val, n_samples_test = 0, 0, 0

            check(train_dataset in datasets, KeyError, f'{train_dataset} Not Loaded!')
            if validation_dataset is not None and validation_dataset not in datasets:
                self.visualizer.warning(f'Validation Dataset [{validation_dataset}] Not Loaded. The training will continue without validation')
            if test_dataset is not None and test_dataset not in datasets:
                self.visualizer.warning(f'Test Dataset [{test_dataset}] Not Loaded. The training will continue without test')

            ## Split into train, validation and test
            n_samples_train = self.num_of_samples[train_dataset]
            XY_train = {key: torch.from_numpy(val).to(torch.float32) for key, val in self.data[train_dataset].items()}
            if validation_dataset in datasets:
                n_samples_val = self.num_of_samples[validation_dataset]
                XY_val = {key: torch.from_numpy(val).to(torch.float32) for key, val in self.data[validation_dataset].items()}
            if test_dataset in datasets:
                n_samples_test = self.num_of_samples[test_dataset]
                XY_test = {key: torch.from_numpy(val).to(torch.float32) for key, val in self.data[test_dataset].items()}

        assert n_samples_train > 0, f'There are {n_samples_train} samples for training.'
        self.run_training_params['n_samples_train'] = n_samples_train
        self.run_training_params['n_samples_val'] = n_samples_val
        self.run_training_params['n_samples_test'] = n_samples_test
        train_batch_size, val_batch_size, test_batch_size = self.__get_batch_sizes(train_batch_size, val_batch_size, test_batch_size)

        ## Define the optimizer
        optimizer = self.__inizilize_optimizer(optimizer, optimizer_params, optimizer_defaults, add_optimizer_params, add_optimizer_defaults, models, lr, lr_param)
        self.run_training_params['optimizer'] = optimizer.name
        self.run_training_params['optimizer_params'] = optimizer.optimizer_params
        self.run_training_params['optimizer_defaults'] = optimizer.optimizer_defaults
        self.optimizer = optimizer.get_torch_optimizer()

        ## Get num_of_epochs
        num_of_epochs = self.__get_parameter(num_of_epochs = num_of_epochs)

        ## Define the loss functions
        minimize_gain = self.__get_parameter(minimize_gain = minimize_gain)
        self.run_training_params['minimize'] = {}
        for name, values in self.minimize_dict.items():
            self.losses[name] = CustomLoss(values['loss'])
            self.run_training_params['minimize'][name] = {}
            self.run_training_params['minimize'][name]['A'] = values['A'].name
            self.run_training_params['minimize'][name]['B'] = values['B'].name
            self.run_training_params['minimize'][name]['loss'] = values['loss']
            if name in minimize_gain:
                self.run_training_params['minimize'][name]['gain'] = minimize_gain[name]

        # Clean the dict of the training parameter
        del self.run_training_params['minimize_gain']
        del self.run_training_params['lr']
        #del self.run_training_params['weight_decay']
        del self.run_training_params['lr_param']
        #del self.run_training_params['weight_decay_param']
        if not recurrent_train:
            del self.run_training_params['connect']
            del self.run_training_params['closed_loop']
            del self.run_training_params['step']
            del self.run_training_params['prediction_samples']

        ## Create the train, validation and test loss dictionaries
        train_losses, val_losses, test_losses = {}, {}, {}
        for key in self.minimize_dict.keys():
            train_losses[key] = []
            if n_samples_val > 0:
                val_losses[key] = []

        # Check the needed keys are in the datasets
        keys = set(self.model_def['Inputs'].keys())
        keys |= {value['A'].name for value in self.minimize_dict.values()}|{value['B'].name for value in self.minimize_dict.values()}
        keys -= set(self.model_def['Relations'].keys())
        keys -= set(self.model_def['States'].keys())
        keys -= set(self.model_def['Outputs'].keys())
        if 'connect' in self.run_training_params:
            keys -= set(self.run_training_params['connect'].keys())
        if 'closed_loop' in self.run_training_params:
            keys -= set(self.run_training_params['closed_loop'].keys())
        check(set(keys).issubset(set(XY_train.keys())), KeyError, f"Not all the mandatory keys {keys} are present in the training dataset {set(XY_train.keys())}.")

        # Show the training params
        self.visualizer.showTrainParams()
        check((n_samples_train - train_batch_size - prediction_samples + 1) > 0, ValueError, f"The number of available sample are (n_samples_train - train_batch_size - prediction_samples + 1) = {(n_samples_train - train_batch_size - prediction_samples + 1)}.")


        import time
        ## start the train timer
        start = time.time()
        for epoch in range(num_of_epochs):
            ## TRAIN
            self.model.train()
            if recurrent_train:
                losses = self.__recurrentTrain(XY_train, n_samples_train, train_batch_size, minimize_gain, prediction_samples, closed_loop, step, connect, shuffle=shuffle_data, train=True)
            else:
                losses = self.__Train(XY_train,n_samples_train, train_batch_size, minimize_gain, shuffle=shuffle_data, train=True)
            ## save the losses
            for ind, key in enumerate(self.minimize_dict.keys()):
                train_losses[key].append(torch.mean(losses[ind]).tolist())

            if n_samples_val > 0:
                ## VALIDATION
                self.model.eval()
                if recurrent_train:
                    losses = self.__recurrentTrain(XY_val, n_samples_val, val_batch_size, minimize_gain, prediction_samples, closed_loop, step, connect, shuffle=False, train=False)
                else:
                    losses = self.__Train(XY_val, n_samples_val, val_batch_size, minimize_gain, shuffle=False, train=False)
                ## save the losses
                for ind, key in enumerate(self.minimize_dict.keys()):
                    val_losses[key].append(torch.mean(losses[ind]).tolist())

            ## Early-stopping
            if early_stopping:
                if early_stopping(train_losses, val_losses, training_params):
                    self.visualizer.warning('Stopping the training..')
                    break

            ## visualize the training...
            self.visualizer.showTraining(epoch, train_losses, val_losses)

        ## save the training time
        end = time.time()
        ## visualize the training time
        self.visualizer.showTrainingTime(end-start)

        ## Test the model
        if n_samples_test > 0:
            ## TEST
            self.model.eval()
            if recurrent_train:
                losses = self.__recurrentTrain(XY_test, n_samples_test, test_batch_size, minimize_gain, prediction_samples, closed_loop, step, connect, shuffle=False, train=False)
            else:
                losses = self.__Train(XY_test, n_samples_test, test_batch_size, minimize_gain, shuffle=False, train=False)
            ## save the losses
            for ind, key in enumerate(self.minimize_dict.keys()):
                test_losses[key] = torch.mean(losses[ind]).tolist()

        self.resultAnalysis(train_dataset, XY_train)
        if self.run_training_params['n_samples_val'] > 0:
            self.resultAnalysis(validation_dataset, XY_val)
        if self.run_training_params['n_samples_test'] > 0:
            self.resultAnalysis(test_dataset, XY_test)

        self.visualizer.showResults()
        return train_losses, val_losses, test_losses


    def __recurrentTrain(self, data, n_samples, batch_size, loss_gains, prediction_samples, closed_loop, step, connect, shuffle=True, train=True):
        ## Sample Shuffle
        initial_value = random.randint(0, step - 1) if shuffle else 0
        ## Initialize the batch_size
        self.model.batch_size = batch_size
        ## Initialize the train losses vector
        aux_losses = torch.zeros([len(self.minimize_dict), n_samples//batch_size])
        ## Initialize connect inputs
        if connect:
            self.model.connect = connect
        ## +2 means that n_samples = 1 - batch_size = 1 - prediction_samples = 1 + 2 = 1 # one epochs
        for idx in range(initial_value, (n_samples - batch_size - prediction_samples + 1), (batch_size + step - 1)):
            if train:
                self.optimizer.zero_grad() ## Reset the gradient

            ## Build the input tensor
            XY = {key: val[idx:idx+batch_size] for key, val in data.items()}
            ## collect the horizon labels
            XY_horizon = {key: val[idx:idx+batch_size+prediction_samples] for key, val in data.items()}
            horizon_losses = {ind: [] for ind in range(len(self.minimize_dict))}
            for horizon_idx in range(prediction_samples + 1):
                ## Model Forward
                if self.model.connect and horizon_idx==0:
                    self.model.clear_connect_variables()
                out, minimize_out = self.model(XY)  ## Forward pass
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

                ## Update the input with the recurrent prediction
                if horizon_idx < prediction_samples:
                    for key in XY.keys():
                        if key in closed_loop.keys(): ## the input is recurrent
                            dim = out[closed_loop[key]].shape[1]  ## take the output time dimension
                            XY[key] = torch.roll(XY[key], shifts=-dim, dims=1) ## Roll the time window
                            XY[key][:, self.input_ns_backward[key]-dim:self.input_ns_backward[key], :] = out[closed_loop[key]] ## substitute with the predicted value
                            XY[key][:, self.input_ns_backward[key]:, :] = XY_horizon[key][horizon_idx:horizon_idx+batch_size, self.input_ns_backward[key]:, :]  ## fill the remaining values from the dataset
                        else: ## the input is not recurrent
                            XY[key] = torch.roll(XY[key], shifts=-1, dims=0)  ## Roll the sample window
                            XY[key][-1] = XY_horizon[key][batch_size+horizon_idx]  ## take the next sample from the dataset

            ## Calculate the total loss
            for ind in range(len(self.minimize_dict)):
                total_loss = sum(horizon_losses[ind])/(prediction_samples+1)
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
        ## Initialize the batch_size
        self.model.batch_size = batch_size
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
            self.model.clear_state(state=state)
        else:
            self.visualizer.warning('The model does not have state variables!')


    '''
    def exportONNX(self, tracer_path):
        # Step 1: Define the mapping dictionary
        trace_mapping = {}
        forward = 'def forward(self,'
        dummy_inputs = []
        input_names = []
        for key, item in self.model_def['Inputs'].items():
            value = f'kwargs[\'{key}\']'
            trace_mapping[value] = key
            forward = forward + f' {key},'
            input_names.append(key)
            window_size = self.input_n_samples[key]
            dummy_inputs.append(torch.randn(size=(1, window_size, item['dim'])))
        forward = forward + '):'
        output_names = [name for name in self.model_def['Outputs'].keys()]
        dummy_inputs = tuple(dummy_inputs)

        # Step 2: Open and read the file
        with open(tracer_path, 'r') as file:
            file_content = file.read()

        file_content = file_content.replace('def forward(self, kwargs):', forward)

        # Step 3: Perform the substitution
        for key, value in trace_mapping.items():
            file_content = file_content.replace(key, value)

        # Step 4: Write the modified content back to a new file
        onnx_path = tracer_path.replace('.py','_onnx.py')
        with open(onnx_path, 'w') as file:
            file.write(file_content)

        # Step 5: Import the compatible tracer
        self.importTracer(onnx_path)

        self.model.eval()

        onnx_path = tracer_path.replace('.py','.onnx')
        torch.onnx.export(
                    self.model,                            # The model to be exported
                    dummy_inputs,                          # Tuple of inputs to match the forward signature
                    onnx_path,                             # File path to save the ONNX model
                    export_params=True,                    # Store the trained parameters in the model file
                    opset_version=12,                      # ONNX version to export to (you can use 11 or higher)
                    do_constant_folding=True,              # Optimize constant folding for inference
                    input_names=input_names,               # Name each input as they will appear in ONNX
                    output_names=output_names,             # Name the output
                    )
    
    
    def exportModel(self):
        import io
        import onnx
        from onnx import reference as onnxreference

        features = {}
        for name, value in self.model_def['Inputs'].items():
            window_size = self.input_n_samples[name]
            features[name] = torch.randn(size=(1, window_size, value['dim']))
        #torch_out, torch_min = self.model(features)

        f = io.BytesIO()
        #torch.onnx.export(self.model, {"x": features}, f)
        torch.onnx.export(self.model, features, f)
        onnx_model = onnx.load_model_from_string(f.getvalue())

        sess = onnxreference.ReferenceEvaluator(onnx_model)
        model_input_names = [i.name for i in onnx_model.graph.input]
        input_dict = dict(zip(model_input_names, features.values()))
        onnx_out = sess.run(None, input_dict)
        print("onnx_out:", onnx_out)
    '''

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        if not self.neuralized:
            print('The model is not neuralized yet!')
            return
        self.model.load_state_dict(torch.load(path))


    def exportJSON(self,):
        # Specify the JSON file name
        file_name = "model.json"
        # Combine the folder path and file name to form the complete file path
        file_path = os.path.join(self.folder_path, file_name)
        # Export the dictionary as a JSON file
        with open(file_path, 'w') as json_file:
            pformat(self.model_def, width=80).strip().splitlines()
            json_file.write(pformat(self.model_def, width=80).strip().replace('\'', '\"'))
        self.visualizer.warning(f"The model definition has been exported to {file_name} as a JSON file.")
        return file_path


    def exportTracer(self,):
        if not self.neuralized:
            self.visualizer.warning('Export Error: the model is not neuralized yet.')
            return

        ## Export to python file
        python_path = model_to_python(self.model_def, self.model, folder_path=self.folder_path)
        ## Export to python file (onnx compatible)
        python_onnx_path = model_to_python_onnx(self.model_def, tracer_path=python_path)
        ## Export to onnx file
        self.importTracer(python_onnx_path)
        self.model.eval()
        onnx_path = model_to_onnx(self.model, self.model_def, self.input_n_samples, python_path)

        self.visualizer.warning(f"The pytorch model has been exported to {self.folder}.")
        return python_path, python_onnx_path, onnx_path


    def importTracer(self, file_path):
        import sys
        import os
        # Add the directory containing your file to sys.path
        directory = os.path.dirname(file_path)
        sys.path.insert(0, directory)
        # Import the module by filename (without .py)
        module_name = os.path.basename(file_path)[:-3]
        module = __import__(module_name)

        self.model = module.TracerModel()

    def ExportReport(self, data, train_loss, val_loss):
        file_name = "report.pdf"
        # Combine the folder path and file name to form the complete file path
        file_path = os.path.join(self.folder_path, file_name)

        # Create PDF
        c = canvas.Canvas(file_path, pagesize=letter)
        width, height = letter

        with torch.inference_mode():
            out, minimize_out = self.model(data)

        for key, value in self.minimize_dict.items():
            # Create loss plot
            plt.figure(figsize=(10, 5))
            plt.plot(train_loss[key], label='train loss')
            if val_loss:
                plt.plot(val_loss[key], label='validation loss')
            plt.title(f'{key} Error Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            loss_plot_buffer = io.BytesIO()
            plt.savefig(loss_plot_buffer, format='png')
            loss_plot_buffer.seek(0)
            plt.close()

            # Add loss plot
            c.drawString(50, height - 20, f'{key} Report')
            c.drawImage(ImageReader(loss_plot_buffer), 70, height - 270, width=500, height=250)

            # Convert tensors to numpy arrays
            name_a, name_b = value['A'].name, value['B'].name
            if isinstance(minimize_out[value['A'].name], torch.Tensor):
                y_pred = minimize_out[value['A'].name].squeeze().squeeze().detach().cpu().numpy()
            if isinstance(minimize_out[value['B'].name], torch.Tensor):
                y_true = minimize_out[value['B'].name].squeeze().squeeze().detach().cpu().numpy()
            # Create the scatter plot
            plt.figure(figsize=(10, 6))
            plt.scatter(y_true, y_pred, alpha=0.5)
            # Plot the perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            # Customize the plot
            plt.title(f"Predicted({name_a}) vs Real Values({name_b})")
            # Add a text box with correlation coefficient
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes,
                     verticalalignment='top')
            pred_real_plot_buffer = io.BytesIO()
            plt.savefig(pred_real_plot_buffer, format='png')
            pred_real_plot_buffer.seek(0)
            plt.close()

            # Add predicted vs real values plot
            c.drawImage(ImageReader(pred_real_plot_buffer), 70, height - 520, width=500, height=250)

            # Create the scatter plot
            plt.figure(figsize=(10, 6))
            plt.plot(y_pred, label=name_a)
            plt.plot(y_true, label=name_b)
            # Customize the plot
            plt.title(f"{key}: Predicted({name_a}) vs Real Values({name_b})")
            plt.xlabel("Samples")
            plt.ylabel("Values")
            plt.legend()
            plot_buffer = io.BytesIO()
            plt.savefig(plot_buffer, format='png')
            plot_buffer.seek(0)
            plt.close()

            # Add predicted vs real values plot
            c.drawImage(ImageReader(plot_buffer), 70, height - 770, width=500, height=250)
            c.showPage()

        for name, params in self.model_def['Functions'].items():
            if 'Fuzzify' in name:
                fig = plot_fuzzify(params=params)
                fuzzy_buffer = io.BytesIO()
                fig.savefig(fuzzy_buffer, format='png')
                fuzzy_buffer.seek(0)

                c.drawString(100, height - 50, f"fuzzy function : {name}")
                c.drawImage(ImageReader(fuzzy_buffer), 50, height - 350, width=500, height=250)

                c.showPage()

        c.save()
        self.visualizer.warning(f"Training report saved as {file_name}")


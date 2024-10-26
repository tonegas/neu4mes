# Extern packages
import copy, os, json, random, torch
import numpy as np
import pandas as pd

# Neu4mes packages
from neu4mes.input import closedloop_name, connect_name
from neu4mes.relation import MAIN_JSON
from neu4mes.visualizer import TextVisualizer, Visualizer, MPLVisualizer, MPLNotebookVisualizer
from neu4mes.loss import CustomLoss
from neu4mes.output import Output
from neu4mes.relation import Stream
from neu4mes.model import Model
from neu4mes.utils import check, argmax_max, argmin_min, merge, tensor_to_list
from neu4mes.optimizer import Optimizer, SGD, Adam
from neu4mes.exporter import Exporter, StandardExporter
from neu4mes.modeldef import ModelDef

from neu4mes import LOG_LEVEL
from neu4mes.logger import logging
log = logging.getLogger(__name__)
log.setLevel(max(logging.DEBUG, LOG_LEVEL))

class Neu4mes:
    def __init__(self,
                 visualizer:str|Visualizer|None = 'Standard',
                 exporter:str|Exporter|None = 'Standard',
                 seed:int|None = None,
                 workspace:str|None = None,
                 log_internal:bool = False,
                 save_history:bool = False):

        # Visualizer
        if visualizer == 'Standard':
            self.visualizer = TextVisualizer(1)
        elif visualizer != None:
            self.visualizer = visualizer
        else:
            self.visualizer = Visualizer()
        self.visualizer.set_n4m(self)

        # Exporter
        if exporter == 'Standard':
            self.exporter = StandardExporter(workspace, save_history)
        elif exporter != None:
            self.exporter = exporter
        else:
            self.exporter = Exporter()
        self.exporter.set_n4m(self)

        ## Set the random seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed=seed) ## set the pytorch seed
            random.seed(seed) ## set the random module seed
            np.random.seed(seed) ## set the numpy seed

        # Save internal
        self.log_internal = log_internal
        if self.log_internal == True:
            self.internals = {}

        # Inizialize the model definition
        #self.model_dict = {}
        #self.minimize_dict = {}
        #self.update_state_dict = {}

        # Models definition
        self.model_def = ModelDef()
        #self.model_def_loaded = None
        #self.model_def_values = None

        # Network Parametrs
        # To be removed
        self.input_n_samples = {}
        self.max_n_samples = 0
        # To be removed
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
            'shuffle_data' : True,
            'early_stopping' : None, 'early_stopping_params' : {},
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

    def __call__(self, inputs = {}, sampled = False, closed_loop = {}, connect = {}, prediction_samples = 'auto', num_of_samples = 'auto'):#, align_input = False):
        ## Copy dict for avoid python bug
        inputs = copy.deepcopy(inputs)
        closed_loop = copy.deepcopy(closed_loop)
        connect = copy.deepcopy(connect)

        ## Check neuralize
        check(self.neuralized, ValueError, "The network is not neuralized.")

        ## Bild the list of inputs
        model_inputs = list(self.model_def['Inputs'].keys())
        model_states = list(self.model_def['States'].keys())
        provided_inputs = list(inputs.keys())
        missing_inputs = list(set(model_inputs) - set(provided_inputs)) #- set(connect.keys()))
        extra_inputs = list(set(provided_inputs) - set(model_inputs) - set(model_states))
        if not set(provided_inputs).issubset(set(model_inputs) | set(model_states)):
            ## Ignoring extra inputs
            self.visualizer.warning(f'The complete model inputs are {model_inputs}, the provided input are {provided_inputs}. Ignoring {extra_inputs}...')
            for key in extra_inputs:
                del inputs[key]
            provided_inputs = list(inputs.keys())
        non_recurrent_inputs = list(set(provided_inputs) - set(closed_loop.keys()) - set(connect.keys()) - set(model_states))
        recurrent_inputs = set(closed_loop.keys())|set(connect.keys())|set(model_states)

        ## Define input windows and check closed loop and connect
        input_windows = {}
        for in_var, out_var in (closed_loop.items() | connect.items()):
            check(in_var in self.model_def['Inputs'], ValueError, f'the tag {in_var} is not an input variable.')
            check(out_var in self.model_def['Outputs'], ValueError, f'the tag {out_var} is not an output of the network')
            if in_var in inputs.keys():
                input_windows[in_var] = len(inputs[in_var]) if sampled else len(inputs[in_var]) - self.input_n_samples[in_var] + 1
            else:
                input_windows[in_var] = 1
        for key in model_states:
            if key in inputs.keys():
                input_windows[key] = len(inputs[key]) if sampled else len(inputs[key]) - self.input_n_samples[key] + 1
            else:
                input_windows[key] = 1

        ## Determine the Maximal number of samples that can be created
        if non_recurrent_inputs:
            if sampled:
                min_dim_ind, min_dim  = argmin_min([len(inputs[key]) for key in non_recurrent_inputs])
                max_dim_ind, max_dim = argmax_max([len(inputs[key]) for key in non_recurrent_inputs])
            else:
                min_dim_ind, min_dim = argmin_min([len(inputs[key])-self.input_n_samples[key]+1 for key in non_recurrent_inputs])
                max_dim_ind, max_dim  = argmax_max([len(inputs[key])-self.input_n_samples[key]+1 for key in non_recurrent_inputs])
            min_din_key = non_recurrent_inputs[min_dim_ind]
            max_din_key = non_recurrent_inputs[max_dim_ind]
        else:
            if recurrent_inputs:
                #ps = 0 if prediction_samples=='auto' or prediction_samples is None else prediction_samples
                if provided_inputs:
                    min_dim_ind, min_dim = argmin_min([input_windows[key]  for key in provided_inputs])
                    max_dim_ind, max_dim = argmax_max([input_windows[key]  for key in provided_inputs])
                    min_din_key = provided_inputs[min_dim_ind]
                    max_din_key = provided_inputs[max_dim_ind]
                else:
                    min_dim = max_dim =  1
            else:
                min_dim = max_dim = 0

        ## Define the number of samples
        if num_of_samples != 'auto':
            window_dim = min_dim = max_dim = num_of_samples
        else:
            # Use the minimum number of input samples if the net is not autonoma otherwise the minimum number of state samples
            window_dim = min_dim
        check(window_dim > 0, StopIteration, f'Missing at least {abs(min_dim)+1} samples in the input window')

        ## Autofill the missing inputs
        if missing_inputs:
            self.visualizer.warning(f'Inputs not provided: {missing_inputs}. Autofilling with zeros..')
            for key in missing_inputs:
                inputs[key] = np.zeros(
                    shape=(self.input_n_samples[key] + window_dim - 1, self.model_def['Inputs'][key]['dim']),
                    dtype=np.float32).tolist()

        n_samples_input = {}
        for key in inputs.keys():
            if key in missing_inputs:
                n_samples_input[key] = 1
            else:
                n_samples_input[key] = len(inputs[key]) if sampled else len(inputs[key]) - self.input_n_samples[key] + 1

        # Vettore di input
        if num_of_samples != 'auto':
            for key in inputs.keys():
                if key in model_inputs:
                    input_dim = self.model_def['Inputs'][key]['dim']
                elif key in model_states:
                    input_dim = self.model_def['States'][key]['dim']
                if input_dim > 1:
                    inputs[key] += [[0 for val in range(input_dim)] for val in
                                    range(num_of_samples - (len(inputs[key]) - self.input_n_samples[key] + 1))]
                else:
                    inputs[key] += [0 for val in range(num_of_samples - (len(inputs[key]) - self.input_n_samples[key] + 1))]
                #n_samples_input[key] = num_of_samples

        ## Warning the users about different time windows between samples
        if min_dim != max_dim:
            self.visualizer.warning(f'Different number of samples between inputs [MAX {max_din_key} = {max_dim}; MIN {min_din_key} = {min_dim}]')

        result_dict = {} ## initialize the resulting dictionary
        for key in self.model_def['Outputs'].keys():
            result_dict[key] = []

        ## Initialize the state variables
        if prediction_samples == None:
            # If the prediction sample is None the connection are removed
            self.model.init_states({}, connect = connect)
        else:
            self.model.init_states(self.model_def['States'], connect = connect, reset_states = False)

        ## Cycle through all the samples provided
        with torch.inference_mode():
            X = {}
            for i in range(window_dim):
                for key, val in inputs.items():
                    # If the prediction sample is None take the input
                    # If the prediction sample is auto and the sample is less than the available samples take the input
                    # Every prediction sample take the input
                    # Otherwise if the key is a state or a connect or a closed_loop variable keep the same input
                    # If the key is a state or connect input remove the input
                    if not (prediction_samples is None \
                        or ((prediction_samples is not None and prediction_samples != 'auto') and i % (prediction_samples + 1) == 0) \
                        or (prediction_samples == 'auto' and i < n_samples_input[key])):
                        if key in (closed_loop|connect).keys() or key in model_states:
                            if (key in model_states or key in connect.keys()) and key in X.keys():
                                del X[key]
                            continue
                    X[key] = torch.from_numpy(np.array(val[i])).to(torch.float32) if sampled else torch.from_numpy(
                            np.array(val[i:i + self.input_n_samples[key]])).to(torch.float32)

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

                ## Reset the state variable
                if  prediction_samples is None:
                    ## If prediction sample is None the state is reset every step
                    self.model.reset_states(X, only=False)
                    self.model.reset_connect_variables(connect, X, only=False)
                elif prediction_samples == 'auto':
                    ## If prediction sample is auto is reset with the available samples
                    self.model.reset_states(X)
                    self.model.reset_connect_variables(connect, X)
                else:
                    ## Otherwise the variable are reset every prediction samples
                    if i%(prediction_samples+1) == 0:
                        self.model.reset_states(X, only=False)
                        self.model.reset_connect_variables(connect, X, only=False)

                result, _ = self.model(X)

                ## Update the recurrent variable
                for close_in, out_var in closed_loop.items():
                    #if i >= input_windows[close_in]-1:
                    dim = result[out_var].shape[1]  ## take the output time dimension
                    X[close_in] = torch.roll(X[close_in], shifts=-1, dims=1) ## Roll the time window
                    X[close_in][:, -dim:, :] = result[out_var] ## substitute with the predicted value

                ## Append the prediction of the current sample to the result dictionary
                for key in self.model_def['Outputs'].keys():
                    if result[key].shape[-1] == 1:
                        result[key] = result[key].squeeze(-1)
                        if result[key].shape[-1] == 1:
                            result[key] = result[key].squeeze(-1)
                    result_dict[key].append(result[key].detach().squeeze(dim=0).tolist())

        return result_dict

    def getSamples(self, dataset, index = None, window=1):
        if index is None:
            index = random.randint(0, self.num_of_samples[dataset] - window)
        if self.data_loaded:
            result_dict = {}
            for key in (self.model_def['Inputs'].keys() | self.model_def['States'].keys()):
                result_dict[key] = []
            for idx in range(window):
                for key ,samples in self.data[dataset].items():
                    if key in (self.model_def['Inputs'].keys() | self.model_def['States'].keys()):
                        result_dict[key].append(samples[index+idx])
            return result_dict
        else:
            print('The Dataset must first be loaded using <loadData> function!')
            return {}


    def addConnect(self, stream_out, state_list_in):
        self.model_def.addConnect(stream_out, state_list_in)

    def addClosedLoop(self, stream_out, state_list_in):
        self.model_def.addClosedLoop(stream_out, state_list_in)

    def addModel(self, name, stream_list):
        self.model_def.addModel(name, stream_list)

    def removeModel(self, name_list):
        self.model_def.removeModel(name_list)

    def addMinimize(self, name, streamA, streamB, loss_function='mse'):
        self.model_def.addMinimize(name, streamA, streamB, loss_function)
        self.visualizer.showaddMinimize(name)

    def removeMinimize(self, name_list):
        self.model_def.removeMinimize(name_list)


    # Use this function to get the parameters form the torch model and set the model_def_values
    def __get_torch_model(self, clear_model = False):
        if self.model is not None and clear_model == False:
            self.model_def_values = copy.deepcopy(self.model_def)
            for key in self.model_def_values['Parameters'].keys():
                if key in self.model.all_parameters:
                    self.model_def_values['Parameters'][key]['values'] = self.model.all_parameters[key].tolist()
                    if 'init_fun' in self.model_def_values['Parameters'][key]:
                        del self.model_def_values['Parameters'][key]['init_fun']
            model_def = copy.deepcopy(self.model_def_values)
        else:
            model_def = copy.deepcopy(self.model_def)
        return model_def

    # Use this function to create the torch model from a model_def
    def __neuralize_model(self, model_def):
        check(model_def['Inputs'] | model_def['States'] != {}, RuntimeError, "No model is defined!")
        json_inputs = model_def['Inputs'] | model_def['States']

        for key,value in model_def['States'].items():
            check(closedloop_name in model_def['States'][key] or connect_name in model_def['States'][key],
                  KeyError, f'Update function is missing for state {key}. Use Connect or ClosedLoop to update the state.')

        input_ns_backward = {key:value['ns'][0] for key, value in (model_def['Inputs']|model_def['States']).items()}
        input_ns_forward = {key:value['ns'][1] for key, value in (model_def['Inputs']|model_def['States']).items()}
        for key, value in json_inputs.items():
            self.input_n_samples[key] = input_ns_backward[key] + input_ns_forward[key]
        self.max_n_samples = max(input_ns_backward.values()) + max(input_ns_forward.values())

        ## Build the network
        self.model = Model(model_def)
        self.neuralized = True
        self.traced = False

    # Use this function for finilize the model_def with the samples time and build the torch model
    def neuralizeModel(self, sample_time = None, clear_model = False):
        self.model_def.setBuildWindow(sample_time)
        if self.model_def['Info']['ns'][0] < 0:
            self.visualizer.warning(
                f"The input is only in the far past the max_samples_backward is: {self.model_def['Info']['ns'][0]}")
        if self.model_def['Info']['ns'][1] < 0:
            self.visualizer.warning(
                f"The input is only in the far future the max_sample_forward is: {self.model_def['Info']['ns'][1]}")

        # Set sample time on the model_def
        if sample_time is None:
            if 'SampleTime' not in self.model_def or self.model_def['SampleTime'] == 0:
                sample_time = 1
            else:
                sample_time = self.model_def['SampleTime']
        else:
            check(sample_time > 0, RuntimeError, 'Sample time must be strictly positive!')
        self.model_def["SampleTime"] = sample_time

        ## Get the trained model
        # To be removed
        model_def = self.__get_torch_model(clear_model)
        # To be removed
        self.__neuralize_model(model_def)
        self.__get_torch_model(clear_model)
        self.visualizer.showModel(model_def.model_def)
        self.visualizer.showModelInputWindow()
        self.visualizer.showBuiltModel()

    def loadData(self, name, source, format=None, skiplines=0, delimiter=',', header=None):
        check(self.neuralized, ValueError, "The network is not neuralized.")
        check(delimiter in ['\t', '\n', ';', ',', ' '], ValueError, 'delimiter not valid!')

        json_inputs = self.model_def['Inputs'] | self.model_def['States']
        model_inputs = list(json_inputs.keys())
        ## Initialize the dictionary containing the data
        if name in list(self.data.keys()):
            self.visualizer.warning(f'Dataset named {name} already loaded! overriding the existing one..')
        self.data[name] = {}

        input_ns_backward = {key:value['ns'][0] for key, value in json_inputs.items()}
        input_ns_forward = {key:value['ns'][1] for key, value in json_inputs.items()}
        max_samples_backward = max(input_ns_backward.values())
        max_samples_forward = max(input_ns_forward.values())
        max_n_samples = max_samples_backward + max_samples_forward

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
                    back, forw = input_ns_backward[key], input_ns_forward[key]
                    ## Save as numpy array the data
                    data = df.iloc[:, idxs[0]:idxs[1]].to_numpy()
                    self.data[name][key] += [data[i-back:i+forw] for i in range(max_samples_backward, len(df)-max_samples_forward+1)]

            ## Stack the files
            for key in format_idx.keys():
                self.data[name][key] = np.stack(self.data[name][key])
                num_of_samples.append(self.data[name][key].shape[0])

        elif type(source) is dict:  ## we have a crafted dataset
            self.file_count = 1

            ## Check if the inputs are correct
            #assert set(model_inputs).issubset(source.keys()), f'The dataset is missing some inputs. Inputs needed for the model: {model_inputs}'

            # Merge a list of
            for key in model_inputs:
                if key not in source.keys():
                    continue

                self.data[name][key] = []  ## Initialize the dataset

                back, forw = input_ns_backward[key], input_ns_forward[key]
                for idx in range(len(source[key]) - max_n_samples+1):
                    self.data[name][key].append(source[key][idx + (max_samples_backward - back):idx + (max_samples_backward + forw)])

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

    def resetStates(self, values = None, only = True):
        self.model.init_states(self.model_def['States'], reset_states=False)
        self.model.reset_states(values, only)

    def __save_internal(self, key, value):
        self.internals[key] = tensor_to_list(value)

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

        ## Get parameter to be trained
        json_models = []
        models = self.__get_parameter(models=models)
        if 'Models' in self.model_def:
            json_models = list(self.model_def['Models'].keys()) if type(self.model_def['Models']) is dict else [self.model_def['Models']]
        if models is None:
            models = json_models
        self.run_training_params['models'] = models
        params_to_train = set()
        if isinstance(models, str):
            models = [models]
        for model in models:
            check(model in json_models, ValueError, f'The model {model} is not in the model definition')
            if type(self.model_def['Models']) is dict:
                params_to_train |= set(self.model_def['Models'][model]['Parameters'])
            else:
                params_to_train |= set(self.model_def['Parameters'].keys())

        # Get the optimizer
        if type(optimizer) is str:
            if optimizer == 'SGD':
                optimizer = SGD({},[])
            elif optimizer == 'Adam':
                optimizer = Adam({},[])
        else:
            check(issubclass(type(optimizer), Optimizer), TypeError,
                  "The optimizer must be an Optimizer or str")

        optimizer.set_params_to_train(self.model.all_parameters, params_to_train)

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
                    shuffle_data = None,
                    early_stopping = None, early_stopping_params = None,
                    minimize_gain = None,
                    num_of_epochs = None,
                    train_batch_size = None, val_batch_size = None, test_batch_size = None,
                    optimizer = None,
                    lr = None, lr_param = None,
                    optimizer_params = None, optimizer_defaults = None,
                    training_params = None,
                    add_optimizer_params = None, add_optimizer_defaults = None
                   ):

        check(self.data_loaded, RuntimeError, 'There is no data loaded! The Training will stop.')
        check(list(self.model.parameters()), RuntimeError, 'There are no modules with learnable parameters! The Training will stop.')

        ## Get running parameter from dict
        self.run_training_params = copy.deepcopy(self.__get_train_parameters(training_params))

        ## Get connect and closed_loop
        prediction_samples = self.__get_parameter(prediction_samples = prediction_samples)
        check(prediction_samples >= 0, KeyError, 'The sample horizon must be positive!')

        ## Check close loop and connect
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
        if early_stopping:
            self.run_training_params['early_stopping'] = early_stopping.__name__
        early_stopping_params = self.__get_parameter(early_stopping_params = early_stopping_params)

        ## Get dataset for training
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

            ## Collect the split sizes
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
        self.run_training_params['minimizers'] = {}
        for name, values in self.model_def['Minimizers'].items():
            self.losses[name] = CustomLoss(values['loss'])
            self.run_training_params['minimizers'][name] = {}
            self.run_training_params['minimizers'][name]['A'] = values['A']
            self.run_training_params['minimizers'][name]['B'] = values['B']
            self.run_training_params['minimizers'][name]['loss'] = values['loss']
            if name in minimize_gain:
                self.run_training_params['minimizers'][name]['gain'] = minimize_gain[name]

        ## Clean the dict of the training parameter
        del self.run_training_params['minimize_gain']
        del self.run_training_params['lr']
        del self.run_training_params['lr_param']
        if not recurrent_train:
            del self.run_training_params['connect']
            del self.run_training_params['closed_loop']
            del self.run_training_params['step']
            del self.run_training_params['prediction_samples']
        if early_stopping is None:
            del self.run_training_params['early_stopping']
            del self.run_training_params['early_stopping_params']

        ## Create the train, validation and test loss dictionaries
        train_losses, val_losses, test_losses = {}, {}, {}
        for key in self.model_def['Minimizers'].keys():
            train_losses[key] = []
            if n_samples_val > 0:
                val_losses[key] = []

        ## Check the needed keys are in the datasets
        keys = set(self.model_def['Inputs'].keys())
        keys |= {value['A'] for value in self.model_def['Minimizers'].values()}|{value['B'] for value in self.model_def['Minimizers'].values()}
        keys -= set(self.model_def['Relations'].keys())
        keys -= set(self.model_def['States'].keys())
        keys -= set(self.model_def['Outputs'].keys())
        if 'connect' in self.run_training_params:
            keys -= set(self.run_training_params['connect'].keys())
        if 'closed_loop' in self.run_training_params:
            keys -= set(self.run_training_params['closed_loop'].keys())
        check(set(keys).issubset(set(XY_train.keys())), KeyError, f"Not all the mandatory keys {keys} are present in the training dataset {set(XY_train.keys())}.")

        # Show the training params
        check((n_samples_train - train_batch_size - prediction_samples + 1) > 0, ValueError, f"The number of available sample are (n_samples_train - train_batch_size - prediction_samples + 1) = {(n_samples_train - train_batch_size - prediction_samples + 1)}.")
        self.visualizer.showTrainParams()

        import time
        ## start the train timer
        start = time.time()
        self.visualizer.showStartTraining()
        for epoch in range(num_of_epochs):
            ## TRAIN
            self.model.train()
            if recurrent_train:
                losses = self.__recurrentTrain(XY_train, n_samples_train, train_batch_size, minimize_gain, prediction_samples, closed_loop, step, connect, shuffle=shuffle_data, train=True)
            else:
                losses = self.__Train(XY_train,n_samples_train, train_batch_size, minimize_gain, shuffle=shuffle_data, train=True)
            ## save the losses
            for ind, key in enumerate(self.model_def['Minimizers'].keys()):
                train_losses[key].append(torch.mean(losses[ind]).tolist())

            if n_samples_val > 0:
                ## VALIDATION
                self.model.eval()
                if recurrent_train:
                    losses = self.__recurrentTrain(XY_val, n_samples_val, val_batch_size, minimize_gain, prediction_samples, closed_loop, step, connect, shuffle=False, train=False)
                else:
                    losses = self.__Train(XY_val, n_samples_val, val_batch_size, minimize_gain, shuffle=False, train=False)
                ## save the losses
                for ind, key in enumerate(self.model_def['Minimizers'].keys()):
                    val_losses[key].append(torch.mean(losses[ind]).tolist())

            ## Early-stopping
            if early_stopping:
                if early_stopping(train_losses, val_losses, early_stopping_params):
                    self.visualizer.warning('Stopping the training..')
                    break

            ## Visualize the training...
            self.visualizer.showTraining(epoch, train_losses, val_losses)
            self.visualizer.showWeightsInTrain(epoch = epoch)

        ## Save the training time
        end = time.time()
        ## Visualize the training time
        self.visualizer.showEndTraining(num_of_epochs-1, train_losses, val_losses)
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
            for ind, key in enumerate(self.model_def['Minimizers'].keys()):
                test_losses[key] = torch.mean(losses[ind]).tolist()

        self.train_losses = train_losses
        self.val_losses = val_losses
        self.test_losses = test_losses

        self.resultAnalysis(train_dataset, XY_train, connect, closed_loop)
        self.visualizer.showResult(train_dataset)
        if self.run_training_params['n_samples_val'] > 0:
            self.resultAnalysis(validation_dataset, XY_val, connect, closed_loop)
            self.visualizer.showResult(validation_dataset)
        if self.run_training_params['n_samples_test'] > 0:
            self.resultAnalysis(test_dataset, XY_test, connect, closed_loop)
            self.visualizer.showResult(test_dataset)

        # if self.run_training_params['n_samples_test'] > 0:
        #     self.ExportReport(XY_test, train_loss=train_losses, val_loss=val_losses)
        # elif self.run_training_params['n_samples_val'] > 0:
        #     self.ExportReport(XY_val, train_loss=train_losses, val_loss=val_losses)
        self.visualizer.showResults()

        ## Get trained model from torch and set the model_def_values
        self.__get_torch_model()

    def __recurrentTrain(self, data, n_samples, batch_size, loss_gains, prediction_samples, closed_loop, step, connect, shuffle=True, train=True):
        ## Sample Shuffle
        initial_value = random.randint(0, step - 1) if shuffle else 0

        ## Initialize the train losses vector
        aux_losses = torch.zeros([len(self.model_def['Minimizers']), n_samples//batch_size])

        json_inputs = self.model_def['Inputs'] | self.model_def['States']
        input_ns_backward = {key:value['ns'][0] for key, value in json_inputs.items()}

        ## +1 means that n_samples = 1 - batch_size = 1 - prediction_samples = 1 + 1 = 0 # zero epochs
        ## +1 means that n_samples = 2 - batch_size = 1 - prediction_samples = 1 + 1 = 1 # one epochs
        for idx in range(initial_value, (n_samples - batch_size - prediction_samples + 1), (batch_size + step - 1)):
            if train:
                self.optimizer.zero_grad() ## Reset the gradient

            ## Build the input tensor
            XY = {key: val[idx:idx+batch_size] for key, val in data.items()}
            ## collect the horizon labels
            XY_horizon = {key: val[idx:idx+batch_size+prediction_samples] for key, val in data.items()}
            horizon_losses = {ind: [] for ind in range(len(self.model_def['Minimizers']))}

            ## Reset state variables with zeros or using inputs
            self.model.reset_states(XY, only = False)
            self.model.reset_connect_variables(connect, XY, only= False)

            for horizon_idx in range(prediction_samples + 1):
                out, minimize_out = self.model(XY)  ## Forward pass
                if self.log_internal:
                    self.__save_internal('inout_'+str(idx)+'_'+str(horizon_idx),{'XY':XY,'out':out,'state':self.model.states,'param':self.model.all_parameters,'connect':self.model.connect_variables})

                ## Loss Calculation
                for ind, (key, value) in enumerate(self.model_def['Minimizers'].items()):
                    loss = self.losses[key](minimize_out[value['A']], minimize_out[value['B']])
                    loss = (loss * loss_gains[key]) if key in loss_gains.keys() else loss  ## Multiply by the gain if necessary
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
                            XY[key] = torch.roll(XY[key], shifts=-1, dims=1) ## Roll the time window
                            XY[key][:, input_ns_backward[key]-dim:input_ns_backward[key], :] = out[closed_loop[key]] ## substitute with the predicted value
                            XY[key][:, input_ns_backward[key]:, :] = XY_horizon[key][horizon_idx:horizon_idx+batch_size, input_ns_backward[key]:, :]  ## fill the remaining values from the dataset
                        else: ## the input is not recurrent
                            XY[key] = torch.roll(XY[key], shifts=-1, dims=0)  ## Roll the sample window
                            XY[key][-1] = XY_horizon[key][batch_size+horizon_idx]  ## take the next sample from the dataset

            ## Calculate the total loss
            total_loss = 0
            for ind in range(len(self.model_def['Minimizers'])):
                total_loss += sum(horizon_losses[ind])/(prediction_samples+1)
                aux_losses[ind][idx//batch_size] = total_loss.item()

            ## Gradient Step
            if train:
                total_loss.backward() ## Backpropagate the error
                self.optimizer.step()
                self.visualizer.showWeightsInTrain(batch = idx/(batch_size + step - 1))

        ## return the losses
        return aux_losses
    
    def __Train(self, data, n_samples, batch_size, loss_gains, shuffle=True, train=True):
        if shuffle:
            randomize = torch.randperm(n_samples)
            data = {key: val[randomize] for key, val in data.items()}
        ## Initialize the train losses vector
        aux_losses = torch.zeros([len(self.model_def['Minimizers']),n_samples//batch_size])
        for idx in range(0, (n_samples - batch_size + 1), batch_size):
            ## Build the input tensor
            XY = {key: val[idx:idx+batch_size] for key, val in data.items()}
            ## Reset gradient
            if train:
                self.optimizer.zero_grad()
            ## Model Forward
            _, minimize_out = self.model(XY)  ## Forward pass
            ## Loss Calculation
            total_loss = 0
            for ind, (key, value) in enumerate(self.model_def['Minimizers'].items()):
                loss = self.losses[key](minimize_out[value['A']], minimize_out[value['B']])
                loss = (loss * loss_gains[key]) if key in loss_gains.keys() else loss  ## Multiply by the gain if necessary
                aux_losses[ind][idx//batch_size]= loss.item()
                total_loss += loss
            ## Gradient step
            if train:
                total_loss.backward()
                self.optimizer.step()
                self.visualizer.showWeightsInTrain(batch = idx/batch_size)

        ## return the losses
        return aux_losses

    def resultAnalysis(self, name_data, XY_data, connect, closed_loop):
        import warnings
        with torch.inference_mode():
            ## Init model for retults analysis
            self.model.eval()
            self.performance[name_data] = {}
            self.prediction[name_data] = {}
            A = {}
            B = {}
            aux_losses = {}

            ## Update State variables if necessary
            self.model.reset_states(XY_data, only = False)
            self.model.reset_connect_variables(connect, XY_data, only=False)

            _, minimize_out = self.model(XY_data)
            for ind, (key, value) in enumerate(self.model_def['Minimizers'].items()):
                A[key] = minimize_out[value['A']]
                B[key] = minimize_out[value['B']]
                loss = self.losses[key](minimize_out[value['A']], minimize_out[value['B']])
                aux_losses[key] = loss.detach().numpy()

            for ind, (key, value) in enumerate(self.model_def['Minimizers'].items()):
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
            self.performance[name_data]['total']['fvu'] = np.mean([self.performance[name_data][key]['fvu']['total'] for key in self.model_def['Minimizers'].keys()])
            self.performance[name_data]['total']['aic'] = np.mean([self.performance[name_data][key]['aic']['value']for key in self.model_def['Minimizers'].keys()])

    def getWorkspace(self):
        return self.exporter.getWorkspace()

    def saveModel(self, name = 'net', model_path = None):
        if self.model_def is not None:
            self.exporter.saveModel(self.model_def.model_def, name, model_path)
        # if self.model_def_values is not None:
        #     self.exporter.saveModel(self.model_def_values, name + '_trained', model_path)

    def loadModel(self, name = None, model_folder = None):
        if name is None:
            name = 'net'
            model_def = self.exporter.loadModel(name + '_trained', model_folder)
            if model_def is None:
                 model_def = self.exporter.loadModel(name, model_folder)
        else:
            model_def = self.exporter.loadModel(name, model_folder)

        if model_def:
            self.model_def.update(model_def)
            self.__neuralize_model(self.model_def)

    def exportPythonModel(self, name = 'net', model_path = None):
        check(self.model_def['States'] == {}, TypeError, "The network has state variables. The export to python is not possible.")
        if self.model_def is not None:
            self.exporter.saveModel(self.model_def, name, model_path)
            self.exporter.exportPythonModel(name, model_path)

    def importPythonModel(self, name = None, model_folder = None):
        if name is None:
            name = 'net'
        self.model_def_loaded = self.exporter.loadModel(name, model_folder)
        self.__update_model(model_def=self.model_def_loaded)
        self.__neuralize_model(self.model_def_loaded)
        self.model = self.exporter.importPythonModel(name, model_folder)
        self.traced = True
        self.__get_torch_model()

    def exportONNX(self, inputs_order, outputs_order,  models = None, name = 'net', model_path = None):
        check(self.model_def is not None, TypeError, "The network has not been defined.")
        old_model_def = copy.deepcopy(self.model_def)
        model_def = copy.deepcopy(MAIN_JSON)
        model_def['SampleTime'] = self.model_def['SampleTime']
        if models is None:
            self.__update_model(model_def, minimize_dict={})
        else:
            name += '_' + '_'.join(models)
            self.__update_model(model_def, {key: self.model_dict[key] for key in models if key in self.model_dict}, minimize_dict={})
        model_def = self.__get_torch_model()
        self.__neuralize_model(model_def)
        self.exporter.exportONNX(inputs_order, outputs_order, name, model_path)
        self.__neuralize_model(old_model_def)

    def exportReport(self, name = 'net', model_path = None):
        self.exporter.exportReport(name, model_path)
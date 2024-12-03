# Extern packages
import copy, os, json, random, torch
import numpy as np
import pandas as pd

# Neu4mes packages
from neu4mes.visualizer import TextVisualizer, Visualizer
from neu4mes.loss import CustomLoss
from neu4mes.model import Model
from neu4mes.utils import check, tensor_to_list, argmax_dict, argmin_dict
from neu4mes.optimizer import Optimizer, SGD, Adam
from neu4mes.exporter import Exporter, StandardExporter
from neu4mes.modeldef import ModelDef

from neu4mes.logger import logging, Neu4MesLogger
log = Neu4MesLogger(__name__, logging.INFO)

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
            self.exporter = StandardExporter(workspace, self.visualizer, save_history)
        elif exporter != None:
            self.exporter = exporter
        else:
            self.exporter = Exporter()

        ## Set the random seed for reproducibility
        if seed is not None:
            self.resetSeed(seed)

        # Save internal
        self.log_internal = log_internal
        if self.log_internal == True:
            self.internals = {}

        # Models definition
        self.model_def = ModelDef()
        self.input_n_samples = {}
        self.max_n_samples = 0
        self.neuralized = False
        self.traced = False
        self.model = None
        self.states = {}

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
            'select_model' : 'last', 'select_model_params' : {},
            'minimize_gain' : {},
            'num_of_epochs': 100,
            'train_batch_size' : 128, 'val_batch_size' : None, 'test_batch_size' : None,
            'optimizer' : 'Adam',
            'lr' : 0.001, 'lr_param' : {},
            'optimizer_params' : [], 'add_optimizer_params' : [],
            'optimizer_defaults' : {}, 'add_optimizer_defaults' : {}
        }

        # Optimizer
        self.optimizer = None

        # Training Losses
        self.loss_functions = {}

        # Validation Parameters
        self.training = {}
        self.performance = {}
        self.prediction = {}

    def resetSeed(self, seed):
        torch.manual_seed(seed)  ## set the pytorch seed
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)  ## set the random module seed
        np.random.seed(seed)  ## set the numpy seed

    def __call__(self, inputs={}, sampled=False, closed_loop={}, connect={}, prediction_samples='auto', num_of_samples=None): ##, align_input=False):
        inputs = copy.deepcopy(inputs)
        closed_loop = copy.deepcopy(closed_loop)
        connect = copy.deepcopy(connect)

        ## Check neuralize
        check(self.neuralized, RuntimeError, "The network is not neuralized.")

        ## Check closed loop integrity
        for close_in, close_out in closed_loop.items():
            check(close_in in self.model_def['Inputs'], ValueError, f'the tag {close_in} is not an input variable.')
            check(close_out in self.model_def['Outputs'], ValueError, f'the tag {close_out} is not an output of the network')

        ## List of keys
        model_inputs = list(self.model_def['Inputs'].keys())
        model_states = list(self.model_def['States'].keys())
        state_closed_loop = [key for key, value in self.model_def['States'].items() if 'closedLoop' in value.keys()] + list(closed_loop.keys())
        state_connect = [key for key, value in self.model_def['States'].items() if 'connect' in value.keys()] + list(connect.keys())
        extra_inputs = list(set(list(inputs.keys())) - set(model_inputs) - set(model_states))
        non_mandatory_inputs = state_closed_loop + state_connect 
        mandatory_inputs = list(set(model_inputs) - set(non_mandatory_inputs))

        ## Remove extra inputs
        for key in extra_inputs:
            log.warning(f'The provided input {key} is not used inside the network. the inference will continue without using it')
            del inputs[key]

        ## Get the number of data windows for each input/state
        num_of_windows = {key: len(value) for key, value in inputs.items()} if sampled else {key: len(value) - self.input_n_samples[key] + 1 for key, value in inputs.items()}

        ## Get the maximum inference window
        if num_of_samples:
            window_dim = num_of_samples
            for key in inputs.keys():
                input_dim = self.model_def['Inputs'][key]['dim'] if key in model_inputs else self.model_def['States'][key]['dim']
                if input_dim > 1:
                    inputs[key] += [[0 for _ in range(input_dim)] for _ in range(num_of_samples - (len(inputs[key]) - self.input_n_samples[key] + 1))]
                else:
                    inputs[key] += [0 for _ in range(num_of_samples - (len(inputs[key]) - self.input_n_samples[key] + 1))]
        elif inputs:
            windows = []
            for key in inputs.keys():
                if key in mandatory_inputs:
                    n_samples = len(inputs[key]) if sampled else len(inputs[key]) - self.model_def['Inputs'][key]['ntot'] + 1
                    windows.append(n_samples)
            if not windows:
                for key in inputs.keys():
                    if key in non_mandatory_inputs:
                        if key in model_inputs:
                            n_samples = len(inputs[key]) if sampled else len(inputs[key]) - self.model_def['Inputs'][key]['ntot'] + 1
                        else:
                            n_samples = len(inputs[key]) if sampled else len(inputs[key]) - self.model_def['States'][key]['ntot'] + 1
                        windows.append(n_samples)
            window_dim = min(windows) if windows else 0
        else: ## No inputs
            window_dim = 1 if non_mandatory_inputs else 0
        check(window_dim > 0, StopIteration, f'Missing samples in the input window')

        if len(set(num_of_windows.values())) > 1:
            max_ind_key, max_dim = argmax_dict(num_of_windows)
            min_ind_key, min_dim = argmin_dict(num_of_windows)
            log.warning(f'Different number of samples between inputs [MAX {num_of_windows[max_ind_key]} = {max_dim}; MIN {num_of_windows[min_ind_key]} = {min_dim}]')

        ## Autofill the missing inputs
        provided_inputs = list(inputs.keys())
        missing_inputs = list(set(mandatory_inputs) - set(provided_inputs))
        if missing_inputs:
            log.warning(f'Inputs not provided: {missing_inputs}. Autofilling with zeros..')
            for key in missing_inputs:
                inputs[key] = np.zeros(shape=(self.input_n_samples[key] + window_dim - 1, self.model_def['Inputs'][key]['dim']),dtype=np.float32).tolist()

        ## Transform inputs in 3D Tensors
        for key, val in inputs.items():
            input_dim = self.model_def['Inputs'][key]['dim'] if key in model_inputs else self.model_def['States'][key]['dim']
            inputs[key] = torch.from_numpy(np.array(inputs[key])).to(torch.float32)

            if input_dim > 1:
                correct_dim = 3 if sampled else 2
                check(len(inputs[key].shape) == correct_dim, ValueError,f'The input {key} must have {correct_dim} dimensions')
                check(inputs[key].shape[correct_dim - 1] == input_dim, ValueError,f'The second dimension of the input "{key}" must be equal to {input_dim}')

            if input_dim == 1 and inputs[key].shape[-1] != 1: ## add the input dimension
                inputs[key] = inputs[key].unsqueeze(-1)
            if inputs[key].ndim <= 1: ## add the batch dimension
                inputs[key] = inputs[key].unsqueeze(0)
            if inputs[key].ndim <= 2: ## add the time dimension
                inputs[key] = inputs[key].unsqueeze(0)

        ## initialize the resulting dictionary
        result_dict = {}
        for key in self.model_def['Outputs'].keys():
            result_dict[key] = []

        ## Inference
        with torch.inference_mode():
            self.model.eval()
            ## Update with virtual states
            if prediction_samples is not None:
                self.model.update(closed_loop=closed_loop, connect=connect)
            else:
                prediction_samples = 0
            X = {}
            count = 0
            first = True
            for idx in range(window_dim):
                ## Get mandatory data inputs
                for key in mandatory_inputs:
                    X[key] = inputs[key][idx:idx+1] if sampled else inputs[key][:, idx:idx + self.input_n_samples[key]]
                ## reset states
                if count == 0 or prediction_samples=='auto':
                    count = prediction_samples
                    for key in non_mandatory_inputs: ## Get non mandatory data (from inputs, from states, or with zeros)
                        ## if prediction_samples is 'auto' and i have enough samples
                        ## if prediction_samples is NOT 'auto' but i have enough extended window (with zeros)
                        if (key in inputs.keys() and prediction_samples == 'auto' and idx < num_of_windows[key]) or (key in inputs.keys() and prediction_samples != 'auto' and idx < inputs[key].shape[1]):
                            X[key] = inputs[key][idx:idx+1].clone() if sampled else inputs[key][:, idx:idx + self.input_n_samples[key]].clone()
                        ## if im in the first reset
                        ## if i have a state in memory
                        ## if i have prediction_samples = 'auto' and not enough samples
                        elif (key in self.states.keys() and (first or prediction_samples == 'auto')) and (prediction_samples == 'auto' or prediction_samples == None):
                            X[key] = self.states[key]
                        else: ## if i have no samples and no states
                            window_size = self.input_n_samples[key]
                            dim = self.model_def['Inputs'][key]['dim'] if key in model_inputs else self.model_def['States'][key]['dim']
                            X[key] = torch.zeros(size=(1, window_size, dim), dtype=torch.float32, requires_grad=False)
                            self.states[key] = X[key]
                    first = False
                else:
                    count -= 1
                ## Forward pass
                result, _, out_closed_loop, out_connect = self.model(X)

                ## Append the prediction of the current sample to the result dictionary
                for key in self.model_def['Outputs'].keys():
                    if result[key].shape[-1] == 1:
                        result[key] = result[key].squeeze(-1)
                        if result[key].shape[-1] == 1:
                            result[key] = result[key].squeeze(-1)
                    result_dict[key].append(result[key].detach().squeeze(dim=0).tolist())

                ## Update closed_loop and connect
                if prediction_samples:
                    for key, val in out_closed_loop.items():
                        shift = val.shape[1]  ## take the output time dimension
                        X[key] = torch.roll(X[key], shifts=-1, dims=1) ## Roll the time window
                        X[key][:, -shift:, :] = val ## substitute with the predicted value
                        self.states[key] = X[key]
                    for key, val in out_connect.items():
                        X[key] = val
                        self.states[key] = X[key]

        ## Remove virtual states
        for key in (connect.keys() | closed_loop.keys()):
            if key in self.states.keys():
                del self.states[key]
        
        return result_dict

    def getSamples(self, dataset, index = None, window=1):
        if index is None:
            index = random.randint(0, self.num_of_samples[dataset] - window)
        check(self.data_loade, ValueError, 'The Dataset must first be loaded using <loadData> function!')
        if self.data_loaded:
            result_dict = {}
            for key in (self.model_def['Inputs'].keys() | self.model_def['States'].keys()):
                result_dict[key] = []
            for idx in range(window):
                for key ,samples in self.data[dataset].items():
                    if key in (self.model_def['Inputs'].keys() | self.model_def['States'].keys()):
                        result_dict[key].append(samples[index+idx])
            return result_dict

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

    def resetStates(self, states=[], batch=1):
        if states: ## reset only specific states
            for key in states:
                window_size = self.input_n_samples[key]
                dim = self.model_def['States'][key]['dim']
                self.states[key] = torch.zeros(size=(batch, window_size, dim), dtype=torch.float32, requires_grad=False)
        else: ## reset all states
            self.states = {}
            for key, state in self.model_def['States'].items():
                window_size = self.input_n_samples[key]
                dim = state['dim']
                self.states[key] = torch.zeros(size=(batch, window_size, dim), dtype=torch.float32, requires_grad=False)

    def neuralizeModel(self, sample_time = None, clear_model = False, model_def = None):
        if model_def is not None:
            check(sample_time == None, ValueError, 'The sample_time must be None if a model_def is provided')
            check(clear_model == False, ValueError, 'The clear_model must be False if a model_def is provided')
            self.model_def = ModelDef(model_def)
        else:
            if clear_model:
                self.model_def.update()
            else:
                self.model_def.updateParameters(self.model)

        self.model_def.setBuildWindow(sample_time)
        self.model = Model(self.model_def.json)

        input_ns_backward = {key:value['ns'][0] for key, value in (self.model_def['Inputs']|self.model_def['States']).items()}
        input_ns_forward = {key:value['ns'][1] for key, value in (self.model_def['Inputs']|self.model_def['States']).items()}
        self.input_n_samples = {}
        for key, value in (self.model_def['Inputs'] | self.model_def['States']).items():
            self.input_n_samples[key] = input_ns_backward[key] + input_ns_forward[key]
        self.max_n_samples = max(input_ns_backward.values()) + max(input_ns_forward.values())

        ## Initialize States 
        self.resetStates()

        self.neuralized = True
        self.traced = False
        self.visualizer.showModel(self.model_def.json)
        self.visualizer.showModelInputWindow()
        self.visualizer.showBuiltModel()

    def loadData(self, name, source, format=None, skiplines=0, delimiter=',', header=None):
        check(self.neuralized, ValueError, "The network is not neuralized.")
        check(delimiter in ['\t', '\n', ';', ',', ' '], ValueError, 'delimiter not valid!')

        json_inputs = self.model_def['Inputs'] | self.model_def['States']
        model_inputs = list(json_inputs.keys())
        ## Initialize the dictionary containing the data
        if name in list(self.data.keys()):
            log.warning(f'Dataset named {name} already loaded! overriding the existing one..')
        self.data[name] = {}

        input_ns_backward = {key:value['ns'][0] for key, value in json_inputs.items()}
        input_ns_forward = {key:value['ns'][1] for key, value in json_inputs.items()}
        max_samples_backward = max(input_ns_backward.values())
        max_samples_forward = max(input_ns_forward.values())
        max_n_samples = max_samples_backward + max_samples_forward

        num_of_samples = {}
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
                check(False,StopIteration, f'ERROR: The path "{source}" does not exist!')
                return
            self.file_count = len(files)

            ## Cycle through all the files
            for file in files:
                try:
                    ## read the csv
                    df = pd.read_csv(os.path.join(source,file), skiprows=skiplines, delimiter=delimiter, header=header)
                except:
                    log.warning(f'Cannot read file {os.path.join(source,file)}')
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
                num_of_samples[key] = self.data[name][key].shape[0]

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
                num_of_samples[key] = self.data[name][key].shape[0]

        # Check dim of the samples
        check(len(set(num_of_samples.values())) == 1, ValueError,
              f"The number of the sample of the dataset {name} are not the same for all input in the dataset: {num_of_samples}")
        self.num_of_samples[name] = num_of_samples[list(num_of_samples.keys())[0]]

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

        if self.run_training_params['recurrent_train']:
            if self.run_training_params['train_batch_size'] > self.run_training_params['n_samples_train']:
                self.run_training_params['train_batch_size'] = self.run_training_params['n_samples_train'] - self.run_training_params['prediction_samples']
            if self.run_training_params['val_batch_size'] is None or self.run_training_params['val_batch_size'] > self.run_training_params['n_samples_val']:
                self.run_training_params['val_batch_size'] = max(0,self.run_training_params['n_samples_val'] - self.run_training_params['prediction_samples'])
            if self.run_training_params['test_batch_size'] is None or self.run_training_params['test_batch_size'] > self.run_training_params['n_samples_test']:
                self.run_training_params['test_batch_size'] = max(0,self.run_training_params['n_samples_test'] - self.run_training_params['prediction_samples'])
        else:
            if self.run_training_params['train_batch_size'] > self.run_training_params['n_samples_train']:
                self.run_training_params['train_batch_size'] = self.run_training_params['n_samples_train']
            if self.run_training_params['val_batch_size'] is None or self.run_training_params['val_batch_size'] > self.run_training_params['n_samples_val']:
                self.run_training_params['val_batch_size'] = self.run_training_params['n_samples_val']
            if self.run_training_params['test_batch_size'] is None or self.run_training_params['test_batch_size'] > self.run_training_params['n_samples_test']:
                self.run_training_params['test_batch_size'] = self.run_training_params['n_samples_test']

        check(self.run_training_params['train_batch_size'] > 0, ValueError, f'The auto train_batch_size ({self.run_training_params["train_batch_size"] }) = n_samples_train ({self.run_training_params["n_samples_train"]}) - prediction_samples ({self.run_training_params["prediction_samples"]}), must be greater than 0.')

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
                    select_model = None, select_model_params = None,
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
                log.warning(f'Recurrent train: closing the loop between the the input ports {input} and the output ports {output} for {prediction_samples} samples')
        elif connect:
            for connect_in, connect_out in connect.items():
                check(connect_in in self.model_def['Inputs'], ValueError, f'the tag {connect_in} is not an input variable.')
                check(connect_out in self.model_def['Outputs'], ValueError, f'the tag {connect_out} is not an output of the network')
                log.warning(f'Recurrent train: connecting the input ports {connect_in} with output ports {connect_out} for {prediction_samples} samples')
        elif self.model_def['States']: ## if we have state variables we have to do the recurrent train
            log.warning(f"Recurrent train: update States variables {list(self.model_def['States'].keys())} for {prediction_samples} samples")
        else:
            if prediction_samples != 0:
                log.warning(
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
                    XY_train[key] = torch.from_numpy(samples[:n_samples_train]).to(torch.float32)
                    XY_test[key] = torch.from_numpy(samples[n_samples_train:]).to(torch.float32)
                elif val_size != 0.0 and test_size == 0.0: ## we have only training and validation set
                    XY_train[key] = torch.from_numpy(samples[:n_samples_train]).to(torch.float32)
                    XY_val[key] = torch.from_numpy(samples[n_samples_train:]).to(torch.float32)
                else: ## we have training, validation and test set
                    XY_train[key] = torch.from_numpy(samples[:n_samples_train]).to(torch.float32)
                    XY_val[key] = torch.from_numpy(samples[n_samples_train:-n_samples_test]).to(torch.float32)
                    XY_test[key] = torch.from_numpy(samples[n_samples_train+n_samples_val:]).to(torch.float32)

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
                log.warning(f'Validation Dataset [{validation_dataset}] Not Loaded. The training will continue without validation')
            if test_dataset is not None and test_dataset not in datasets:
                log.warning(f'Test Dataset [{test_dataset}] Not Loaded. The training will continue without test')

            ## Split into train, validation and test
            XY_train, XY_val, XY_test = {}, {}, {}
            n_samples_train = self.num_of_samples[train_dataset]
            XY_train = {key: torch.from_numpy(val).to(torch.float32) for key, val in self.data[train_dataset].items()}
            if validation_dataset in datasets:
                n_samples_val = self.num_of_samples[validation_dataset]
                XY_val = {key: torch.from_numpy(val).to(torch.float32) for key, val in self.data[validation_dataset].items()}
            if test_dataset in datasets:
                n_samples_test = self.num_of_samples[test_dataset]
                XY_test = {key: torch.from_numpy(val).to(torch.float32) for key, val in self.data[test_dataset].items()}

        for key in XY_train.keys():
            assert n_samples_train == XY_train[key].shape[0], f'The number of train samples {n_samples_train}!={XY_train[key].shape[0]} not compliant.'
            if key in XY_val:
                assert n_samples_val == XY_val[key].shape[0], f'The number of val samples {n_samples_val}!={XY_val[key].shape[0]} not compliant.'
            if key in XY_test:
                assert n_samples_test == XY_test[key].shape[0], f'The number of test samples {n_samples_test}!={XY_test[key].shape[0]} not compliant.'

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
            self.loss_functions[name] = CustomLoss(values['loss'])
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

        # Evaluate the number of update for epochs and the unsued samples
        if recurrent_train:
            list_of_batch_indexes = range(0, (n_samples_train - train_batch_size - prediction_samples + 1), (train_batch_size + step - 1))
            check(n_samples_train - train_batch_size - prediction_samples + 1 > 0, ValueError,
                  f"The number of available sample are (n_samples_train ({n_samples_train}) - train_batch_size ({train_batch_size}) - prediction_samples ({prediction_samples}) + 1) = {n_samples_train - train_batch_size - prediction_samples + 1}.")
            update_per_epochs = (n_samples_train - train_batch_size - prediction_samples + 1)//(train_batch_size + step - 1) + 1
            unused_samples = n_samples_train - list_of_batch_indexes[-1] - train_batch_size - prediction_samples
        else:
            update_per_epochs =  (n_samples_train - train_batch_size)/train_batch_size + 1
            unused_samples = n_samples_train - update_per_epochs * train_batch_size

        self.run_training_params['update_per_epochs'] = update_per_epochs
        self.run_training_params['unused_samples'] = unused_samples

        ## Select the model
        select_model = self.__get_parameter(select_model = select_model)
        select_model_params = self.__get_parameter(select_model_params = select_model_params)
        selected_model_def = ModelDef(self.model_def.json)

        ## Show the training parameters
        self.visualizer.showTrainParams()

        import time
        ## start the train timer
        start = time.time()
        self.visualizer.showStartTraining()

        for epoch in range(num_of_epochs):
            ## TRAIN
            self.model.train()
            if recurrent_train:
                losses = self.__recurrentTrain(XY_train, n_samples_train, train_batch_size, minimize_gain, closed_loop, connect, prediction_samples, step, shuffle=shuffle_data, train=True)
            else:
                losses = self.__Train(XY_train,n_samples_train, train_batch_size, minimize_gain, shuffle=shuffle_data, train=True)
            ## save the losses
            for ind, key in enumerate(self.model_def['Minimizers'].keys()):
                train_losses[key].append(torch.mean(losses[ind]).tolist())

            if n_samples_val > 0:
                ## VALIDATION
                self.model.eval()
                if recurrent_train:
                    losses = self.__recurrentTrain(XY_val, n_samples_val, val_batch_size, minimize_gain, closed_loop, connect, prediction_samples, step, shuffle=False, train=False)
                else:
                    losses = self.__Train(XY_val, n_samples_val, val_batch_size, minimize_gain, shuffle=False, train=False)
                ## save the losses
                for ind, key in enumerate(self.model_def['Minimizers'].keys()):
                    val_losses[key].append(torch.mean(losses[ind]).tolist())

            ## Early-stopping
            if callable(early_stopping):
                if early_stopping(train_losses, val_losses, early_stopping_params):
                    log.info(f'Stopping the training at epoch {epoch} due to early stopping.')
                    break

            if callable(select_model):
                if select_model(train_losses, val_losses, select_model_params):
                    best_model_epoch = epoch
                    selected_model_def.updateParameters(self.model)

            ## Visualize the training...
            self.visualizer.showTraining(epoch, train_losses, val_losses)
            self.visualizer.showWeightsInTrain(epoch = epoch)

        ## Save the training time
        end = time.time()
        ## Visualize the training time
        for key in self.model_def['Minimizers'].keys():
            self.training[key] = {'train': train_losses[key]}
            if n_samples_val > 0:
                self.training[key]['val'] = val_losses[key]
        self.visualizer.showEndTraining(num_of_epochs-1, train_losses, val_losses)
        self.visualizer.showTrainingTime(end-start)

        ## Select the model
        if callable(select_model):
            log.info(f'Selected the model at the epoch {best_model_epoch+1}.')
            self.model = Model(selected_model_def)
        else:
            log.info('The selected model is the LAST model of the training.')

        self.resultAnalysis(train_dataset, XY_train, minimize_gain, closed_loop, connect,  prediction_samples, step, train_batch_size)
        if self.run_training_params['n_samples_val'] > 0:
            self.resultAnalysis(validation_dataset, XY_val, minimize_gain, closed_loop, connect,  prediction_samples, step, val_batch_size)
        if self.run_training_params['n_samples_test'] > 0:
            self.resultAnalysis(test_dataset, XY_test, minimize_gain, closed_loop, connect,  prediction_samples, step, test_batch_size)

        self.visualizer.showResults()

        ## Get trained model from torch and set the model_def
        self.model_def.updateParameters(self.model)

    def __recurrentTrain(self, data, n_samples, batch_size, loss_gains, closed_loop, connect, prediction_samples, step, shuffle, train=True):

        model_inputs = list(self.model_def['Inputs'].keys())
        state_closed_loop = [key for key, value in self.model_def['States'].items() if 'closedLoop' in value.keys()] + list(closed_loop.keys())
        state_connect = [key for key, value in self.model_def['States'].items() if 'connect' in value.keys()] + list(connect.keys())
        non_mandatory_inputs = state_closed_loop + state_connect 
        mandatory_inputs = list(set(model_inputs) - set(non_mandatory_inputs))

        ## shuffle TODO
        initial_value = 0  #np.random.randint(0, prediction_samples)
        n_available_samples = n_samples - batch_size - prediction_samples + 1
        list_of_batch_indexes = range(initial_value, n_available_samples, (batch_size + step - 1))

        ## Loss vector 
        aux_losses = torch.zeros([len(self.model_def['Minimizers']), len(list_of_batch_indexes)])

        ## Update with virtual states
        self.model.update(closed_loop=closed_loop, connect=connect)
        X = {}
        for batch_val, idx in enumerate(list_of_batch_indexes):
            if train:
                self.optimizer.zero_grad() ## Reset the gradient

            ## Reset 
            horizon_losses = {ind: [] for ind in range(len(self.model_def['Minimizers']))}
            for key in non_mandatory_inputs:
                if key in data.keys():
                ## with data
                    X[key] = data[key][idx:idx+batch_size]
                else: ## with zeros
                    window_size = self.input_n_samples[key]
                    dim = self.model_def['Inputs'][key]['dim'] if key in model_inputs else self.model_def['States'][key]['dim']
                    X[key] = torch.zeros(size=(batch_size, window_size, dim), dtype=torch.float32, requires_grad=False)
                    self.states[key] = X[key]

            for horizon_idx in range(prediction_samples + 1):
                ## Get data 
                for key in mandatory_inputs:
                    X[key] = data[key][idx+horizon_idx:idx+horizon_idx+batch_size]
                ## Forward pass
                out, minimize_out, out_closed_loop, out_connect = self.model(X)

                internals_dict = {'XY':X,'out':out,'param':self.model.all_parameters,'closedLoop':self.model.closed_loop_update,'connect':self.model.connect_update}

                ## Loss Calculation
                for ind, (key, value) in enumerate(self.model_def['Minimizers'].items()):
                    loss = self.loss_functions[key](minimize_out[value['A']], minimize_out[value['B']])
                    loss = (loss * loss_gains[key]) if key in loss_gains.keys() else loss  ## Multiply by the gain if necessary
                    horizon_losses[ind].append(loss)

                ## Update
                for key, val in out_closed_loop.items():
                    shift = val.shape[1]  ## take the output time dimension
                    X[key] = torch.roll(X[key], shifts=-1, dims=1) ## Roll the time window
                    X[key][:, -shift:, :] = val ## substitute with the predicted value
                    self.states[key] = X[key].clone()
                for key, value in out_connect.items():
                    X[key] = value
                    self.states[key] = X[key].clone()

                internals_dict['state'] = self.states
                if self.log_internal:
                    self.__save_internal('inout_'+str(idx)+'_'+str(horizon_idx),internals_dict)

            ## Calculate the total loss
            total_loss = 0
            for ind in range(len(self.model_def['Minimizers'])):
                loss = sum(horizon_losses[ind])/(prediction_samples+1)
                aux_losses[ind][batch_val] = loss.item()
                total_loss += loss

            ## Gradient Step
            if train:
                total_loss.backward() ## Backpropagate the error
                self.optimizer.step()
                self.visualizer.showWeightsInTrain(batch = batch_val)

        ## Remove virtual states
        for key in (connect.keys() | closed_loop.keys()):
            if key in self.states.keys():
                del self.states[key]

        ## return the losses
        return aux_losses
    
    def __Train(self, data, n_samples, batch_size, loss_gains, shuffle=True, train=True):
        check((n_samples - batch_size + 1) > 0, ValueError,
              f"The number of available sample are (n_samples_train - train_batch_size + 1) = {n_samples - batch_size + 1}.")
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
            _, minimize_out, _, _ = self.model(XY)  ## Forward pass
            ## Loss Calculation
            total_loss = 0
            for ind, (key, value) in enumerate(self.model_def['Minimizers'].items()):
                loss = self.loss_functions[key](minimize_out[value['A']], minimize_out[value['B']])
                loss = (loss * loss_gains[key]) if key in loss_gains.keys() else loss  ## Multiply by the gain if necessary
                aux_losses[ind][idx//batch_size] = loss.item()
                total_loss += loss
            ## Gradient step
            if train:
                total_loss.backward()
                self.optimizer.step()
                self.visualizer.showWeightsInTrain(batch = idx//batch_size)

        ## return the losses
        return aux_losses

    def resultAnalysis(self, dataset, data = None, minimize_gain = {}, closed_loop = {}, connect = {},  prediction_samples = None, step = 1, batch_size = None):
        import warnings
        with torch.inference_mode():
            ## Init model for retults analysis
            self.model.eval()
            self.performance[dataset] = {}
            self.prediction[dataset] = {}
            A = {}
            B = {}
            total_losses = {}

            # Create the losses
            losses = {}
            for name, values in self.model_def['Minimizers'].items():
                losses[name] = CustomLoss(values['loss'])

            recurrent = False
            if (closed_loop or connect or self.model_def['States']) and prediction_samples is not None:
                recurrent = True

            if data is None:
                check(dataset in self.data.keys(), ValueError, f'The dataset {dataset} is not loaded!')
                data = {key: torch.from_numpy(val).to(torch.float32) for key, val in self.data[dataset].items()}
            n_samples = len(data[list(data.keys())[0]])

            if recurrent:
                json_inputs = self.model_def['Inputs'] | self.model_def['States']
                batch_size = batch_size if batch_size is not None else n_samples - prediction_samples
                initial_value = 0

                model_inputs = list(self.model_def['Inputs'].keys())

                state_closed_loop = [key for key, value in self.model_def['States'].items() if 'closedLoop' in value.keys()] + list(closed_loop.keys())
                state_connect = [key for key, value in self.model_def['States'].items() if 'connect' in value.keys()] + list(connect.keys())

                non_mandatory_inputs = state_closed_loop + state_connect 
                mandatory_inputs = list(set(model_inputs) - set(non_mandatory_inputs))

                for key, value in self.model_def['Minimizers'].items():
                    total_losses[key], A[key], B[key] = [], [], []
                    for horizon_idx in range(prediction_samples + 1):
                        A[key].append([])
                        B[key].append([])

                X = {}
                ## Update with virtual states
                self.model.update(closed_loop=closed_loop, connect=connect)
                for idx in range(initial_value, (n_samples - batch_size - prediction_samples + 1), (batch_size + step - 1)):
                    ## Reset 
                    horizon_losses = {key: [] for key in self.model_def['Minimizers'].keys()}
                    for key in non_mandatory_inputs:
                        if key in data.keys(): # and len(data[key]) >= (idx + self.input_n_samples[key]): 
                        ## with data
                            X[key] = data[key][idx:idx+batch_size]
                        #elif key in self.states.keys(): ## with state
                            #X[key] = self.states[key]
                        else: ## with zeros
                            window_size = self.input_n_samples[key]
                            dim = self.model_def['Inputs'][key]['dim'] if key in model_inputs else self.model_def['States'][key]['dim']
                            X[key] = torch.zeros(size=(batch_size, window_size, dim), dtype=torch.float32, requires_grad=False)
                            self.states[key] = X[key]

                    for horizon_idx in range(prediction_samples + 1):
                        ## Get data 
                        for key in mandatory_inputs:
                            X[key] = data[key][idx+horizon_idx:idx+horizon_idx+batch_size]
                        ## Forward pass
                        out, minimize_out, out_closed_loop, out_connect = self.model(X)

                        ## Loss Calculation
                        for key, value in self.model_def['Minimizers'].items():
                            A[key][horizon_idx].append(minimize_out[value['A']])
                            B[key][horizon_idx].append(minimize_out[value['B']])
                            loss = losses[key](minimize_out[value['A']], minimize_out[value['B']])
                            loss = (loss * minimize_gain[key]) if key in minimize_gain.keys() else loss  ## Multiply by the gain if necessary
                            horizon_losses[key].append(loss)

                        ## Update
                        for key, val in out_closed_loop.items():
                            shift = val.shape[1]  ## take the output time dimension
                            X[key] = torch.roll(X[key], shifts=-1, dims=1) ## Roll the time window
                            X[key][:, -shift:, :] = val ## substitute with the predicted value
                            self.states[key] = X[key].clone()
                        for key, value in out_connect.items():
                            X[key] = value
                            self.states[key] = X[key].clone()

                    ## Calculate the total loss
                    for key in self.model_def['Minimizers'].keys():
                        loss = sum(horizon_losses[key]) / (prediction_samples + 1)
                        total_losses[key].append(loss.detach().numpy())

                for key, value in self.model_def['Minimizers'].items():
                    for horizon_idx in range(prediction_samples + 1):
                        A[key][horizon_idx] = np.concatenate(A[key][horizon_idx])
                        B[key][horizon_idx] = np.concatenate(B[key][horizon_idx])
                    total_losses[key] = np.mean(total_losses[key])

            else:
                if batch_size is None:
                    batch_size = n_samples

                for key, value in self.model_def['Minimizers'].items():
                    total_losses[key], A[key], B[key] = [], [], []

                for idx in range(0, (n_samples - batch_size + 1), batch_size):
                    ## Build the input tensor
                    XY = {key: val[idx:idx + batch_size] for key, val in data.items()}
                    #if (closed_loop or connect or self.model_def['States']):
                        ## Reset state variables with zeros or using inputs
                        #self.model.reset_states(XY, only=False)
                        #self.model.reset_connect_variables(connect, XY, only=False)

                    ## Model Forward
                    _, minimize_out, _, _ = self.model(XY)  ## Forward pass
                    ## Loss Calculation
                    for key, value in self.model_def['Minimizers'].items():
                        A[key].append(minimize_out[value['A']].numpy())
                        B[key].append(minimize_out[value['B']].numpy())
                        loss = losses[key](minimize_out[value['A']], minimize_out[value['B']])
                        loss = (loss * minimize_gain[key]) if key in minimize_gain.keys() else loss
                        total_losses[key].append(loss.detach().numpy())

                for key, value in self.model_def['Minimizers'].items():
                    A[key] = np.concatenate(A[key])
                    B[key] = np.concatenate(B[key])
                    total_losses[key] = np.mean(total_losses[key])

            for ind, (key, value) in enumerate(self.model_def['Minimizers'].items()):
                A_np = np.array(A[key])
                B_np = np.array(B[key])
                self.performance[dataset][key] = {}
                self.performance[dataset][key][value['loss']] = np.mean(total_losses[key]).item()
                self.performance[dataset][key]['fvu'] = {}
                # Compute FVU
                residual = A_np - B_np
                error_var = np.var(residual)
                error_mean = np.mean(residual)
                #error_var_manual = np.sum((residual-error_mean) ** 2) / (len(self.prediction['B'][ind]) - 0)
                #print(f"{key} var np:{new_error_var} and var manual:{error_var_manual}")
                with warnings.catch_warnings(record=True) as w:
                    self.performance[dataset][key]['fvu']['A'] = (error_var / np.var(A_np)).item()
                    self.performance[dataset][key]['fvu']['B'] = (error_var / np.var(B_np)).item()
                    if w and np.var(A_np) == 0.0 and  np.var(B_np) == 0.0:
                        self.performance[dataset][key]['fvu']['A'] = np.nan
                        self.performance[dataset][key]['fvu']['B'] = np.nan
                self.performance[dataset][key]['fvu']['total'] = np.mean([self.performance[dataset][key]['fvu']['A'],self.performance[dataset][key]['fvu']['B']]).item()
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
                self.performance[dataset][key]['aic'] = {'value':aic,'total_params':total_params,'log_likelihood':log_likelihood}
                # Prediction and target
                self.prediction[dataset][key] = {}
                self.prediction[dataset][key]['A'] = A_np.tolist()
                self.prediction[dataset][key]['B'] = B_np.tolist()

            ## Remove virtual states
            for key in (connect.keys() | closed_loop.keys()):
                if key in self.states.keys():
                    del self.states[key]

            self.performance[dataset]['total'] = {}
            self.performance[dataset]['total']['mean_error'] = np.mean([value for key,value in total_losses.items()])
            self.performance[dataset]['total']['fvu'] = np.mean([self.performance[dataset][key]['fvu']['total'] for key in self.model_def['Minimizers'].keys()])
            self.performance[dataset]['total']['aic'] = np.mean([self.performance[dataset][key]['aic']['value']for key in self.model_def['Minimizers'].keys()])

        self.visualizer.showResult(dataset)

    def getWorkspace(self):
        return self.exporter.getWorkspace()

    def saveTorchModel(self, name = 'net', model_folder = None, models = None):
        check(self.neuralized == True, RuntimeError, 'The model is not neuralized yet!')
        if models is not None:
            if name == 'net':
                name += '_' + '_'.join(models)
            model_def = ModelDef()
            model_def.update(model_dict = {key: self.model_dict[key] for key in models if key in self.model_dict})
            model_def.setBuildWindow(self.model_def['Info']['SampleTime'])
            model_def.updateParameters(self.model)
            model = Model(model_def.json)
        else:
            model = self.model
        self.exporter.saveTorchModel(model, name, model_folder)

    def loadTorchModel(self, name = 'net', model_folder = None):
        check(self.neuralized == True, RuntimeError, 'The model is not neuralized yet.')
        self.exporter.loadTorchModel(self.model, name, model_folder)

    def saveModel(self, name = 'net', model_path = None, models = None):
        if models is not None:
            if name == 'net':
                name += '_' + '_'.join(models)
            model_def = ModelDef()
            model_def.update(model_dict = {key: self.model_dict[key] for key in models if key in self.model_dict})
            model_def.setBuildWindow(self.model_def['Info']['SampleTime'])
            model_def.updateParameters(self.model)
        else:
            model_def = self.model_def
        check(model_def.isDefined(), RuntimeError, "The network has not been defined.")
        self.exporter.saveModel(model_def.json, name, model_path)

    def loadModel(self, name = None, model_folder = None):
        if name is None:
            name = 'net'
        model_def = self.exporter.loadModel(name, model_folder)
        check(model_def, RuntimeError, "Error to load the network.")
        self.model_def = ModelDef(model_def)
        self.model = None
        self.neuralized = False
        self.traced = False

    def exportPythonModel(self, name = 'net', model_path = None, models = None):
        if models is not None:
            if name == 'net':
                name += '_' + '_'.join(models)
            model_def = ModelDef()
            model_def.update(model_dict = {key: self.model_dict[key] for key in models if key in self.model_dict})
            model_def.setBuildWindow(self.model_def['Info']['SampleTime'])
            model_def.updateParameters(self.model)
            model = Model(model_def.json)
        else:
            model_def = self.model_def
            model = self.model
        check(model_def['States'] == {}, TypeError, "The network has state variables. The export to python is not possible.")
        check(model_def.isDefined(), RuntimeError, "The network has not been defined.")
        check(self.traced == False, RuntimeError,
                  'The model is traced and cannot be exported to Python.\n Run neuralizeModel() to recreate a standard model.')
        check(self.neuralized == True, RuntimeError, 'The model is not neuralized yet.')
        self.exporter.saveModel(model_def.json, name, model_path)
        self.exporter.exportPythonModel(model_def, model, name, model_path)

    def importPythonModel(self, name = None, model_folder = None):
        if name is None:
            name = 'net'
        model_def = self.exporter.loadModel(name, model_folder)
        check(model_def is not None, RuntimeError, "Error to load the network.")
        self.neuralizeModel(model_def=model_def)
        self.model = self.exporter.importPythonModel(name, model_folder)
        self.traced = True
        self.model_def.updateParameters(self.model)

    def exportONNX(self, inputs_order, outputs_order,  models = None, name = 'net', model_folder = None):
        check(self.model_def.isDefined(), RuntimeError, "The network has not been defined.")
        check(self.traced == False, RuntimeError, 'The model is traced and cannot be exported to ONNX.\n Run neuralizeModel() to recreate a standard model.')
        check(self.neuralized == True, RuntimeError, 'The model is not neuralized yet.')
        check(self.model_def.model_dict != {}, RuntimeError, 'The model is loaded and not created.')
        model_def = ModelDef()
        if models is not None:
            if name == 'net':
                name += '_' + '_'.join(models)
            model_def.update(model_dict = {key: self.model_def.model_dict[key] for key in models if key in self.model_def.model_dict})
        else:
            model_def.update(model_dict = self.model_def.model_dict)
        model_def.setBuildWindow(self.model_def['Info']['SampleTime'])
        model_def.updateParameters(self.model)
        model = Model(model_def.json)
        self.exporter.exportONNX(model_def, model, inputs_order, outputs_order, name, model_folder)

    def exportReport(self, name = 'net', model_folder = None):
        self.exporter.exportReport(self, name, model_folder)
import copy
import torch

from neu4mes.utilis import check

class Optimizer:
    def __init__(self, name, optimizer_defaults = {}, optimizer_params = []):
        self.name = name
        self.optimizer_defaults = copy.deepcopy(optimizer_defaults)
        self.optimizer_params = self.unfold(copy.deepcopy(optimizer_params))
        self.all_params = None
        self.params_to_train = None

    def set_params_to_train(self, all_params, params_to_train):
        self.all_params = all_params
        self.params_to_train = params_to_train
        if self.optimizer_params == []:
            for param_name in self.all_params.keys():
                if param_name in self.params_to_train:
                    self.optimizer_params.append({'params': param_name})
                else:
                    self.optimizer_params.append({'params': param_name, 'lr': 0.0})

    def set_defaults(self, optimizer_defaults):
        self.optimizer_defaults = optimizer_defaults

    def set_params(self, optimizer_params):
        self.optimizer_params = self.unfold(optimizer_params)

    def unfold(self, params):
        optimizer_params = []
        check(type(params) is list, KeyError, f'The params {params} must be a list')
        for param in params:
            if type(param['params']) is list:
                par_copy = copy.deepcopy(param)
                del par_copy['params']
                for par in param['params']:
                    optimizer_params.append({'params':par}|par_copy)
            else:
                optimizer_params.append(param)
        return optimizer_params

    def add_defaults(self, option_name, params, overwrite = True):
        if params is not None:
            if overwrite:
                self.optimizer_defaults[option_name] = params
            elif option_name not in self.optimizer_defaults:
                self.optimizer_defaults[option_name] = params

    def add_option_to_params(self, option_name, params, overwrite = True):
        if params is None:
            return
        for key, value in params.items():
            check(self.all_params is not None, RuntimeError, "Call set_params before add_option_to_params")
            old_key = False
            for param in self.optimizer_params:
                if param['params'] == key:
                    old_key = True
                    if overwrite:
                        param[option_name] = value
                    elif option_name not in param:
                        param[option_name] = value
            if old_key == False:
                self.optimizer_params.append({'params': key, option_name: value})

    def replace_key_with_params(self):
        params = copy.deepcopy(self.optimizer_params)
        for param in params:
            if type(param['params']) is list:
                for ind, par in enumerate(param['params']):
                    param['params'][ind] = self.all_params[par]
            else:
                param['params'] = self.all_params[param['params']]
        return params

    def get_torch_optimizer(self):
        raise NotImplemented('The function get_torch_optimizer must be implemented.')

class SGD(Optimizer):
    def __init__(self, optimizer_defaults = {}, optimizer_params = []):
        super(SGD, self).__init__('SGD', optimizer_defaults, optimizer_params)

    def get_torch_optimizer(self):
        return torch.optim.SGD(self.replace_key_with_params(), **self.optimizer_defaults)

class Adam(Optimizer):
    def __init__(self, optimizer_defaults = {}, optimizer_params = []):
        super(Adam, self).__init__('Adam', optimizer_defaults, optimizer_params)

    def get_torch_optimizer(self):
        return torch.optim.Adam(self.replace_key_with_params(), **self.optimizer_defaults)
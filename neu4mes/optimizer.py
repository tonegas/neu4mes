import copy
import torch

from neu4mes.utilis import check

class Optimizer():
    def __init__(self, name, optimizer_defaults, optimizer_params = []):
        self.name = name
        self.optimizer_defaults = optimizer_defaults
        self.optimizer_params = optimizer_params
        self.all_params = None
        self.params_to_train = None

    def set_params(self, all_params, params_to_train):
        self.all_params = all_params
        self.params_to_train = params_to_train
        for param_name in all_params.keys():
            if param_name in params_to_train:
                self.optimizer_params.append({'params': param_name })
            else:
                self.optimizer_params.append({'params': param_name, 'lr': 0.0})
    def add_option_to_params(self, option_name, params, overwrite = True):
        if params is None:
            return
        for key, value in params.items():
            check(self.all_params is not None, RuntimeError, "Call set_params before add_option_to_params")
            old_key = False
            for param in self.optimizer_params:
                if param['params'] == key:
                    if overwrite:
                        param[option_name] = value
                        old_key = True
            if old_key == False:
                self.optimizer_params.append({'params': key, option_name: value})

    def replace_key_with_params(self):
        params = copy.deepcopy(self.optimizer_params)
        for param in params:
            param['params'] = self.all_params[param['params']]
        return params

    def get_torch_optimizer(self):
        raise NotImplemented('The function get_torch_optimizer must be implemented.')

        #
        # freezed_model_parameters = freezed_params - set(self.lr_param.keys())
        # # print('freezed model parameters: ', freezed_model_parameters)
        # learned_model_parameters = params - freezed_model_parameters
        # # print('learned model parameters: ', learned_model_parameters)
        # model_parameters = []
        #
        # for param_name, param_value in self.model.all_parameters.items():
        #     if param_name in lr_param.keys():  ## if the parameter is specified it has top priority
        #         model_parameters.append({'params':param_value, 'lr':self.lr_param[param_name], 'weight_decay':self.l2_param[param_name]})
        #     elif param_name in freezed_model_parameters: ## if the parameter is not in the training model, it's freezed
        #         model_parameters.append({'params':param_value, 'lr':0.0})
        #     elif param_name in learned_model_parameters: ## if the parameter is in the training model, it's learned with the default learning rate
        #         model_parameters.append({'params':param_value, 'lr':self.lr})
        # bias_params = [p for name, p in self.named_parameters() if 'bias' in name]
        # others = [p for name, p in self.named_parameters() if 'bias' not in name]
        #
        # optim.SGD([
        #     {'params': others},
        #     {'params': bias_params, 'weight_decay': 0}
        # ], weight_decay=1e-2, lr=1e-2)

        #print('model parameters: ', model_parameters)
        #self.optimizer = optimizer(model_parameters, **optimizer_params)
        #self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate, weight_decay=self.weight_decay)

class SGD(Optimizer):
    def __init__(self, optimizer_defaults, optimizer_params = []):
        super(SGD, self).__init__('SGD', optimizer_defaults, optimizer_params)

    def get_torch_optimizer(self):
        return torch.optim.SGD(self.replace_key_with_params(), **self.optimizer_defaults)

class Adam(Optimizer):
    def __init__(self, optimizer_defaults, optimizer_params = []):
        super(Adam, self).__init__('Adam', optimizer_defaults, optimizer_params)

    def get_torch_optimizer(self):
        return torch.optim.Adam(self.replace_key_with_params(), **self.optimizer_defaults)
import numpy as np
from pprint import pformat

import subprocess
import json

from neu4mes.visualizer.visualizer import Visualizer, color, GREEN, RED

class TextVisualizer(Visualizer):
    def __init__(self, verbose=1):
        self.verbose = verbose

    def __title(self,msg, lenght = 80):
        print(color((msg).center(lenght, '='), GREEN, True))

    def __line(self):
        print(color('='.center(80, '='),GREEN))

    def __singleline(self):
        print(color('-'.center(80, '-'),GREEN))

    def __paramjson(self,name, value, dim =30):
        lines = pformat(value, width=80 - dim).strip().splitlines()
        vai = ('\n' + (' ' * dim)).join(x for x in lines)
        # pformat(value).strip().splitlines().rjust(40)
        print(color((name).ljust(dim) + vai,GREEN))

    def __param(self,name, value, dim =30):
        print(color((name).ljust(dim) + value,GREEN))

    def showModel(self, model):
        if self.verbose >= 1:
            self.__title(" Neu4mes Model ")
            print(color(pformat(model),GREEN))
            self.__line()

    def showaddMinimize(self,variable_name):
        if self.verbose >= 2:
            self.__title(f" Minimize Error of {variable_name} with {self.n4m.model_def['Minimizers'][variable_name]['loss']} ")
            self.__paramjson(f"Model {self.n4m.model_def['Minimizers'][variable_name]['A'].name}", self.n4m.model_def['Minimizers'][variable_name]['A'].json)
            self.__paramjson(f"Model {self.n4m.model_def['Minimizers'][variable_name]['B'].name}", self.n4m.model_def['Minimizers'][variable_name]['B'].json)
            self.__line()

    def showModelInputWindow(self):
        if self.verbose >= 2:
            self.__title(" Neu4mes Model Input Windows ")
            self.__paramjson("time_window_backward:",self.n4m.input_tw_backward)
            self.__paramjson("time_window_forward:",self.n4m.input_tw_forward)
            self.__paramjson("sample_window_backward:", self.n4m.input_ns_backward)
            self.__paramjson("sample_window_forward:", self.n4m.input_ns_forward)
            self.__paramjson("input_n_samples:", self.n4m.input_n_samples)
            self.__param("max_samples [backw, forw]:", f"[{self.n4m.max_samples_backward},{self.n4m.max_samples_forward}]")
            self.__param("max_samples total:",f"{self.n4m.max_n_samples}")
            self.__line()

    def showModelRelationSamples(self):
        if self.verbose >= 2:
            self.__title(" Neu4mes Model Relation Samples ")
            self.__paramjson("Relation_samples:", self.n4m.relation_samples)
            self.__line()

    def showBuiltModel(self):
        if self.verbose >= 2:
            self.__title(" Neu4mes Built Model ")
            print(color(pformat(self.n4m.model),GREEN))
            self.__line()

    def showWeights(self, batch = None, epoch = None):
        if self.verbose >= 2:
            par = self.n4m.run_training_params
            dim = len(self.n4m.model_def['Minimizers'])
            COLOR = RED
            if epoch is not None:
                print(color('|' + (f"{epoch + 1}/{par['num_of_epochs']}").center(10, ' ') + '|',COLOR), end='')
                print(color((f' Params end epochs {epoch + 1} ').center(20 * (dim + 1) - 1, '-') + '|',COLOR))

            if batch is not None:
                print(color('|' + (f"{batch + 1}").center(10, ' ') + '|', COLOR), end='')
                print(color((f' Params end batch {batch + 1} ').center(20 * (dim + 1) - 1, '-') + '|', COLOR))

            for key,param in self.n4m.model.all_parameters.items():
                print(color('|' + (f"{key}").center(10, ' ') + '|', COLOR), end='')
                print(color((f'{param.tolist()}').center(20 * (dim + 1) - 1, ' ') + '|', COLOR))

            if epoch is not None:
                print(color('|'+(f'').center(10+20*(dim+1), '-') + '|'))

    def showDataset(self, name):
        if self.verbose >= 1:
            self.__title(" Neu4mes Model Dataset ")
            self.__param("Dataset Name:", name)
            self.__param("Number of files:", f'{self.n4m.file_count}')
            self.__param("Total number of samples:", f'{self.n4m.num_of_samples[name]}')
            for key in self.n4m.model_def['Inputs'].keys():
                if key in self.n4m.data[name].keys():
                    self.__param(f"Shape of {key}:", f'{self.n4m.data[name][key].shape}')
            self.__line()

    def showStartTraining(self):
        if self.verbose >= 1:
            par = self.n4m.run_training_params
            dim = len(self.n4m.model_def['Minimizers'])
            self.__title(" Neu4mes Training ", 12+(len(self.n4m.model_def['Minimizers'])+1)*20)
            print(color('|'+(f'Epoch').center(10,' ')+'|'),end='')
            for key in self.n4m.model_def['Minimizers'].keys():
                print(color((f'{key}').center(19, ' ') + '|'), end='')
            print(color((f'Total').center(19, ' ') + '|'))

            print(color('|' + (f' ').center(10, ' ') + '|'), end='')
            for key in self.n4m.model_def['Minimizers'].keys():
                print(color((f'Loss').center(19, ' ') + '|'),end='')
            print(color((f'Loss').center(19, ' ') + '|'))

            print(color('|' + (f' ').center(10, ' ') + '|'), end='')
            for key in self.n4m.model_def['Minimizers'].keys():
                if par['n_samples_val']:
                    print(color((f'train').center(9, ' ') + '|'),end='')
                    print(color((f'val').center(9, ' ') + '|'),end='')
                else:
                    print(color((f'train').center(19, ' ') + '|'), end='')
            if par['n_samples_val']:
                print(color((f'train').center(9, ' ') + '|'), end='')
                print(color((f'val').center(9, ' ') + '|'))
            else:
                print(color((f'train').center(19, ' ') + '|'))

            print(color('|'+(f'').center(10+20*(dim+1), '-') + '|'))

    def showTraining(self, epoch, train_losses, val_losses):
        if self.verbose >= 1:
            eng = lambda val: np.format_float_scientific(val, precision=3)
            par = self.n4m.run_training_params
            show_epoch = 1 if par['num_of_epochs'] <= 20 else 10
            dim = len(self.n4m.model_def['Minimizers'])
            if epoch < par['num_of_epochs']:
                print('', end='\r')
                print('|' + (f"{epoch + 1}/{par['num_of_epochs']}").center(10, ' ') + '|', end='')
                train_loss = []
                val_loss = []
                for key in self.n4m.model_def['Minimizers'].keys():
                    train_loss.append(train_losses[key][epoch])
                    if val_losses:
                        val_loss.append(val_losses[key][epoch])
                        print((f'{eng(train_losses[key][epoch])}').center(9, ' ') + '|', end='')
                        print((f'{eng(val_losses[key][epoch])}').center(9, ' ') + '|', end='')
                    else:
                        print((f'{eng(train_losses[key][epoch])}').center(19, ' ') + '|', end='')

                if val_losses:
                    print((f'{eng(np.mean(train_loss))}').center(9, ' ') + '|', end='')
                    print((f'{eng(np.mean(val_loss))}').center(9, ' ') + '|', end='')
                else:
                    print((f'{eng(np.mean(train_loss))}').center(19, ' ') + '|', end='')

                if (epoch + 1) % show_epoch == 0:
                    print('', end='\r')
                    print(color('|' + (f"{epoch + 1}/{par['num_of_epochs']}").center(10, ' ') + '|'), end='')
                    for key in self.n4m.model_def['Minimizers'].keys():
                        if val_losses:
                            print(color((f'{eng(train_losses[key][epoch])}').center(9, ' ') + '|'), end='')
                            print(color((f'{eng(val_losses[key][epoch])}').center(9, ' ') + '|'), end='')
                        else:
                            print(color((f'{eng(train_losses[key][epoch])}').center(19, ' ') + '|'), end='')

                    if val_losses:
                        print(color((f'{eng(np.mean(train_loss))}').center(9, ' ') + '|'), end='')
                        print(color((f'{eng(np.mean(val_loss))}').center(9, ' ') + '|'))
                    else:
                        print(color((f'{eng(np.mean(train_loss))}').center(19, ' ') + '|'))

            if epoch+1 == par['num_of_epochs']:
                print(color('|'+(f'').center(10+20*(dim+1), '-') + '|'))

    def showTrainingTime(self, time):
        if self.verbose >= 1:
            self.__title(" Neu4mes Training Time ")
            self.__param("Total time of Training:", f'{time}')
            self.__line()

    def showTrainParams(self):
        if self.verbose >= 1:
            self.__title(" Neu4mes Model Train Parameters ")
            par = self.n4m.run_training_params
            self.__paramjson("models:", par['models'])
            self.__param("train dataset:", f"{par['train_dataset']}")
            self.__param("train {batch size, samples}:", f"{{{par['train_batch_size']}, {par['n_samples_train']}}}")
            if par['n_samples_val']:
                self.__param("val dataset:", f"{par['validation_dataset']}")
                self.__param("val {batch size, samples}:", f"{{{par['val_batch_size']}, {par['n_samples_val']}}}")
            if par['n_samples_test']:
                self.__param("test dataset:", f"{par['test_dataset']}")
                self.__param("test {batch size, samples}:", f"{{{par['test_batch_size']}, {par['n_samples_test']}}}")

            self.__paramjson("num of epochs:", par['num_of_epochs'])
            if par['shuffle_data']:
                self.__param('shuffle data:', str(par['shuffle_data']))
            if 'early_stopping' in par:
                self.__param('early stopping:', par['early_stopping'])
                self.__paramjson('early stopping params:', par['early_stopping_params'])

            self.__paramjson('minimizers:', par['minimizers'])

            if par['recurrent_train']:
                self.__param("prediction samples:", str(par['prediction_samples']))
                self.__param("step:", str(par['step']))
                self.__paramjson("closed loop:", par['closed_loop'])
                self.__paramjson("connect:", par['connect'])

            self.__param("optimizer:", par['optimizer'])
            self.__paramjson("optimizer defaults:",self.n4m.run_training_params['optimizer_defaults'])
            if self.n4m.run_training_params['optimizer_params'] is not None:
                self.__paramjson("optimizer params:", self.n4m.run_training_params['optimizer_params'])

            self.__line()

    def showResult(self, name_data):
        eng = lambda val: np.format_float_scientific(val, precision=3)
        if self.verbose >= 1:
            loss_type_list = set([value["loss"] for ind, (key, value) in enumerate(self.n4m.model_def['Minimizers'].items())])
            self.__title(f" Neu4mes Model Results for {name_data} ", 12 + (len(loss_type_list) + 2) * 20)
            print(color('|' + (f'Loss').center(10, ' ') + '|'), end='')
            for loss in loss_type_list:
                print(color((f'{loss}').center(19, ' ') + '|'), end='')
            print(color((f'FVU').center(19, ' ') + '|'), end='')
            print(color((f'AIC').center(19, ' ') + '|'))

            print(color('|' + (f'').center(10, ' ') + '|'), end='')
            for i in range(len(loss_type_list)):
                print(color((f'small better').center(19, ' ') + '|'), end='')
            print(color((f'small better').center(19, ' ') + '|'), end='')
            print(color((f'lower better').center(19, ' ') + '|'))

            print(color('|' + (f'').center(10 + 20 * (len(loss_type_list) + 2), '-') + '|'))
            for ind, (key, value) in enumerate(self.n4m.model_def['Minimizers'].items()):
                print(color('|'+(f'{key}').center(10, ' ') + '|'), end='')
                for loss in list(loss_type_list):
                    if value["loss"] == loss:
                        print(color((f'{eng(self.n4m.performance[name_data][key][value["loss"]])}').center(19, ' ') + '|'), end='')
                    else:
                        print(color((f' ').center(19, ' ') + '|'), end='')
                print(color((f'{eng(self.n4m.performance[name_data][key]["fvu"]["total"])}').center(19, ' ') + '|'), end='')
                print(color((f'{eng(self.n4m.performance[name_data][key]["aic"]["value"])}').center(19, ' ') + '|'))

            print(color('|' + (f'').center(10 + 20 * (len(loss_type_list) + 2), '-') + '|'))
            print(color('|'+(f'Total').center(10, ' ') + '|'), end='')
            print(color((f'{eng(self.n4m.performance[name_data]["total"]["mean_error"])}').center(len(loss_type_list)*20-1, ' ') + '|'), end='')
            print(color((f'{eng(self.n4m.performance[name_data]["total"]["fvu"])}').center(19, ' ') + '|'), end='')
            print(color((f'{eng(self.n4m.performance[name_data]["total"]["aic"])}').center(19, ' ') + '|'))

            print(color('|' + (f'').center(10 + 20 * (len(loss_type_list) + 2), '-') + '|'))

        if self.verbose >= 2:
            self.__title(" Detalied Results ")
            print(color(pformat(self.n4m.performance), GREEN))
            self.__line()

    def saveModel(self, name, path):
        if self.verbose >= 1:
            self.__title(f" Save {name} ")
            self.__param("Model saved in:", path)
            self.__line()

    def loadModel(self, name, path):
        if self.verbose >= 1:
            self.__title(f" Load {name} ")
            self.__param("Model loaded from:", path)
            self.__line()

    def exportModel(self, name, path):
        if self.verbose >= 1:
            self.__title(f" Export {name} ")
            self.__param("Model exported in:", path)
            self.__line()

    def importModel(self, name, path):
        if self.verbose >= 1:
            self.__title(f" Import {name} ")
            self.__param("Model imported from:", path)
            self.__line()
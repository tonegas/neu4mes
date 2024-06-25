import numpy as np
from pprint import pformat

import subprocess
import json

from neu4mes.visualizer.visualizer import Visualizer, color, GREEN

class TextVisualizer(Visualizer):
    def __init__(self, verbose=1):
        self.verbose = verbose

    def __title(self,msg, lenght = 80):
        print(color((msg).center(lenght, '='), GREEN, True))

    def __line(self):
        print(color('='.center(80, '='),GREEN))

    def __paramjson(self,name, value, dim =30):
        lines = pformat(value, width=80 - dim).strip().splitlines()
        vai = ('\n' + (' ' * dim)).join(x for x in lines)
        # pformat(value).strip().splitlines().rjust(40)
        print(color((name).ljust(dim) + vai,GREEN))

    def __param(self,name, value, dim =30):
        print(color((name).ljust(dim) + value,GREEN))

    def showModel(self):
        if self.verbose >= 1:
            self.__title(" Neu4mes Model ")
            print(color(pformat(self.n4m.model_def),GREEN))
            self.__line()

    def showMinimizeError(self,variable_name):
        if self.verbose >= 2:
            self.__title(f" Minimize Error of {variable_name} with {self.n4m.minimize_dict[variable_name]['loss']} ")
            self.__paramjson(f"Model {self.n4m.minimize_dict[variable_name]['A'][0]}", self.n4m.minimize_dict[variable_name]['A'][1].json)
            self.__paramjson(f"Model {self.n4m.minimize_dict[variable_name]['B'][0]}", self.n4m.minimize_dict[variable_name]['B'][1].json)
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
        if self.verbose >= 1:
            self.__title(" Neu4mes Built Model ")
            print(color(pformat(self.n4m.model),GREEN))
            self.__line()

    def showDataset(self):
        if self.verbose >= 1:
            self.__title(" Neu4mes Model Dataset ")
            self.__param("Number of files:", f'{self.n4m.file_count}')
            self.__param("Total number of samples:", f'{self.n4m.num_of_samples}')
            self.__line()

    def showTraining(self, epoch, train_losses, test_losses):
        show_epoch = 1 if self.n4m.num_of_epochs <= 20 else 10
        dim = len(self.n4m.minimize_dict)
        if self.verbose >= 1:
            if epoch == 0:
                self.__title(" Neu4mes Training ",12+(len(self.n4m.minimize_dict)+1)*20)
                print(color('|'+(f'Epoch').center(10,' ')+'|'),end='')
                for key in self.n4m.minimize_dict.keys():
                    print(color((f'{key}').center(19, ' ') + '|'), end='')
                print(color((f'Total').center(19, ' ') + '|'))

                print(color('|' + (f' ').center(10, ' ') + '|'), end='')
                for key in self.n4m.minimize_dict.keys():
                    print(color((f'Loss').center(19, ' ') + '|'),end='')
                print(color((f'Loss').center(19, ' ') + '|'))

                print(color('|' + (f' ').center(10, ' ') + '|'), end='')
                for key in self.n4m.minimize_dict.keys():
                    print(color((f'train').center(9, ' ') + '|'),end='')
                    print(color((f'test').center(9, ' ') + '|'), end='')
                print(color((f'train').center(9, ' ') + '|'), end='')
                print(color((f'test').center(9, ' ') + '|'))

                print(color('|'+(f'').center(10+20*(dim+1), '-') + '|'))
            if epoch < self.n4m.num_of_epochs:
                print('', end='\r')
                print('|' + (f'{epoch + 1}/{self.n4m.num_of_epochs}').center(10, ' ') + '|', end='')
                train_loss = []
                test_loss = []
                for key in self.n4m.minimize_dict.keys():
                    train_loss.append(train_losses[key][epoch])
                    test_loss.append(test_losses[key][epoch])
                    print((f'{train_losses[key][epoch]:.4f}').center(9, ' ') + '|', end='')
                    print((f'{test_losses[key][epoch]:.4f}').center(9, ' ') + '|', end='')
                print((f'{np.mean(train_loss):.4f}').center(9, ' ') + '|', end='')
                print((f'{np.mean(test_loss):.4f}').center(9, ' ') + '|', end='')

                if (epoch + 1) % show_epoch == 0:
                    print('', end='\r')
                    print(color('|' + (f'{epoch + 1}/{self.n4m.num_of_epochs}').center(10, ' ') + '|'), end='')
                    for key in self.n4m.minimize_dict.keys():
                        print(color((f'{train_losses[key][epoch]:.4f}').center(9, ' ') + '|'), end='')
                        print(color((f'{test_losses[key][epoch]:.4f}').center(9, ' ') + '|'), end='')
                    print(color((f'{np.mean(train_loss):.4f}').center(9, ' ') + '|'), end='')
                    print(color((f'{np.mean(test_loss):.4f}').center(9, ' ') + '|'))

            if epoch+1 == self.n4m.num_of_epochs:
                print(color('|'+(f'').center(10+20*(dim+1), '-') + '|'))

    def showTrainingTime(self, time):
        if self.verbose >= 1:
            self.__title(" Neu4mes Training Time ")
            self.__param("Total time of Training:", f'{time}')
            self.__line()

    def showResults(self):
        if self.verbose >= 1:
            loss_type_list = set([value["loss"] for ind, (key, value) in enumerate(self.n4m.minimize_dict.items())])
            self.__title(" Neu4mes Model Results ", 12 + (len(loss_type_list) + 2) * 20)
            print(color('|' + (f'Loss').center(10, ' ') + '|'), end='')
            for loss in loss_type_list:
                print(color((f'{loss} (test)').center(19, ' ') + '|'), end='')
            print(color((f'FVU (test)').center(19, ' ') + '|'), end='')
            print(color((f'AIC (test)').center(19, ' ') + '|'))

            print(color('|' + (f'').center(10, ' ') + '|'), end='')
            for i in range(len(loss_type_list)):
                print(color((f'small better').center(19, ' ') + '|'), end='')
            print(color((f'small better').center(19, ' ') + '|'), end='')
            print(color((f'lower better').center(19, ' ') + '|'))

            print(color('|' + (f'').center(10 + 20 * (len(loss_type_list) + 2), '-') + '|'))
            for ind, (key, value) in enumerate(self.n4m.minimize_dict.items()):
                print(color('|'+(f'{key}').center(10, ' ') + '|'), end='')
                for loss in list(loss_type_list):
                    if value["loss"] == loss:
                        print(color((f'{self.n4m.performance[key][value["loss"]]["test"]:.4f}').center(19, ' ') + '|'), end='')
                    else:
                        print(color((f' ').center(19, ' ') + '|'), end='')
                print(color((f'{self.n4m.performance[key]["fvu"]["total"]:.4f}').center(19, ' ') + '|'), end='')
                print(color((f'{self.n4m.performance[key]["aic"]["value"]:.4f}').center(19, ' ') + '|'))

            print(color('|' + (f'').center(10 + 20 * (len(loss_type_list) + 2), '-') + '|'))
            print(color('|'+(f'Total').center(10, ' ') + '|'), end='')
            print(color((f'{self.n4m.performance["total"]["mean_error"]["test"]:.4f}').center(len(loss_type_list)*20-1, ' ') + '|'), end='')
            print(color((f'{self.n4m.performance["total"]["fvu"]:.4f}').center(19, ' ') + '|'), end='')
            print(color((f'{self.n4m.performance["total"]["aic"]:.4f}').center(19, ' ') + '|'))

            print(color('|' + (f'').center(10 + 20 * (len(loss_type_list) + 2), '-') + '|'))

        if self.verbose >= 2:
            self.__title(" Detalied Results ")
            print(color(pformat(self.n4m.performance), GREEN))
            self.__line()
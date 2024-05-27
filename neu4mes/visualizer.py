import numpy as np
from pprint import pformat

from neu4mes import LOG_LEVEL
from neu4mes.logger import logging
log = logging.getLogger(__name__)
log.setLevel(max(logging.CRITICAL, LOG_LEVEL))

RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[%dm"
COLOR_BOLD_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
def color(msg, color_val = GREEN, bold = False):
    if bold:
        return COLOR_BOLD_SEQ % (30 + color_val) + msg + RESET_SEQ
    return COLOR_SEQ % (30 + color_val) + msg + RESET_SEQ


class Visualizer():
    def __init__(self, neu4mes):
        self.n4m = neu4mes

    def warning(self, msg):
        print(color(msg, YELLOW))

    def showModel(self):
        pass

    def showModelInputWindow(self):
        pass

    def showModelRelationSamples(self):
        pass

    def showBuiltModel(self):
        pass

    def showDataset(self):
        pass

    def showTraining(self):
        pass

    def showResults(self, neu4mes, output_keys, performance = None):
        pass

class TextVisualizer(Visualizer):
    def __init__(self, neu4mes, verbose=1):
        super().__init__(neu4mes)
        self.verbose = verbose

    def __title(self,msg):
        print(color((msg).center(80, '='), GREEN, True))

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
            self.__param("Total numeber of samples:", f'{self.n4m.num_of_samples}')
            self.__line()

    def showTraining(self, iter, train_losses, test_losses):
        dim = len(self.n4m.minimize_dict)
        if self.verbose >= 1:
            if iter == 0:
                self.__title(" Training ")
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
            if iter < self.n4m.num_of_epochs:
                print('', end='\r')
                print('|' + (f'{iter+1}/{self.n4m.num_of_epochs}').center(10, ' ') + '|', end='')
                train_loss = []
                test_loss = []
                for key in self.n4m.minimize_dict.keys():
                    train_loss.append(train_losses[key][iter])
                    test_loss.append(test_losses[key][iter])
                    print((f'{train_losses[key][iter]:.4f}').center(9, ' ') + '|',end='')
                    print((f'{test_losses[key][iter]:.4f}').center(9, ' ') + '|', end='')
                print((f'{np.mean(train_loss):.4f}').center(9, ' ') + '|', end='')
                print((f'{np.mean(test_loss):.4f}').center(9, ' ') + '|', end='')

                if (iter+1) % 10 == 0:
                    print('', end='\r')
                    print(color('|' + (f'{iter+1}/{self.n4m.num_of_epochs}').center(10, ' ') + '|'), end='')
                    for key in self.n4m.minimize_dict.keys():
                        print(color((f'{train_losses[key][iter]:.4f}').center(9, ' ') + '|'), end='')
                        print(color((f'{test_losses[key][iter]:.4f}').center(9, ' ') + '|'), end='')
                    print(color((f'{np.mean(train_loss):.4f}').center(9, ' ') + '|'), end='')
                    print(color((f'{np.mean(test_loss):.4f}').center(9, ' ') + '|'))

            if iter+1 == self.n4m.num_of_epochs:
                print(color('|'+(f'').center(10+20*(dim+1), '-') + '|'))

    def showResults(self, neu4mes, output_keys, performance = None):
        for i in range(0, neu4mes.prediction.shape[0]):
            text = "Rmse training: {:3.6f}\nRmse test: {:3.6f}\nAIC: {:3.6f}\nFVU: {:3.6f}".format(neu4mes.performance['rmse_train'][i], neu4mes.performance['rmse_test'][i], neu4mes.performance['aic'][i], neu4mes.performance['fvu'][i])
            print(text)

class StandardVisualizer(Visualizer):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt

    def showResults(self, neu4mes, output_keys, performance = None):
        # Plot
        self.fig, self.ax = self.plt.subplots(2*len(output_keys), 2,
                                        gridspec_kw={'width_ratios': [5, 1], 'height_ratios': [2, 1]*len(output_keys)})
        if len(self.ax.shape) == 1:
            self.ax = np.expand_dims(self.ax, axis=0)
        #plotsamples = self.prediction.shape[1]s
        plotsamples = 200
        for i in range(0, neu4mes.prediction.shape[0]):
            # Zoomed test data
            self.ax[2*i,0].plot(neu4mes.prediction[i], linestyle='dashed')
            self.ax[2*i,0].plot(neu4mes.label[i])
            self.ax[2*i,0].grid('on')
            self.ax[2*i,0].set_xlim((performance['max_se_idxs'][i]-plotsamples, performance['max_se_idxs'][i]+plotsamples))
            self.ax[2*i,0].vlines(performance['max_se_idxs'][i], neu4mes.prediction[i][performance['max_se_idxs'][i]], neu4mes.label[i][performance['max_se_idxs'][i]],
                                    colors='r', linestyles='dashed')
            self.ax[2*i,0].legend(['predicted', 'test'], prop={'family':'serif'})
            self.ax[2*i,0].set_title(output_keys[i], family='serif')
            # Statitics
            self.ax[2*i,1].axis("off")
            self.ax[2*i,1].invert_yaxis()
            if performance:
                text = "Rmse test: {:3.6f}\nFVU: {:3.6f}".format(#\nAIC: {:3.6f}
                    neu4mes.performance['rmse_test'][i], 
                    #neu4mes.performance['aic'][i], 
                    neu4mes.performance['fvu'][i])
                self.ax[2*i,1].text(0, 0, text, family='serif', verticalalignment='top')
            # test data
            self.ax[2*i+1,0].plot(neu4mes.prediction[i], linestyle='dashed')
            self.ax[2*i+1,0].plot(neu4mes.label[i])
            self.ax[2*i+1,0].grid('on')
            self.ax[2*i+1,0].legend(['predicted', 'test'], prop={'family':'serif'})
            self.ax[2*i+1,0].set_title(output_keys[i], family='serif')
            # Empty
            self.ax[2*i+1,1].axis("off")
        self.fig.tight_layout()
        self.plt.show()
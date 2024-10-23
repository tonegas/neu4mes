import subprocess, json, torch, inspect

import matplotlib.pyplot as plt
import numpy as np

from neu4mes.visualizer.textvisualizer import TextVisualizer
from neu4mes.fuzzify import triangular, rectangular, custom_function
from neu4mes.utils import check
from mplplots import plots

class MPLStaticVisualizer(TextVisualizer):
    def __init__(self, verbose = 1):
        super().__init__(verbose)

    def showEndTraining(self, epoch, train_losses, val_losses):
        for key in self.n4m.model_def['Minimizers'].keys():
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plots.plot_training(ax, "Training", key, epoch, train_losses[key], val_losses[key])
        plt.show()

    def showResult(self, name_data):
        super().showResult(name_data)
        if self.dynamic is False:
            for key in self.n4m.model_def['Minimizers'].keys():
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plots.plot_results(ax, name_data, key, self.n4m.prediction[name_data][key]['A'], self.n4m.prediction[name_data][key]['B'], self.n4m.model_def["SampleTime"])
            plt.show()
        else:
            for key in self.n4m.model_def['Minimizers'].keys():
                # Start the data visualizer process
                self.process_results[key] = subprocess.Popen(['python', self.time_series_visualizer_script], stdin=subprocess.PIPE,
                                                    text=True)
                data = {"name_data": name_data,
                        "key": key,
                        "performance": self.n4m.performance[name_data][key],
                        "prediction_A": self.n4m.prediction[name_data][key]['A'],
                        "prediction_B": self.n4m.prediction[name_data][key]['B'],
                        "sample_time": self.n4m.model_def["SampleTime"]}
                try:
                    # Send data to the visualizer process
                    self.process_results[key].stdin.write(f"{json.dumps(data)}\n")
                    self.process_results[key].stdin.flush()
                except BrokenPipeError:
                    if self.closed_process is False:
                        self.closed_process = True
                        self.warning("The visualizer process has been closed.")

    def showResults(self):
        pass
        # if self.dynamic is False:
        # self.resultAnalysis(train_dataset, XY_train, connect, closed_loop)
        # self.visualizer.showResult(train_dataset)
        # if self.run_training_params['n_samples_val'] > 0:
        #     self.resultAnalysis(validation_dataset, XY_val, connect, closed_loop)
        #     self.visualizer.showResult(validation_dataset)
        # if self.run_training_params['n_samples_test'] > 0:
        #     self.resultAnalysis(test_dataset, XY_test, connect, closed_loop)
        #     self.visualizer.showResult(test_dataset)
        #
        # super().showResult(name_data)
        # if self.dynamic is False:
        #     for key in self.n4m.model_def['Minimizers'].keys():
        #         fig = plt.figure()
        #         ax = fig.add_subplot(111)
        #         plots.plot_results(ax, name_data, key, self.n4m.prediction[name_data][key]['A'], self.n4m.prediction[name_data][key]['B'], self.n4m.model_def["SampleTime"])
        #     plt.show()
        # else:
        #     for key in self.n4m.model_def['Minimizers'].keys():
        #         # Start the data visualizer process
        #         self.process_results[key] = subprocess.Popen(['python', self.time_series_visualizer_script], stdin=subprocess.PIPE,
        #                                             text=True)
        #         data = {"name_data": name_data,
        #                 "key": key,
        #                 "performance": self.n4m.performance[name_data][key],
        #                 "prediction_A": self.n4m.prediction[name_data][key]['A'],
        #                 "prediction_B": self.n4m.prediction[name_data][key]['B'],
        #                 "sample_time": self.n4m.model_def["SampleTime"]}
        #         try:
        #             # Send data to the visualizer process
        #             self.process_results[key].stdin.write(f"{json.dumps(data)}\n")
        #             self.process_results[key].stdin.flush()
        #         except BrokenPipeError:
        #             if self.closed_process is False:
        #                 self.closed_process = True
        #                 self.warning("The visualizer process has been closed.")

    def showWeights(self, weights = None):
        pass

    def showFunctions(self, functions = None, xlim = None, num_points = 1000):
        check(self.n4m.neuralized, ValueError, "The model has not been neuralized.")
        for key, value in self.n4m.model_def['Functions'].items():
            if key in functions:
                if 'functions' in self.n4m.model_def['Functions'][key]:
                    if xlim is not None:
                        x_test = torch.from_numpy(np.linspace(xlim[0], xlim[1], num=num_points))
                    else:
                        x_test = torch.from_numpy(np.linspace(value['centers'][0] - 2, value['centers'][-1] + 2, num=num_points))
                    chan_centers = np.array(value['centers'])
                    activ_fun = {}
                    if isinstance(value['names'], list):
                        n_func = len(value['names'])
                    else:
                        n_func = 1
                    for i in range(len(chan_centers)):
                        if value['functions'] == 'Triangular':
                            activ_fun[i] = triangular(x_test, i, chan_centers).tolist()
                        elif value['functions'] == 'Rectangular':
                            activ_fun[i] = rectangular(x_test, i, chan_centers).tolist()
                        else:
                            if isinstance(value['names'], list):
                                if i >= n_func:
                                    func_idx = i - round(n_func * (i // n_func))
                                else:
                                    func_idx = i
                                exec(value['functions'][func_idx], globals())
                                function_to_call = globals()[value['names'][func_idx]]
                            else:
                                exec(value['functions'], globals())
                                function_to_call = globals()[value['names']]
                            activ_fun[i] = custom_function(function_to_call, x_test, i, chan_centers).tolist()
                    data = {"name": key,
                            "x": x_test.tolist(),
                            "y": activ_fun,
                            "chan_centers": chan_centers.tolist()}
                    # Start the data visualizer process
                    self.process_function[key] = subprocess.Popen(['python', self.fuzzy_visualizer_script],
                                                                  stdin=subprocess.PIPE,
                                                                  text=True)
                elif 'code':
                    check(value['n_input'] == 1 or value['n_input'] == 2, ValueError, "The function must have only one or two inputs.")
                    fun_inputs = tuple()
                    data = {"name": key}
                    for i in range(value['n_input']):
                        dim = value['in_dim'][i]
                        check(dim['dim'] == 1, ValueError, "The input dimension must be 1.")
                        if 'tw' in dim:
                            check(dim['tw'] == self.n4m.model_def['SampleTime'], ValueError, "The input window must be 1.")
                        elif 'sw' in dim:
                            check(dim['sw'] == 1, ValueError, "The input window must be 1.")

                        if xlim is not None:
                            if value['n_input'] == 2:
                                check(np.array(xlim).shape == (value['n_input'], 2), ValueError,
                                      "The xlim must have the same shape as the number of inputs.")
                                x_value = np.linspace(xlim[i][0], xlim[i][1], num=num_points)
                            else:
                                check(np.array(xlim).shape == (2,), ValueError,
                                      "The xlim must have the same shape as the number of inputs.")
                                x_value = np.linspace(xlim[0], xlim[1], num=num_points)
                        else:
                            x_value = np.linspace(0, 1, num=num_points)
                        data['x' + str(i)] = x_value.tolist()

                        if value['n_input'] == 2:
                            x_value = torch.from_numpy(x_value).repeat(num_points).unsqueeze(1).unsqueeze(1)
                        else:
                            x_value = torch.from_numpy(x_value).unsqueeze(1).unsqueeze(1)
                        fun_inputs += (x_value,)

                    data['params'] = []
                    for key, val in (self.n4m.model.all_parameters | self.n4m.model.all_constants).items():
                        if key in value['params_and_consts']:
                            data['params'] += [val.clone().tolist()]
                            fun_inputs += tuple(val.clone())
                    exec(value['code'], globals())
                    function_to_call = globals()[value['name']]
                    funinfo = inspect.getfullargspec(function_to_call)
                    data['input_names'] = funinfo.args
                    output = function_to_call(*fun_inputs)
                    check(output.shape[1] == 1, ValueError, "The output dimension must be 1.")
                    check(output.shape[2] == 1, ValueError, "The output window must be 1.")
                    if value['n_input'] == 2:
                        data['output'] = output.reshape(num_points,num_points).tolist()
                    else:
                        data['output'] = output.reshape(num_points).tolist()

                    # Start the data visualizer process
                    self.process_function[key] = subprocess.Popen(['python', self.function_visualizer_script],
                                                                  stdin=subprocess.PIPE,
                                                                  text=True)
                try:
                    # Send data to the visualizer process
                    self.process_function[key].stdin.write(f"{json.dumps(data)}\n")
                    self.process_function[key].stdin.flush()
                except BrokenPipeError:
                    if self.closed_process is False:
                        self.closed_process = True
                        self.warning("The visualizer process has been closed.")







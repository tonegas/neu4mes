import subprocess, json, torch, inspect

import matplotlib.pyplot as plt
import numpy as np

from neu4mes.visualizer.textvisualizer import TextVisualizer
from neu4mes.fuzzify import triangular, rectangular, custom_function
from neu4mes.utils import check

class MPLVisulizer(TextVisualizer):
    def __init__(self, verbose=1):
        import signal
        import sys
        super().__init__(verbose)
        # Path to the data visualizer script
        self.training_visualizer_script = 'neu4mes/visualizer/trainingplot.py'
        self.time_series_visualizer_script = 'neu4mes/visualizer/resultsplot.py'
        self.fuzzy_visualizer_script = 'neu4mes/visualizer/fuzzyplot.py'
        self.function_visualizer_script = 'neu4mes/visualizer/functionplot.py'
        self.process_training = {}
        self.process_results = {}
        self.process_function = {}

        def signal_handler(sig, frame):
            for key in self.n4m.model_def['Minimizers'].keys():
                self.process_training[key].terminate()
                self.process_results[key].terminate()
            sys.exit()

        signal.signal(signal.SIGINT, signal_handler)

    def showStartTraining(self):
        pass

    def showTraining(self, epoch, train_losses, val_losses):
        if epoch == 0:
            for key in self.n4m.model_def['Minimizers'].keys():
                # Start the data visualizer process
                self.process_training[key] = subprocess.Popen(['python', self.training_visualizer_script], stdin=subprocess.PIPE, text=True)

        num_of_epochs = self.n4m.run_training_params['num_of_epochs']
        if epoch+1 <= num_of_epochs:
            for key in self.n4m.model_def['Minimizers'].keys():
                if val_losses:
                    val_loss = val_losses[key][epoch]
                else:
                    val_loss = []
                data = {"key":key, "last": num_of_epochs-(epoch+1), "epoch":epoch, "train_losses": train_losses[key][epoch], "val_losses": val_loss}
                try:
                    # Send data to the visualizer process
                    self.process_training[key].stdin.write(f"{json.dumps(data)}\n")
                    self.process_training[key].stdin.flush()
                except BrokenPipeError:
                    print("The visualizer process has been closed.")

        if epoch+1 == num_of_epochs:
            for key in self.n4m.model_def['Minimizers'].keys():
                self.process_training[key].terminate()

    def showResult(self, name_data):
        super().showResult(name_data)
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
                print("The visualizer process has been closed.")

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

                # line = json.dumps(data)
                # name, x, x0, x1, params, output = None, None, None, None, None, None
                # if line:
                #     try:
                #         # Convert to float and append to buffer
                #         data_point = json.loads(line)
                #         name = data_point['name']
                #         if 'x1' in data_point.keys():
                #             x0 = data_point['x0']
                #             x1 = data_point['x1']
                #         else:
                #             x = data_point['x0']
                #         params = data_point['params']
                #         input_names = data_point['input_names']
                #         output = data_point['output']
                #     except ValueError:
                #         exit()
                #
                # fig = plt.figure()
                # # Clear the current plot
                # plt.clf()
                # if 'x1' in data_point.keys():
                #     x0, x1 = np.meshgrid(x0, x1)
                #     ax = fig.add_subplot(111, projection='3d')
                #     ax.plot_surface(x0, x1, np.array(output), cmap='viridis')
                #     ax.set_xlabel(input_names[0])
                #     ax.set_ylabel(input_names[1])
                #     ax.set_zlabel(f'{name} output')
                #     for ind in range(len(input_names)-2):
                #         fig.text(0.01, 0.9-0.05*ind, f"{input_names[ind+2]} ={params[ind]}", fontsize=10, color='blue',  style='italic')
                #
                # else:
                #     plt.plot(np.array(x), np.array(output),  linewidth=2)
                #     plt.xlabel(input_names[0])
                #     plt.ylabel(f'{name} output')
                #     for ind in range(len(input_names)-1):
                #         plt.text(0.95, 0.5, f"{input_names[ind+1]} ={params[ind]}", fontsize=14, color='blue', weight='bold', style='italic')
                #
                # plt.title(f'Function {name}')
                # plt.show()
                #exit()

                self.process_function[key] = subprocess.Popen(['python', self.function_visualizer_script],
                                                                   stdin=subprocess.PIPE,
                                                                   text=True)
                try:
                    # Send data to the visualizer process
                    self.process_function[key].stdin.write(f"{json.dumps(data)}\n")
                    self.process_function[key].stdin.flush()
                except BrokenPipeError:
                    print("The visualizer process has been closed.")




import subprocess, json, torch, inspect

import matplotlib.pyplot as plt
import numpy as np

from neu4mes.visualizer.textvisualizer import TextVisualizer
from neu4mes.fuzzify import triangular, rectangular, custom_function, return_fuzzify
from neu4mes.parametricfunction import return_standard_inputs, return_function
from neu4mes.utils import check
from mplplots import plots

class MPLVisualizer(TextVisualizer):
    def __init__(self, verbose = 1):
        super().__init__(verbose)
        # Path to the data visualizer script
        import signal
        import sys
        self.training_visualizer_script = 'neu4mes/visualizer/dynamicmpl/trainingplot.py'
        self.time_series_visualizer_script = 'neu4mes/visualizer/dynamicmpl/resultsplot.py'
        self.fuzzy_visualizer_script = 'neu4mes/visualizer/dynamicmpl/fuzzyplot.py'
        self.function_visualizer_script = 'neu4mes/visualizer/dynamicmpl/functionplot.py'
        self.process_training = {}
        self.process_results = {}
        self.process_function = {}
        self.closed_process = False
        def signal_handler(sig, frame):
            for key in self.process_training.keys():
                self.process_training[key].terminate()
            for key in self.process_results.keys():
                self.process_results[key].terminate()
            for key in self.process_function.keys():
                self.process_function[key].terminate()
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
                data = {"title":"Training", "key": key, "last": num_of_epochs - (epoch + 1), "epoch": epoch,
                        "train_losses": train_losses[key][epoch], "val_losses": val_loss}
                try:
                    # Send data to the visualizer process
                    self.process_training[key].stdin.write(f"{json.dumps(data)}\n")
                    self.process_training[key].stdin.flush()
                except BrokenPipeError:
                    if self.closed_process is False:
                        self.closed_process = True
                        self.warning("The visualizer process has been closed.")

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
                if self.closed_process is False:
                    self.closed_process = True
                    self.warning("The visualizer process has been closed.")

    def showWeights(self, weights = None):
        pass

    def showFunctions(self, functions = None, xlim = None, num_points = 1000):
        check(self.n4m.neuralized, ValueError, "The model has not been neuralized.")
        for key, value in self.n4m.model_def['Functions'].items():
            if key in functions:
                if 'functions' in self.n4m.model_def['Functions'][key]:
                    x, activ_fun = return_fuzzify(value, xlim, num_points)
                    data = {"name": key,
                            "x": x,
                            "y": activ_fun,
                            "chan_centers": value['centers']}
                    # Start the data visualizer process
                    self.process_function[key] = subprocess.Popen(['python', self.fuzzy_visualizer_script],
                                                                  stdin=subprocess.PIPE,
                                                                  text=True)
                elif 'code':
                    function_inputs = return_standard_inputs(value, self.n4m.model_def_values, xlim, num_points)
                    function_output, function_input_list = return_function(value, function_inputs)

                    data = {"name": key}
                    if value['n_input'] == 2:
                        data['x0'] = function_inputs[0].reshape(num_points, num_points).tolist()
                        data['x1'] = function_inputs[1].reshape(num_points, num_points).tolist()
                        data['output'] = function_output.reshape(num_points, num_points).tolist()
                    else:
                        data['x0'] = function_inputs[0].reshape(num_points).tolist()
                        data['output'] = function_output.reshape(num_points).tolist()
                    data['params'] = []
                    for i, key in enumerate(value['params_and_consts']):
                        data['params'] += [function_inputs[i+value['n_input']].tolist()]
                    data['input_names'] = function_input_list

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







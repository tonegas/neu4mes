import subprocess, json, torch

import numpy as np

from neu4mes.visualizer.textvisualizer import TextVisualizer
from neu4mes.fuzzify import triangular, rectangular, custom_function

class MPLVisulizer(TextVisualizer):
    def __init__(self, verbose=1):
        import signal
        import sys
        super().__init__(verbose)
        # Path to the data visualizer script
        self.training_visualizer_script = 'neu4mes/visualizer/trainingplot.py'
        self.time_series_visualizer_script = 'neu4mes/visualizer/resultsplot.py'
        self.fuzzy_visualizer_script = 'neu4mes/visualizer/fuzzyplot.py'
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

    def showFunctions(self, functions = None, range = None):
        for key, value in self.n4m.model_def['Functions'].items():
            if key in functions:
                if 'functions' in self.n4m.model_def['Functions'][key]:
                    x_test = torch.from_numpy(np.linspace(value['centers'][0] - 2, value['centers'][-1] + 2, num=1000))
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
                            "y": activ_fun}
                elif 'code':
                    pass

                self.process_function[key] = subprocess.Popen(['python', self.fuzzy_visualizer_script],
                                                                   stdin=subprocess.PIPE,
                                                                   text=True)
                try:
                    # Send data to the visualizer process
                    self.process_function[key].stdin.write(f"{json.dumps(data)}\n")
                    self.process_function[key].stdin.flush()
                except BrokenPipeError:
                    print("The visualizer process has been closed.")




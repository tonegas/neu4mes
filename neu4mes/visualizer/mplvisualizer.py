import subprocess, json, os, importlib

from neu4mes.visualizer.textvisualizer import TextVisualizer
from neu4mes.fuzzify import return_fuzzify
from neu4mes.parametricfunction import return_standard_inputs, return_function
from neu4mes.utils import check

from neu4mes.logger import logging, Neu4MesLogger
log = Neu4MesLogger(__name__, logging.INFO)

def get_library_path(library_name):
    spec = importlib.util.find_spec(library_name)
    if spec is None:
        raise ImportError(f"Library {library_name} not found")
    return os.path.dirname(spec.origin)

class MPLVisualizer(TextVisualizer):
    def __init__(self, verbose = 1):
        super().__init__(verbose)
        # Path to the data visualizer script
        import signal
        import sys
        get_library_path('neu4mes')
        self.training_visualizer_script = os.path.join(get_library_path('neu4mes'),'visualizer/dynamicmpl/trainingplot.py')
        self.time_series_visualizer_script = os.path.join(get_library_path('neu4mes'),'visualizer/dynamicmpl/resultsplot.py')
        self.fuzzy_visualizer_script = os.path.join(get_library_path('neu4mes'),'visualizer/dynamicmpl/fuzzyplot.py')
        self.function_visualizer_script = os.path.join(get_library_path('neu4mes'),'visualizer/dynamicmpl/functionplot.py')
        self.process_training = {}
        self.process_results = {}
        self.process_function = {}
        def signal_handler(sig, frame):
            for key in self.process_training.keys():
                self.process_training[key].terminate()
                self.process_training[key].wait()
            for name_data in self.process_results.keys():
                for key in self.process_results[name_data].keys():
                    self.process_results[name_data][key].terminate()
                    self.process_results[name_data][key].wait()
            self.process_results = {}
            for key in self.process_function.keys():
                self.process_function[key].terminate()
                self.process_functios[key].wait()
            sys.exit()

        signal.signal(signal.SIGINT, signal_handler)

    def showStartTraining(self):
        pass

    def showTraining(self, epoch, train_losses, val_losses):
        for key in self.n4m.model_def['Minimizers'].keys():
            if key in self.process_training and self.process_training[key].poll() is None:
                self.process_training[key].terminate()
                self.process_training[key].wait()
        self.process_training[key] = {}

        if epoch == 0:
            for key in self.n4m.model_def['Minimizers'].keys():
                # Start the data visualizer process
                self.process_training[key] = subprocess.Popen(['python', self.training_visualizer_script], stdin=subprocess.PIPE, text=True)

        num_of_epochs = self.n4m.run_training_params['num_of_epochs']
        train_dataset = self.n4m.run_training_params['train_dataset']
        validation_dataset = self.n4m.run_training_params['validation_dataset']
        if epoch+1 <= num_of_epochs:
            for key in self.n4m.model_def['Minimizers'].keys():
                if val_losses:
                    val_loss = val_losses[key][epoch]
                else:
                    val_loss = []
                data = {"title":f"Training on {train_dataset} and {validation_dataset}", "key": key, "last": num_of_epochs - (epoch + 1), "epoch": epoch,
                        "train_losses": train_losses[key][epoch], "val_losses": val_loss}
                try:
                    # Send data to the visualizer process
                    self.process_training[key].stdin.write(f"{json.dumps(data)}\n")
                    self.process_training[key].stdin.flush()
                    self.process_training[key].stdin.close()
                except BrokenPipeError:
                    self.closeTraining()
                    log.warning("The visualizer process has been closed.")

    def showResult(self, name_data):
        super().showResult(name_data)
        check(name_data in self.n4m.performance, ValueError, f"Results not available for {name_data}.")
        if name_data in self.process_results:
            for key in self.n4m.model_def['Minimizers'].keys():
                if key in self.process_results[name_data] and self.process_results[name_data][key].poll() is None:
                    self.process_results[name_data][key].terminate()
                    self.process_results[name_data][key].wait()
                self.process_results[name_data][key] = None
        self.process_results[name_data] = {}

        for key in self.n4m.model_def['Minimizers'].keys():
            # Start the data visualizer process
            self.process_results[name_data][key] = subprocess.Popen(['python', self.time_series_visualizer_script], stdin=subprocess.PIPE,
                                                    text=True)
            data = {"name_data": name_data,
                    "key": key,
                    "performance": self.n4m.performance[name_data][key],
                    "prediction_A": self.n4m.prediction[name_data][key]['A'],
                    "prediction_B": self.n4m.prediction[name_data][key]['B'],
                    "sample_time": self.n4m.model_def['Info']["SampleTime"]}
            try:
                # Send data to the visualizer process
                self.process_results[name_data][key].stdin.write(f"{json.dumps(data)}\n")
                self.process_results[name_data][key].stdin.flush()
                self.process_results[name_data][key].stdin.close()
            except BrokenPipeError:
                self.closeResult(self, name_data)
                log.warning(f"The visualizer {name_data} process has been closed.")

    def showWeights(self, weights = None):
        pass

    def showFunctions(self, functions = None, xlim = None, num_points = 1000):
        check(self.n4m.neuralized, ValueError, "The model has not been neuralized.")
        for key, value in self.n4m.model_def['Functions'].items():
            if key in functions:
                if key in self.process_function and self.process_function[key].poll() is None:
                    self.process_function[key].terminate()
                    self.process_function[key].wait()

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
                    function_inputs = return_standard_inputs(value, self.n4m.model_def, xlim, num_points)
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
                    self.process_function[key].stdin.close()
                except BrokenPipeError:
                    self.closeFunctions()
                    log.warning(f"The visualizer {functions} process has been closed.")

    def closeFunctions(self, functions = None):
        if functions is None:
            for key in self.process_function.keys():
                self.process_function[key].terminate()
                self.process_function[key].wait()
            self.process_function = {}
        else:
            for key in functions:
                self.process_function[key].terminate()
                self.process_function[key].wait()
                self.process_function.pop(key)

    def closeTraining(self, minimizer = None):
        if minimizer is None:
            for key in self.n4m.model_def['Minimizers'].keys():
                if key in self.process_training and self.process_training[key].poll() is None:
                    self.process_training[key].terminate()
                    self.process_training[key].wait()
            self.process_training[key] = {}
        else:
            self.process_training[minimizer].terminate()
            self.process_training[minimizer].wait()
            self.process_training.pop(minimizer)

    def closeResult(self, name_data = None, minimizer = None):
        if name_data is None:
            check(minimizer is None, ValueError, "If name_data is None, minimizer must be None.")
            for name_data in self.process_results.keys():
                for key in self.process_results[name_data].keys():
                    self.process_results[name_data][key].terminate()
                    self.process_results[name_data][key].wait()
            self.process_results = {}
        else:
            if minimizer is None:
                for key in self.process_results[name_data].keys():
                    self.process_results[name_data][key].terminate()
                    self.process_results[name_data][key].wait()
                self.process_results[name_data] = {}
            else:
                self.process_results[name_data][minimizer].terminate()
                self.process_results[name_data][minimizer].wait()
                self.process_results[name_data].pop(minimizer)






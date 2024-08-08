import subprocess
import json

from neu4mes.visualizer.textvisualizer import TextVisualizer

class MPLVisulizer(TextVisualizer):
    def __init__(self, verbose=1):
        import signal
        import sys
        super().__init__(verbose)
        # Path to the data visualizer script
        self.training_visualizer_script = 'neu4mes/visualizer/trainingplot.py'
        self.time_series_visualizer_script = 'neu4mes/visualizer/resultsplot.py'
        self.process_training = {}
        self.process_results = {}

        def signal_handler(sig, frame):
            for key in self.n4m.minimize_dict.keys():
                self.process_training[key].terminate()
                self.process_results[key].terminate()
            sys.exit()

        signal.signal(signal.SIGINT, signal_handler)

    def showTraining(self, epoch, train_losses, val_losses):
        if epoch == 0:
            for key in self.n4m.minimize_dict.keys():
                # Start the data visualizer process
                self.process_training[key] = subprocess.Popen(['python', self.training_visualizer_script], stdin=subprocess.PIPE, text=True)

        if epoch+1 <= self.n4m.num_of_epochs:
            for key in self.n4m.minimize_dict.keys():
                if val_losses:
                    val_loss = val_losses[key][epoch]
                else:
                    val_loss = []
                data = {"key":key, "last": self.n4m.num_of_epochs-(epoch+1), "epoch":epoch, "train_losses": train_losses[key][epoch], "val_losses": val_loss}
                try:
                    # Send data to the visualizer process
                    self.process_training[key].stdin.write(f"{json.dumps(data)}\n")
                    self.process_training[key].stdin.flush()
                except BrokenPipeError:
                    print("The visualizer process has been closed.")

        if epoch+1 == self.n4m.num_of_epochs:
            for key in self.n4m.minimize_dict.keys():
                self.process_training[key].terminate()

    def showOneResult(self, name_data = None):
        super().showOneResult(name_data)
        for key in self.n4m.minimize_dict.keys():
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



import numpy as np
from pprint import pformat
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import multiprocessing as mp
from multiprocessing import Process, Queue, current_process
import signal
import sys
import time
import subprocess
import json

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

def plot_graph(data_queue, stop_event):
    def signal_handler(sig, frame):
        print('Ctrl+C pressed plot')
        stop_event.set()
        plt.close('all')

    signal.signal(signal.SIGINT, signal_handler)

    plt.ion()
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = ax.plot([], [], 'ro')

    def init():
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        return ln,

    def update(frame):
        while not data_queue.empty():
            data = data_queue.get()
            xdata.append(data[0])
            ydata.append(data[1])
            ln.set_data(xdata, ydata)
        return ln,

    def handle_close(evt):
        stop_event.set()

    plt.gcf().canvas.mpl_connect('close_event', handle_close)
    ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=50, save_count=50)
    plt.show(block=True)

    # while not stop_event.is_set():
    #     plt.pause(0.1)

class BlitManager:
    def __init__(self, canvas, animated_artists=()):
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for subclasses of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        """
        self.canvas = canvas
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
        cv.flush_events()

class ProcessPlotter:
    def __init__(self):
        self.x = []
        self.y = []

    def terminate(self):
        plt.close('all')

    def call_back(self):
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                self.x.append(command[0])
                self.y.append(command[1])
                self.ax.plot(self.x, self.y, 'ro')
        self.fig.canvas.draw()
        return True

    def __call__(self, pipe):
        print('starting plotter...')

        self.pipe = pipe
        self.fig, self.ax = plt.subplots()
        timer = self.fig.canvas.new_timer(interval=1000)
        timer.add_callback(self.call_back)
        timer.start()

        print('...done')
        plt.show()

class NBPlot:
    def __init__(self):
        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = ProcessPlotter()
        self.plot_process = mp.Process(
            target=self.plotter, args=(plotter_pipe,), daemon=True)
        self.plot_process.start()

    def plot(self, finished=False):
        send = self.plot_pipe.send
        if finished:
            send(None)
        else:
            data = np.random.random(2)
            send(data)

def plotter(queue):
    plt.ion()  # Interactive mode on
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'b-')  # Initial empty plot

    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1)

    while True:
        if not queue.empty():
            x, y = queue.get()
            line.set_xdata(x)
            line.set_ydata(y)
            ax.set_xlim(0, max(10, x[-1] + 1))
            plt.draw()
            plt.pause(0.01)

class Visualizer():
    def __init__(self, neu4mes):
        self.n4m = neu4mes

    def warning(self, msg):
        print(color(msg, YELLOW))

    def showModel(self):
        pass

    def showMinimizeError(self,variable_name):
        pass

    def showModelInputWindow(self):
        pass

    def showModelRelationSamples(self):
        pass

    def showBuiltModel(self):
        pass

    def showDataset(self):
        pass

    def showTraining(self, epoch, train_losses, test_losses):
        pass

    def showResults(self):
        pass

class TextVisualizer(Visualizer):
    def __init__(self, neu4mes, verbose=1):
        super().__init__(neu4mes)
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

    def showDataset(self, name):
        if self.verbose >= 1:
            self.__title(" Neu4mes Model Dataset ")
            self.__param("Dataset Name:", name)
            self.__param("Number of files:", f'{self.n4m.file_count}')
            self.__param("Total number of samples:", f'{self.n4m.num_of_samples[name]}')
            self.__param("Available Datasets:", f'{self.n4m.datasets_loaded}')
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

    def showResults(self):
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

class StandardVisualizer(TextVisualizer):
    def __init__(self, neu4mes, verbose=1):
        super().__init__(neu4mes, verbose)
        # Path to the data visualizer script
        visualizer_script = 'multiprov.py'

        # Start the data visualizer process
        self.process = subprocess.Popen(['python', visualizer_script], stdin=subprocess.PIPE, text=True)


        # super().__init__(neu4mes, verbose)
        # self.terminated = False
        # #time.sleep(1)
        # mp.set_start_method('spawn',force=True)  # Ensure the 'spawn' start method for compatibility
        # self.data_queue = mp.Queue()
        # self.stop_event = mp.Event()
        #
        # def signal_handler(sig, frame):
        #     print('Ctrl+C pressed __main__')
        #     self.stop_event.set()
        #     self.plot_process.join()
        #     if not self.terminated:
        #         sys.exit()
        #
        # signal.signal(signal.SIGINT, signal_handler)


        # sys.exit()

        # self.queue = Queue()
        # plot_process = Process(target=plotter, args=(self.queue,))
        # plot_process.start()

        #plot_process.terminate()

        # if plt.get_backend() == "MacOSX":
        #     mp.set_start_method("forkserver")
        # self.pl = NBPlot()
        # #plt.ion()
        # x = np.linspace(0, 2 * np.pi, 100)
        # #
        # self.fig, self.ax = plt.subplots()
        #
        # # animated=True tells matplotlib to only draw the artist when we
        # # explicitly request it
        # (self.ln,) = self.ax.plot(x, np.sin(x))
        #
        # # make sure the window is raised, but the script keeps going
        # plt.show(block=False)
        #
        # # stop to admire our empty window axes and ensure it is rendered at
        # # least once.
        # #
        # # We need to fully draw the figure at its final size on the screen
        # # before we continue on so that :
        # #  a) we have the correctly sized and drawn background to grab
        # #  b) we have a cached renderer so that ``ax.draw_artist`` works
        # # so we spin the event loop to let the backend process any pending operations
        # plt.pause(0.1)
        #
        # # get copy of entire figure (everything inside fig.bbox) sans animated artist
        # self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        # # draw the animated artist, this uses a cached renderer
        # self.ax.draw_artist(self.ln)
        # # show the result to the screen, this pushes the updated RGBA buffer from the
        # # renderer to the GUI framework so you can see it
        # self.fig.canvas.blit(self.fig.bbox)

        #
        # import numpy as np
        # x = np.linspace(0, 10 * np.pi, 100)
        # y = np.sin(x)
        #
        # plt.ion()
        # #plt.show(block=False)
        # self.fig, self.ax = plt.subplots()
        # self.line1, = self.ax.plot(x, y, 'b-')
        # plt.ioff()

        #pass
        # self.n4m = neu4mes
        # from torch.utils.tensorboard import SummaryWriter
        # import random
        # self.writer = SummaryWriter('runs/'+str(random.random()))


    # updates the data and graph
    # def updateTraining(self, epoch, train_losses, test_losses):
    #     self.plt.ion()
    #     #for key in self.n4m.minimize_dict.keys():
    #     key = list(self.n4m.minimize_dict.keys())[0]
    #     self.train.set_xdata(range(epoch))
    #     self.train.set_ydata(train_losses[key][:epoch])
    #     # if J_test_list is not None:
    #     #     self.test.set_xdata(xrange(len(J_test_list[:epoch + 1])))
    #     #     self.test.set_ydata(J_test_list[:epoch + 1])
    #     #     self.ax.set_ylim(
    #     #         [0, max(max(J_train_list) + max(J_train_list) * 0.1, max(J_test_list) + max(J_test_list) * 0.1)])
    #     # else:
    #     #     self.ax.set_ylim([0, max(J_train_list) + max(J_train_list) * 0.1])
    #     self.fig1.canvas.draw()
    #     self.plt.ioff()
    # #     for key in self.n4m.minimize_dict.keys():
    # #         # creating a new graph or updating the graph
    # #         self.graph_train.set_xdata()
    # #         self.graph_train.set_ydata(y)
    # #         self.graph_test.set_xdata(x)
    # #         self.graph_test.set_ydata(y)
    # #
    # # anim = FuncAnimation(fig, update, frames=None)
    # # plt.show()

    def showTraining(self, epoch, train_losses, test_losses):
        # import random
        try:
            # Generate random data
            data = {"epoch":epoch, "train_losses": train_losses[epoch], "test_losses": test_losses[epoch]}
            try:
                # Send data to the visualizer process
                self.process.stdin.write(f"{json.dumps(data)}\n")
                self.process.stdin.flush()
            except BrokenPipeError:
                print("The visualizer process has been closed.")
                # Wait for a short period before generating the next data point
                # time.sleep(0.001)
        except KeyboardInterrupt:
            # Terminate the visualizer process gracefully
            self.process.terminate()

        # if epoch == 0:
        #     self.plot_process = mp.Process(target=plot_graph, args=(self.data_queue, self.stop_event))
        #     self.plot_process.start()
        #     #self.plot_process.join()
        # # time.sleep(1)
        # # for i in range(100):
        #     # if stop_event.is_set():
        #     #    break
        # x = epoch * 0.1
        # y = np.sin(x)
        # self.data_queue.put((x, y))
        # #time.sleep(0.1)
        # if epoch + 1 == self.n4m.num_of_epochs:
        #     self.stop_event.set()
        #     self.terminated = True
        # # self.pl.plot()
        # x = np.linspace(0, 10, 1000)
        # y = np.sin(x + epoch / 10.0)
        # self.queue.put((x, y))
        # #time.sleep(0.1)
        #
        # x = np.linspace(0, 2 * np.pi, 100)
        # # reset the background back in the canvas state, screen unchanged
        # self.fig.canvas.restore_region(self.bg)
        # # update the artist, neither the canvas state nor the screen have changed
        # self.ln.set_ydata(np.sin(x + (epoch / 100) * np.pi))
        # # re-render the artist, updating the canvas state, but not the screen
        # self.ax.draw_artist(self.ln)
        # # copy the image to the GUI state, but screen might not be changed yet
        # self.fig.canvas.blit(self.fig.bbox)
        # # flush any pending GUI events, re-painting the screen if needed
        # #self.fig.canvas.flush_events()
        # self.fig.canvas.draw()
        # # you can put a pause in if you want to slow things down
        # plt.pause(.001)
        #
        # x = np.linspace(0, 10 * np.pi, 100)
        # plt.ion()
        # self.line1.set_ydata(np.sin(0.5 * x + epoch/10.0))
        # self.fig.canvas.draw()
        # #plt.show(block=False)
        # plt.pause(0.01)
        # plt.ioff()
        #plt.show()
        #print('ok')

        # Add a short pause to improve animation smoothness
        # ...log the running loss
        # key = list(self.n4m.minimize_dict.keys())[0]
        # self.writer.add_scalar('training loss',
        #                   train_losses[key][epoch] / 1000,
        #                   epoch * len(train_losses))
        # print('OK')

        # ...log a Matplotlib Figure showing the model's predictions on a
        # random mini-batch
        # self.writer.add_figure('predictions vs. actuals',
        #                   plot_classes_preds(net, inputs, labels),
        #                   global_step=epoch * len(trainloader) + i)
        # self.plt.ion()
        # if epoch == 0:
        #     self.fig1 = self.plt.figure(1)
        #     self.ax = self.fig1.add_subplot(1,1,1)
        #     self.ax.set_title('Errors History (J)')
        #     self.train, = self.ax.plot([], [], color='green', marker='^', label='Training')
        #     self.test, = self.ax.plot([], [], color='blue', marker='s', label='Test')
        #     self.ax.set_xlabel('Epochs')
        #     self.ax.set_ylabel(r'$||J||_2/N$')
        #     self.ax.legend()
        #
        # if epoch < self.n4m.num_of_epochs:
        #     self.updateTraining(epoch, train_losses, test_losses)
        # self.plt.ioff()
    #
    #
    #
    #     self.__title(" Neu4mes Training ", 12 + (len(self.n4m.minimize_dict) + 1) * 20)
    #     print(color('|' + (f'Epoch').center(10, ' ') + '|'), end='')
    #     for key in self.n4m.minimize_dict.keys():
    #         print(color((f'{key}').center(19, ' ') + '|'), end='')
    #     print(color((f'Total').center(19, ' ') + '|'))
    #
    #
    #     self.fig, self.ax = self.plt.subplots(2*len(output_keys), 2,
    #                                     gridspec_kw={'width_ratios': [5, 1], 'height_ratios': [2, 1]*len(output_keys)})
    #
    # def showResults(self, neu4mes, output_keys, performance = None):
    #     # Plot
    #     self.fig, self.ax = self.plt.subplots(2*len(output_keys), 2,
    #                                     gridspec_kw={'width_ratios': [5, 1], 'height_ratios': [2, 1]*len(output_keys)})
    #     if len(self.ax.shape) == 1:
    #         self.ax = np.expand_dims(self.ax, axis=0)
    #     #plotsamples = self.prediction.shape[1]s
    #     plotsamples = 200
    #     for i in range(0, neu4mes.prediction.shape[0]):
    #         # Zoomed test data
    #         self.ax[2*i,0].plot(neu4mes.prediction[i], linestyle='dashed')
    #         self.ax[2*i,0].plot(neu4mes.label[i])
    #         self.ax[2*i,0].grid('on')
    #         self.ax[2*i,0].set_xlim((performance['max_se_idxs'][i]-plotsamples, performance['max_se_idxs'][i]+plotsamples))
    #         self.ax[2*i,0].vlines(performance['max_se_idxs'][i], neu4mes.prediction[i][performance['max_se_idxs'][i]], neu4mes.label[i][performance['max_se_idxs'][i]],
    #                                 colors='r', linestyles='dashed')
    #         self.ax[2*i,0].legend(['predicted', 'test'], prop={'family':'serif'})
    #         self.ax[2*i,0].set_title(output_keys[i], family='serif')
    #         # Statitics
    #         self.ax[2*i,1].axis("off")
    #         self.ax[2*i,1].invert_yaxis()
    #         if performance:
    #             text = "Rmse test: {:3.6f}\nFVU: {:3.6f}".format(#\nAIC: {:3.6f}
    #                 neu4mes.performance['rmse_test'][i],
    #                 #neu4mes.performance['aic'][i],
    #                 neu4mes.performance['fvu'][i])
    #             self.ax[2*i,1].text(0, 0, text, family='serif', verticalalignment='top')
    #         # test data
    #         self.ax[2*i+1,0].plot(neu4mes.prediction[i], linestyle='dashed')
    #         self.ax[2*i+1,0].plot(neu4mes.label[i])
    #         self.ax[2*i+1,0].grid('on')
    #         self.ax[2*i+1,0].legend(['predicted', 'test'], prop={'family':'serif'})
    #         self.ax[2*i+1,0].set_title(output_keys[i], family='serif')
    #         # Empty
    #         self.ax[2*i+1,1].axis("off")
    #     self.fig.tight_layout()
    #     self.plt.show()

    # def __init__(self, epochs_num=None, weights_list=None):
    #     plt.ion()
    #     self.fig1 = plt.figure(1)
    #     self.ax = self.fig1.add_subplot(111)
    #     self.ax.set_title('Errors History (J)')
    #     self.train, = self.ax.plot(xrange(len([])), [], color='green', marker='^', label='Training')
    #     self.test, = self.ax.plot(xrange(len([])), [], color='blue', marker='s', label='Test')
    #     self.ax.set_xlabel('Epochs')
    #     self.ax.set_ylabel(r'$||J||_2/N$')
    #     self.ax.legend()
    #     if epochs_num is not None:
    #         self.ax.set_xlim([0, epochs_num])
    #
    #     self.fig2 = plt.figure(2)
    #     self.ax2 = self.fig2.add_subplot(111)
    #     self.ax2.set_title('Loss Gradient History (dJ/dy)')
    #     self.dJdy, = self.ax2.plot(xrange(len([])), [], color='red', marker='o')
    #     self.ax2.set_xlabel('Epochs')
    #     self.ax2.set_ylabel(r'$||\delta J/\delta y||_2/N$')
    #     if epochs_num is not None:
    #         self.ax2.set_xlim([0, epochs_num])
    #
    #     self.weightsline = []
    #     self.weights = []
    #     if weights_list is not None:
    #         self.fig3 = plt.figure(3)
    #         self.ax3 = self.fig3.add_subplot(111)
    #         self.ax3.set_title('Weights Norm2')
    #         for weight in weights_list:
    #             self.weights.append(weights_list[weight])
    #             line, = self.ax3.plot(xrange(len([])), [], marker='o', label=weight)
    #             self.weightsline.append(line)
    #         self.ax3.set_xlabel('Epochs')
    #         self.ax3.set_ylabel(r'$||W_i||_2$')
    #         if epochs_num is not None:
    #             self.ax3.set_xlim([0, epochs_num])
    #     self.ax3.legend()
    #     plt.ioff()
    #
    # def show(self, epoch, J_train_list, dJdy_list=None, J_test_list=None):
    #     plt.ion()
    #     self.train.set_xdata(xrange(len(J_train_list[:epoch + 1])))
    #     self.train.set_ydata(J_train_list[:epoch + 1])
    #     if J_test_list is not None:
    #         self.test.set_xdata(xrange(len(J_test_list[:epoch + 1])))
    #         self.test.set_ydata(J_test_list[:epoch + 1])
    #         self.ax.set_ylim(
    #             [0, max(max(J_train_list) + max(J_train_list) * 0.1, max(J_test_list) + max(J_test_list) * 0.1)])
    #     else:
    #         self.ax.set_ylim([0, max(J_train_list) + max(J_train_list) * 0.1])
    #     self.fig1.canvas.draw()
    #
    #     self.dJdy.set_xdata(xrange(len(dJdy_list[:epoch + 1])))
    #     self.dJdy.set_ydata(dJdy_list[:epoch + 1])
    #     self.ax2.set_ylim([min(dJdy_list) - min(dJdy_list) * 0.1, max(dJdy_list) + max(dJdy_list) * 0.1])
    #     self.fig2.canvas.draw()
    #
    #     if len(self.weights) > 0:
    #         minval = 0
    #         maxval = 0
    #         for weight, weightline in zip(self.weights, self.weightsline):
    #             if epoch == 0:
    #                 weightline.set_ydata([])
    #             weightline.set_xdata(xrange(len(dJdy_list[:epoch + 1])))
    #             weightline.set_ydata(np.append(weightline.get_ydata(), np.linalg.norm(weight.get())))
    #             minval = minval if minval < min(weightline.get_ydata()) else min(weightline.get_ydata())
    #             maxval = maxval if maxval > max(weightline.get_ydata()) else max(weightline.get_ydata())
    #         self.ax3.set_ylim([minval - minval * 0.1, maxval + maxval * 0.1])
    #         self.fig3.canvas.draw()
    #     plt.ioff()
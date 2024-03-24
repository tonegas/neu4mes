# Double mass spring damper

from neu4mes import *
from neu4mes.visualizer import StandardVisualizer


massamollasmorzatore = Neu4mes(visualizer = StandardVisualizer())

x = Input('x')
y = Input('y')
F = ControlInput('force')
# F2 = Input('force2')
relation = lambda: Linear(x.tw(2))+Linear(y.tw(2))+Linear(F)

x_z = Output(x.z(-1), relation())
# y_z = Output(y.z(-1), relation())
massamollasmorzatore.addModel(x_z)
# massamollasmorzatore.addModel(y_z)

massamollasmorzatore.neuralizeModel(0.05, prediction_window = 50)

data_struct = ['time','x','y','force','force2']
data_folder = './datasets/pendulum/data/'
massamollasmorzatore.loadData(data_struct, folder = data_folder)

massamollasmorzatore.trainModel(states = [x_z], test_percentage = 10, show_results = True)
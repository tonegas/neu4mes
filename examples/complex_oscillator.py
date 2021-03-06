from neu4mes import *

massamollasmorzatore = Neu4mes()

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
data_folder = './structured_nn-code/data/pendulum-b/data/'
massamollasmorzatore.loadData(data_struct, folder = data_folder)

massamollasmorzatore.trainModel(states = [x_z], validation_percentage = 10, show_results = True)
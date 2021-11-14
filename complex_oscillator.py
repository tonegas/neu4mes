from Neu4mes import *

massamollasmorzatore = Neu4mes()

x = Input('x')
y = Input('y')
F = ControlInput('force')
relation = lambda: Linear(y.tw(2))+Linear(x.tw(2))+Linear(F)

x_z = Output(x.z(-1), relation())
y_z = Output(y.z(-1), relation())
massamollasmorzatore.addModel(x_z)
massamollasmorzatore.addModel(y_z)

massamollasmorzatore.neuralizeModel(0.05)

data_struct = ['time','x','y','force']
data_folder = './structured_nn-code/data/oscillator-linear/data/'
massamollasmorzatore.loadData(data_struct, folder = data_folder)

massamollasmorzatore.trainModel(validation_percentage = 30, states = [x_z,y_z], show_results = True)
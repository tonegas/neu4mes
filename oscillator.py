from Neu4mes import *

massamollasmorzatore = Neu4mes()

x = Input('x')
F = ControlInput('force')

x_z = Output(x.z(-1), Linear(x.tw(2))+Linear(F))
massamollasmorzatore.addModel(x_z)

massamollasmorzatore.neuralizeModel(0.05)

data_struct = ['time','x','x_s','force']
data_folder = './structured_nn-code/data/oscillator-linear/data/'
massamollasmorzatore.loadData(data_struct, folder = data_folder)

massamollasmorzatore.trainModel(validation_percentage = 30, states = [x_z], show_results = True)
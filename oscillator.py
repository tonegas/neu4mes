from Neu4mes import *

massamollasmorzatore = Neu4mes()

x = Input('x')
F = Input('force')

x_z = Output(x.z(-1), Linear(x.tw(2))+Linear(F))
massamollasmorzatore.addModel(x_z)

massamollasmorzatore.neuralizeModel(0.05)

data_struct = ['time','x','x_s','force']
data_folder = './data/data-oscillator-linear/'
massamollasmorzatore.loadData(data_struct, folder = data_folder)

massamollasmorzatore.trainModel(validation_percentage = 30, show_results = True)
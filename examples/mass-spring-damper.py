from neu4mes import *
from neu4mes.visualizer import StandardVisualizer

mass_spring_damper = Neu4mes(verbose = True,  visualizer = StandardVisualizer())
x = Input('x')
F = Input('F')
x_z = Output(x.z(-1), Linear(x.tw(2))+Linear(F))
mass_spring_damper.addModel(x_z)
mass_spring_damper.neuralizeModel(0.05)
data_struct = ['time','x','x_s','F']
data_folder = './datasets/mass-spring-damper/data/'
mass_spring_damper.loadData(data_struct, folder = data_folder)
mass_spring_damper.trainModel(test_percentage = 30, show_results = True)


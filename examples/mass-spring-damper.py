from neu4mes import *
from neu4mes.visualizer import StandardVisualizer

# Create neural model
x = Input('x')
F = Input('F')
x_z = Output(x.z(-1), Linear(x.tw(2))+Linear(F))

# Add the neural model to the neu4mes structure and neuralization of the model
mass_spring_damper = Neu4mes(verbose = True,  visualizer = StandardVisualizer())
mass_spring_damper.addModel(x_z)
mass_spring_damper.neuralizeModel(0.05)

# Data load
data_struct = ['time','x','x_s','F']
data_folder = './datasets/mass-spring-damper/data/'
mass_spring_damper.loadData(data_struct, folder = data_folder)

# Neural network train
mass_spring_damper.trainModel(test_percentage = 30, show_results = True)


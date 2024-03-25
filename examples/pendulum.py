from neu4mes import *
from neu4mes.visualizer import StandardVisualizer

# Create neu4mes structure
pendolum = Neu4mes(verbose = True, visualizer = StandardVisualizer())

# Create neural model
theta = Input('theta')
T     = Input('torque')
lin_theta = Linear(theta.tw(1.5))
sin_theta = LinearBias(Sin(theta))
torque = Linear(T)
theta_z = Output(theta.z(-1), lin_theta+sin_theta+torque)

# Add the neural model to the neu4mes structure and neuralization of the model
pendolum.addModel(theta_z)
pendolum.neuralizeModel(0.05)

# Data load
data_struct = ['time','theta','theta_s','','','torque']
data_folder = './datasets/pendulum/data/'
pendolum.loadData(data_struct, folder = data_folder)

# Neural network train
pendolum.trainModel(test_percentage = 30, show_results = True)
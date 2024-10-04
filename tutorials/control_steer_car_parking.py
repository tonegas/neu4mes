"""
 _____ _____ _____ ___________     _____ _____ _   _ ___________ _____ _      _      ___________    ______  ___  ______ _   _______ _   _ _____ 
/  ___|_   _|  ___|  ___| ___ \   /  __ \  _  | \ | |_   _| ___ \  _  | |    | |    |  ___| ___ \   | ___ \/ _ \ | ___ \ | / /_   _| \ | |  __ \
\ `--.  | | | |__ | |__ | |_/ /   | /  \/ | | |  \| | | | | |_/ / | | | |    | |    | |__ | |_/ /   | |_/ / /_\ \| |_/ / |/ /  | | |  \| | |  \/
 `--. \ | | |  __||  __||    /    | |   | | | | . ` | | | |    /| | | | |    | |    |  __||    /    |  __/|  _  ||    /|    \  | | | . ` | | __ 
/\__/ / | | | |___| |___| |\ \    | \__/\ \_/ / |\  | | | | |\ \\ \_/ / |____| |____| |___| |\ \    | |   | | | || |\ \| |\  \_| |_| |\  | |_\ \
\____/  \_/ \____/\____/\_| \_|    \____/\___/\_| \_/ \_/ \_| \_|\___/\_____/\_____/\____/\_| \_|   \_|   \_| |_/\_| \_\_| \_/\___/\_| \_/\____/
                                                                                                                                                                                                                                   
This tutorial implements the physics-driven neural steering controller of the paper 
'Fast Planning and Tracking of Complex Autonomous Parking Maneuvers With Optimal Control and Pseudo-Neural Networks'
"""

import sys
import os
import torch
clear = lambda: os.system('clear')
clear()
print(os.getcwd())
sys.path.append(os.getcwd())
from neu4mes import *

# Create neural model inputs
curv  = Input('curv')
steer = State('steer')  # the initial condition for the state is taken from the dataset, and also when the state is reset at the end of a window

num_samples_curv  = 15
num_samples_steer = 15

# Initial conditions
h_1_guess = Parameter('h_1',values=[[0.1]])  # initial guess
h_2_guess = Parameter('h_2',values=[[0.1]])  # initial guess
h_3_guess = Parameter('h_3',values=[[0.1]])  # initial guess

# Create neural network relations:
# Curvature diagram
def curvat_diagram(curv,h_1,h_2,h_3):
  L = 3.5   # [m] wheelbase
  return torch.atan(h_1*curv + h_2*torch.pow(curv,3) + h_3*torch.pow(curv,5) + curv*L)

out_curv_diagr = ParamFun(curvat_diagram,parameters=[h_1_guess,h_2_guess,h_3_guess])(curv.sw([0,num_samples_curv]))
out_fir        = Fir()(out_curv_diagr)                    #parameter_init=init_negexp, parameter_init_params={'first_value':1, 'lambda':3}
out_arx        = Fir()(steer.sw([-num_samples_steer,0]))  #parameter_init=init_negexp, parameter_init_params={'first_value':1, 'lambda':3}
out_nn         = out_arx + out_fir
out_nn.closedLoop(steer)

# Create neural network output
out = Output('steering_angle', out_nn)

# Create neu4mes structure
steer_controller = Neu4mes(visualizer=MPLVisulizer(),seed=0)

# Add the neural model to the neu4mes structure and neuralization of the model
steer_controller.addModel('steer_ctrl',[out])
steer_controller.addMinimize('steer_error', steer.next(), out, loss_function='rmse')
steer_controller.neuralizeModel()

# # Load the training and the validation dataset
# data_struct = ['curv','steer']
# data_folder = './tutorials/datasets/vehicle_data/trainingset'
# steer_controller.loadData(name='trainingset', source=data_folder, format=data_struct, skiplines=1)
# data_folder = './tutorials/datasets/vehicle_data/validationset'
# steer_controller.loadData(name='validationset', source=data_folder, format=data_struct, skiplines=1)

# # Neural network train
# optimizer_params = [{'params':'gravity','weight_decay': 0.1}]
# optimizer_defaults = {'weight_decay': 0.00001}
# training_params = {'num_of_epochs':200, 'val_batch_size':128, 'train_batch_size':128, 'lr':0.00003}
# steer_controller.trainModel(train_dataset='trainingset', validation_dataset='validationset', shuffle_data=True, \
#                             add_optimizer_params=optimizer_params, add_optimizer_defaults=optimizer_defaults, training_params=training_params,  
#                             prediction_samples=300, step=300)  
# # the internal state is reset after "prediction_samples".
# # "step" is the number of samples to skip between the windows. The default is 1 meaning the size of a batch. Otherwise if the step is equal to the prediction_samples, then the whole window size is skipped

# # Test on a new dataset
# samples_test_set = steer_controller.get_samples('validationset', window=50) 
# steer_controller.resetStates()  # reset the internal state
# out_nn_test_set  = steer_controller(samples_test_set)

# # by default, the next batch skips a full length of a batch
# # shuffle = True shuffles the order of the batches, so it's ok with the autoregression

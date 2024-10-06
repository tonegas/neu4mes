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
import pandas as pd
import scipy as sp
clear = lambda: os.system('clear')
clear()
print(os.getcwd())
sys.path.append(os.getcwd())
from neu4mes import *

# Neural model inputs
curv  = Input('curv')
steer = Input('steer')

num_samples_future_curv = 15
num_samples_past_steer  = 15

# Known parameters:
data_folder = os.path.join(os.getcwd(),'tutorials','datasets','control_steer_car_parking')
# Import the file with the vehicle data
vehicle_data_csv = os.path.join(data_folder,'other_data','vehicle_data.csv')
L = pd.read_csv(vehicle_data_csv)['L'][0]

# Import the file with the steering maps
steer_maps_file = os.path.join(data_folder,'other_data','steer_map.txt')
steer_map_load  = np.loadtxt(steer_maps_file, delimiter='\t', skiprows=1)
delta_w_avg_map = torch.tensor(np.float64(np.deg2rad(steer_map_load[:,0])))  # [rad] average steering angle at the front wheels
delta_sw_map    = torch.tensor(np.float64(np.deg2rad(steer_map_load[:,1])))  # [rad] steering wheel angle

# Initial guesses
# Load the initial guesses for the curvature diagram approximation, computed in Matlab with a 2nd order optimizer
initial_guesses_curv_diagr = sp.io.loadmat(os.path.join(data_folder,'other_data','initial_guesses_curv_diagram','fit_curv_diagr_5th_ord_poly.mat'))
curv_diagram_params_guess = initial_guesses_curv_diagr['optim_params_poly'][0].astype('float64')
h_1_guess = Parameter('h_1',values=[[curv_diagram_params_guess[0]]])  # initial guess
h_2_guess = Parameter('h_2',values=[[curv_diagram_params_guess[1]]])  # initial guess
h_3_guess = Parameter('h_3',values=[[curv_diagram_params_guess[2]]])  # initial guess

# Create neural network relations:
# Curvature diagram
def curvat_diagram(curv,h_1,h_2,h_3):
  L = 2.6   # [m] wheelbase
  return torch.atan(h_1*curv + h_2*torch.pow(curv,3) + h_3*torch.pow(curv,5) + curv*L)

# Steering maps
def steer_map_spline(x, x_data, y_data):
  # Inputs: 
  # x: average steering angle at the front wheels [rad]
  # x_data: map of average steering angles at the front wheels (delta_w_avg_map) [rad]
  # y_data: map of steering wheel angles (delta_sw_map) [rad]
  # Output:
  # y: steering wheel angle [rad]

  # Linear interpolation of the steering map:
  # Find the indices of the intervals containing each x
  indices = torch.searchsorted(x_data, x, right=True).clamp(1, len(x_data) - 1)
  
  # Get the values for the intervals
  x1 = x_data[indices - 1]
  x2 = x_data[indices]
  y1 = y_data[indices - 1]
  y2 = y_data[indices]
  
  # Linear interpolation formula
  y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
  
  # Saturate the output if x is out of bounds
  y = torch.where(x < x_data[0], y_data[0], y)    # Saturate to minimum y_data
  y = torch.where(x > x_data[-1], y_data[-1], y)  # Saturate to maximum y_data
  return y
  
out_curv_diagr = ParamFun(curvat_diagram,parameters=[h_1_guess,h_2_guess,h_3_guess])(curv.sw([0,num_samples_future_curv]))
out_fir        = Fir(parameter_init=init_negexp, parameter_init_params={'size_index':0, 'first_value':0.1, 'lambda':5})(out_curv_diagr)   
# out_steer_map  = ParamFun(steer_map_spline)(out_fir)     
out_nn         = out_fir

# Create neural network output
out = Output('steering_angle', out_nn)

# Create neu4mes structure
steer_controller_park = Neu4mes(visualizer=MPLVisulizer(),seed=0)

# Add the neural model to the neu4mes structure and neuralization of the model
steer_controller_park.addModel('steer_ctrl',[out])
steer_controller_park.addMinimize('steer_error', 
                                  steer.last(),
                                  out, 
                                  loss_function='rmse')
steer_controller_park.neuralizeModel()

# Load the training and the validation dataset
data_struct = ['curv','steer']
data_folder_train = os.path.join(data_folder,'training')
data_folder_valid = os.path.join(data_folder,'validation')
data_folder_test  = os.path.join(data_folder,'test')
steer_controller_park.loadData(name='training_set', source=data_folder_train, format=data_struct, skiplines=1)
steer_controller_park.loadData(name='validation_set', source=data_folder_valid, format=data_struct, skiplines=1)
steer_controller_park.loadData(name='test_set', source=data_folder_test, format=data_struct, skiplines=1)

# Neural network train
training_pars = {'num_of_epochs':2000, 
                 'val_batch_size':100, 
                 'train_batch_size':100, 
                 'lr':0.001  # learning rate
                 }
steer_controller_park.trainModel(train_dataset='training_set', validation_dataset='validation_set', shuffle_data=False,
                                 training_params=training_pars, optimizer='Adam', prediction_samples=300, step=1)  
# NOTE: the internal state is reset after "prediction_samples".
# NOTE: "step" is the number of samples to skip when going to a new window. The default is 1, meaning the size of a batch. 
# Otherwise if the "step" is equal to "prediction_samples", then the whole window size is skipped

# # Test on a new dataset
# samples_test_set = steer_controller_park.get_samples('validationset', window=50) 
# steer_controller_park.resetStates()  # reset the internal state
# out_nn_test_set  = steer_controller_park(samples_test_set)

# NOTE: by default, the next batch skips a full length of a batch
# NOTE: shuffle = True shuffles only the order of the batches, so it's ok with the autoregression

print('All done!')
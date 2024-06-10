'''
 _     _____ _   _  _____  ___  ______      ___  _____ _____ _____ _   _  ___ _____ _____ _____ _   _    ______ _   _ _   _ _____ _____ _____ _____ _   _  _____ 
| |   |_   _| \ | ||  ___|/ _ \ | ___ \    / _ \/  __ \_   _|_   _| | | |/ _ \_   _|_   _|  _  | \ | |   |  ___| | | | \ | /  __ \_   _|_   _|  _  | \ | |/  ___|
| |     | | |  \| || |__ / /_\ \| |_/ /   / /_\ \ /  \/ | |   | | | | | / /_\ \| |   | | | | | |  \| |   | |_  | | | |  \| | /  \/ | |   | | | | | |  \| |\ `--. 
| |     | | | . ` ||  __||  _  ||    /    |  _  | |     | |   | | | | | |  _  || |   | | | | | | . ` |   |  _| | | | | . ` | |     | |   | | | | | | . ` | `--. \
| |_____| |_| |\  || |___| | | || |\ \    | | | | \__/\ | |  _| |_\ \_/ / | | || |  _| |_\ \_/ / |\  |   | |   | |_| | |\  | \__/\ | |  _| |_\ \_/ / |\  |/\__/ /
\_____/\___/\_| \_/\____/\_| |_/\_| \_|   \_| |_/\____/ \_/  \___/ \___/\_| |_/\_/  \___/ \___/\_| \_/   \_|    \___/\_| \_/\____/ \_/  \___/ \___/\_| \_/\____/ 

'''                                                                                                                                                                 

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Definition of the mono-dimensional (1D) linear activation function
# -------------------------------------------------------
def triangular(x, idx_channel, chan_centers):

  # Compute the number of channels
  num_channels = len(chan_centers)

  # First dimension of activation
  if idx_channel == 0:
    if num_channels != 1:
      ampl    = chan_centers[1] - chan_centers[0]
      act_fcn = np.minimum(
        np.maximum(
          -(x - chan_centers[0])/ampl + 1, 0), 1)
    else:
      # In case the user only wants one channel
      act_fcn = 1
  elif idx_channel != 0 and idx_channel == (num_channels - 1):
    ampl    = chan_centers[-1] - chan_centers[-2]
    act_fcn = np.minimum(
      np.maximum(
        (x - chan_centers[-2])/ampl, 0), 1)
  else:
    ampl_1  = chan_centers[idx_channel] - chan_centers[idx_channel - 1]
    ampl_2  = chan_centers[idx_channel + 1] - chan_centers[idx_channel]
    act_fcn = np.minimum(
      np.maximum(
        (x - chan_centers[idx_channel - 1])/ampl_1, 0),
      np.maximum(
        -(x - chan_centers[idx_channel])/ampl_2 + 1, 0))
  
  return act_fcn

def rectangular(x, idx_channel, chan_centers):
  ## compute number of channels
  num_channels = len(chan_centers)

  ## First dimension of activation
  if idx_channel == 0:
    if num_channels != 1:
      width = (chan_centers[idx_channel+1] - chan_centers[idx_channel]) / 2
      act_fcn = np.where(x < (chan_centers[idx_channel] + width), 1.0, 0.0)
    else:
      # In case the user only wants one channel
      act_fcn = 1
  elif idx_channel != 0 and idx_channel == (num_channels - 1):
    width = (chan_centers[idx_channel] - chan_centers[idx_channel-1]) / 2
    act_fcn = np.where(x >= (chan_centers[idx_channel] - width), 1.0, 0.0)
  else:
    width_forward = (chan_centers[idx_channel+1] - chan_centers[idx_channel]) / 2  
    width_backward = (chan_centers[idx_channel] - chan_centers[idx_channel-1]) / 2
    act_fcn = np.where((x >= (chan_centers[idx_channel] - width_backward)) & (x < (chan_centers[idx_channel] + width_forward)), 1.0, 0.0)
  
  return act_fcn

# -------------------------------------------------------
# Testing the mono-dimensional (1D) linear activation function
# -------------------------------------------------------

# Array of the independent variable
x_test = np.linspace(-5.0,7.0,num=1000) 

# Array of the channel centers
chan_centers = np.array([1.0, 3.0, 4.5, 7.0, 10.0])

# Plot the activation functions
fig = plt.figure(figsize=(10,5))
fig.suptitle('Activation functions')
ax = plt.subplot()
plt.grid(True)

for i in range(len(chan_centers)):
  ax.axvline(x=chan_centers[i], color='r', linestyle='--')
  activ_fun = rectangular(x_test, i, chan_centers)
  ax.plot(x_test,activ_fun,linewidth=3,label='Channel '+str(i+1))
ax.legend()

plt.show()

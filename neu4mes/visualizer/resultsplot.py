import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import json
import numpy as np
import signal

# Buffer to hold the data points

#data = {"performance": self.n4m.performance[key],


line = sys.stdin.readline().strip()
key, A, B, sample_time = None, None, None, None
if line:
    try:
        # Convert to float and append to buffer
        data_point = json.loads(line)
        name_data = data_point['name_data']
        key = data_point['key']
        A = data_point['prediction_A']
        B = data_point['prediction_B']
        sample_time = data_point['sample_time']
    except ValueError:
        exit()

fig, ax = plt.subplots()
# Clear the current plot
plt.clf()
# Plot data
plt.title(f'{name_data} Data of {key}')
A_t = np.transpose(np.array(A))
B_t = np.transpose(np.array(B))
for ind_win in range(A_t.shape[0]):
    for ind_dim in range(A_t.shape[1]):
        plt.plot(np.arange(0,len(A_t[ind_win,ind_dim])*sample_time,sample_time), A_t[ind_win,ind_dim], label=f'real')
        plt.plot(np.arange(0,len(B_t[ind_win,ind_dim])*sample_time,sample_time), B_t[ind_win,ind_dim], '-.', label=f'prediction')

plt.grid(True)
plt.legend(loc='best')
plt.xlabel('Time (s)')
plt.ylabel('Value')
# Set plot limits
# min_val = min([min(A), min(B)])
# max_val = max([max(A), max(B)])
# plt.ylim(min_val - min_val / 10, max_val + max_val / 10)
plt.show()

    # # Plot
    # self.fig, self.ax = self.plt.subplots(2*len(output_keys), 2,
    #                                 gridspec_kw={'width_ratios': [5, 1], 'height_ratios': [2, 1]*len(output_keys)})
    # if len(self.ax.shape) == 1:
    #     self.ax = np.expand_dims(self.ax, axis=0)
    # #plotsamples = self.prediction.shape[1]s
    # plotsamples = 200
    # for i in range(0, neu4mes.prediction.shape[0]):
    #     # Zoomed test data
    #     self.ax[2*i,0].plot(neu4mes.prediction[i], linestyle='dashed')
    #     self.ax[2*i,0].plot(neu4mes.label[i])
    #     self.ax[2*i,0].grid('on')
    #     self.ax[2*i,0].set_xlim((performance['max_se_idxs'][i]-plotsamples, performance['max_se_idxs'][i]+plotsamples))
    #     self.ax[2*i,0].vlines(performance['max_se_idxs'][i], neu4mes.prediction[i][performance['max_se_idxs'][i]], neu4mes.label[i][performance['max_se_idxs'][i]],
    #                             colors='r', linestyles='dashed')
    #     self.ax[2*i,0].legend(['predicted', 'test'], prop={'family':'serif'})
    #     self.ax[2*i,0].set_title(output_keys[i], family='serif')
    #     # Statitics
    #     self.ax[2*i,1].axis("off")
    #     self.ax[2*i,1].invert_yaxis()
    #     if performance:
    #         text = "Rmse test: {:3.6f}\nFVU: {:3.6f}".format(#\nAIC: {:3.6f}
    #             neu4mes.performance['rmse_test'][i],
    #             #neu4mes.performance['aic'][i],
    #             neu4mes.performance['fvu'][i])
    #         self.ax[2*i,1].text(0, 0, text, family='serif', verticalalignment='top')
    #     # test data
    #     self.ax[2*i+1,0].plot(neu4mes.prediction[i], linestyle='dashed')
    #     self.ax[2*i+1,0].plot(neu4mes.label[i])
    #     self.ax[2*i+1,0].grid('on')
    #     self.ax[2*i+1,0].legend(['predicted', 'test'], prop={'family':'serif'})
    #     self.ax[2*i+1,0].set_title(output_keys[i], family='serif')
    #     # Empty
    #     self.ax[2*i+1,1].axis("off")
    # self.fig.tight_layout()
    # self.plt.show()

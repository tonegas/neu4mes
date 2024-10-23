import numpy as np

import matplotlib.colors as mcolors

def plot_training(ax, title, key, data_train, data_val = None, last = None):
    # Plot data
    if last is not None:
        ax.set_title(f'{title} - epochs last {last}')
    else:
        ax.set_title(f'{title}')

    ax.plot([i + 1 for i in range(len(data_train))], data_train, label=f'Train loss {key}')
    if data_val:
        ax.plot([i + 1 for i in range(len(data_val))], data_val, '-.', label=f'Validation loss {key}')

    ax.set_yscale('log')
    ax.grid(True)
    ax.legend(loc='best')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    # Set plot limits
    if data_val:
        min_val = min([min(data_val), min(data_train)])
        max_val = max([max(data_val), max(data_train)])
    else:
        min_val = min(data_train)
        max_val = max(data_train)
    ax.set_ylim(min_val - min_val / 10, max_val + max_val / 10)


def plot_results(ax, name_data, key, A, B, sample_time):
    # Plot data
    ax.set_title(f'{name_data} Data of {key}')
    A_t = np.transpose(np.array(A))
    B_t = np.transpose(np.array(B))
    for ind_win in range(A_t.shape[0]):
        for ind_dim in range(A_t.shape[1]):
            ax.plot(np.arange(0, len(A_t[ind_win, ind_dim]) * sample_time, sample_time), A_t[ind_win, ind_dim],
                    label=f'real')
            ax.plot(np.arange(0, len(B_t[ind_win, ind_dim]) * sample_time, sample_time), B_t[ind_win, ind_dim], '-.',
                    label=f'prediction')
            correlation = np.corrcoef(A_t[ind_win, ind_dim],B_t[ind_win, ind_dim])[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=ax.transAxes, verticalalignment='top')

    ax.grid(True)
    ax.legend(loc='best')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'Value {key}')

    # min_val = min([min(A), min(B)])
    # max_val = max([max(A), max(B)])
    # plt.ylim(min_val - min_val / 10, max_val + max_val / 10)

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


def plot_fuzzy(ax, name, x, y, chan_centers):
    tableau_colors = mcolors.TABLEAU_COLORS
    num_of_colors = len(list(tableau_colors.keys()))
    for ind in range(len(y)):
        ax.axvline(x=chan_centers[ind], color=tableau_colors[list(tableau_colors.keys())[ind % num_of_colors]],
                   linestyle='--')
        ax.plot(x, y[ind], label=f'Channel {int(ind) + 1}', linewidth=2)
    ax.legend(loc='best')
    ax.set_xlabel('Input')
    ax.set_ylabel('Value')
    ax.set_title(f'Function {name}')


def plot_3d_function(plt, name, x0, x1, params, output, input_names):
    fig = plt.figure()
    # Clear the current plot
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(np.array(x0), np.array(x1), np.array(output), cmap='viridis')
    ax.set_xlabel(input_names[0])
    ax.set_ylabel(input_names[1])
    ax.set_zlabel(f'{name} output')
    for ind in range(len(input_names) - 2):
        fig.text(0.01, 0.9 - 0.05 * ind, f"{input_names[ind + 2]} ={params[ind]}", fontsize=10, color='blue',
                 style='italic')
    plt.title(f'Function {name}')

def plot_2d_function(plt, name, x, params, output, input_names):
    fig = plt.figure()
    # Clear the current plot
    plt.clf()
    plt.plot(np.array(x), np.array(output), linewidth=2)
    plt.xlabel(input_names[0])
    plt.ylabel(f'{name} output')
    for ind in range(len(input_names) - 1):
        fig.text(0.01, 0.9 - 0.05 * ind, f"{input_names[ind + 1]} ={params[ind]}", fontsize=10, color='blue',
                 style='italic')
    plt.title(f'Function {name}')
import numpy as np
import matplotlib.pyplot as plt

class Visualizer():
    def showResults(self, neu4mes, output_keys, performance = None):
        pass


class StandardVisualizer(Visualizer):
    def showResults(self, neu4mes, output_keys, performance = None):
        # Plot
        self.fig, self.ax = plt.subplots(2*len(output_keys), 2,
                                        gridspec_kw={'width_ratios': [5, 1], 'height_ratios': [2, 1]*len(output_keys)})
        if len(self.ax.shape) == 1:
            self.ax = np.expand_dims(self.ax, axis=0)
        #plotsamples = self.prediction.shape[1]s
        plotsamples = 200
        for i in range(0, neu4mes.prediction.shape[0]):
            # Zoomed test data
            self.ax[2*i,0].plot(neu4mes.prediction[i].flatten(), linestyle='dashed')
            self.ax[2*i,0].plot(neu4mes.inout_4test[output_keys[i]].flatten())
            self.ax[2*i,0].grid('on')
            self.ax[2*i,0].set_xlim((performance['max_se_idxs'][i]-plotsamples, performance['max_se_idxs'][i]+plotsamples))
            self.ax[2*i,0].vlines(performance['max_se_idxs'][i], neu4mes.prediction[i][performance['max_se_idxs'][i]], neu4mes.inout_4test[output_keys[i]][performance['max_se_idxs'][i]],
                                    colors='r', linestyles='dashed')
            self.ax[2*i,0].legend(['predicted', 'test'], prop={'family':'serif'})
            self.ax[2*i,0].set_title(output_keys[i], family='serif')
            # Statitics
            self.ax[2*i,1].axis("off")
            self.ax[2*i,1].invert_yaxis()
            if performance:
                text = "Rmse training: {:3.6f}\nRmse test: {:3.6f}\nAIC: {:3.6f}\nFVU: {:3.6f}".format(neu4mes.performance['rmse_train'][i], neu4mes.performance['rmse_test'][i], neu4mes.performance['aic'][i], neu4mes.performance['fvu'][i])
                self.ax[2*i,1].text(0, 0, text, family='serif', verticalalignment='top')
            # test data
            self.ax[2*i+1,0].plot(neu4mes.prediction[i].flatten(), linestyle='dashed')
            self.ax[2*i+1,0].plot(neu4mes.inout_4test[output_keys[i]].flatten())
            self.ax[2*i+1,0].grid('on')
            self.ax[2*i+1,0].legend(['predicted', 'test'], prop={'family':'serif'})
            self.ax[2*i+1,0].set_title(output_keys[i], family='serif')
            # Empty
            self.ax[2*i+1,1].axis("off")
        self.fig.tight_layout()
        plt.show()

    def showRecurrentResults(self, neu4mes, output_keys, performance = None):
        # Plot
        self.fig, self.ax = plt.subplots(2*len(output_keys), 2, gridspec_kw={'width_ratios': [5, 1], 'height_ratios': [2, 1]*len(output_keys)})
        if len(self.ax.shape) == 1:
            self.ax = np.expand_dims(self.ax, axis=0)

        #plotsamples = self.prediction.shape[1]
        plotsamples = 200
        for i in range(0, len(output_keys)):
            # Zoomed test data
            self.ax[2*i,0].plot(neu4mes.rnn_output[i].flatten(), linestyle='dashed')
            self.ax[2*i,0].plot(neu4mes.inout_4test[output_keys[i]][neu4mes.idx_of_rows[neu4mes.first_idx_test]-neu4mes.num_of_training_sample:].flatten())
            self.ax[2*i,0].grid('on')
            # self.ax[2*i,0].set_xlim((self.max_se_idxs[i]-plotsamples, self.max_se_idxs[i]+plotsamples))
            # self.ax[2*i,0].vlines(self.max_se_idxs[i], self.output_rnn[i][self.max_se_idxs[i]], self.inout_4test[key[i]][self.max_se_idxs[i]], colors='r', linestyles='dashed')
            self.ax[2*i,0].legend(['predicted', 'test'], prop={'family':'serif'})
            self.ax[2*i,0].set_title(output_keys[i], family='serif')
            # Statitics
            self.ax[2*i,1].axis("off")
            self.ax[2*i,1].invert_yaxis()
            # text = "Rmse: {:3.4f}".format(pred_rmse[i])
            # self.ax[2*i,1].text(0, 0, text, family='serif', verticalalignment='top')
            # test data
            self.ax[2*i+1,0].plot(neu4mes.rnn_output[i].flatten(), linestyle='dashed')
            self.ax[2*i+1,0].plot(neu4mes.inout_4test[output_keys[i]][neu4mes.idx_of_rows[neu4mes.first_idx_test]-neu4mes.num_of_training_sample:-1].flatten())
            self.ax[2*i+1,0].grid('on')
            self.ax[2*i+1,0].legend(['predicted', 'test'], prop={'family':'serif'})
            self.ax[2*i+1,0].set_title(output_keys[i], family='serif')
            # Empty
            self.ax[2*i+1,1].axis("off")
        self.fig.tight_layout()
        plt.show()

    #def __updateSlider(val, i):
    #    pos = self.spos.val
    #    plotsamples = self.prediction.shape[1]
    #    for i in range(self.prediction.shape[0]):
    #        self.ax[i].axis([pos-plotsamples/10,pos+plotsamples/10,1])
    #    self.fig.canvas.draw_idle()
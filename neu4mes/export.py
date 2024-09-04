import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
import numpy as np

def generate_training_report(train_loss, val_loss, y_true, y_pred, output_file='training_report.pdf'):
    # Create loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='train loss')
    if val_loss:
        plt.plot(val_loss, label='validation loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    loss_plot_buffer = io.BytesIO()
    plt.savefig(loss_plot_buffer, format='png')
    loss_plot_buffer.seek(0)
    plt.close()

    # Create predicted vs real values plot
    plt.figure(figsize=(10, 5))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title('Predicted vs Real Values')
    plt.xlabel('Real Values')
    plt.ylabel('Predicted Values')
    pred_real_plot_buffer = io.BytesIO()
    plt.savefig(pred_real_plot_buffer, format='png')
    pred_real_plot_buffer.seek(0)
    plt.close()

    # Create PDF
    c = canvas.Canvas(output_file, pagesize=letter)
    width, height = letter

    # Add loss plot
    c.drawString(100, height - 50, "Training Loss")
    c.drawImage(ImageReader(loss_plot_buffer), 50, height - 350, width=500, height=250)

    # Add predicted vs real values plot
    c.drawString(100, height - 400, "Predicted vs Real Values")
    c.drawImage(ImageReader(pred_real_plot_buffer), 50, height - 700, width=500, height=250)

    c.save()

    print(f"Training report saved as {output_file}")


def triangular(x, idx_channel, chan_centers):
    # Compute the number of channels
    num_channels = len(chan_centers)
    # First dimension of activation
    if idx_channel == 0:
        if num_channels != 1:
            ampl    = chan_centers[1] - chan_centers[0]
            act_fcn = np.minimum(np.maximum(-(x - chan_centers[0])/ampl + 1, 0), 1)
        else:
            # In case the user only wants one channel
            act_fcn = 1
    elif idx_channel != 0 and idx_channel == (num_channels - 1):
        ampl    = chan_centers[-1] - chan_centers[-2]
        act_fcn = np.minimum(np.maximum((x - chan_centers[-2])/ampl, 0), 1)
    else:
        ampl_1  = chan_centers[idx_channel] - chan_centers[idx_channel - 1]
        ampl_2  = chan_centers[idx_channel + 1] - chan_centers[idx_channel]
        act_fcn = np.minimum(np.maximum((x - chan_centers[idx_channel - 1])/ampl_1, 0),np.maximum(-(x - chan_centers[idx_channel])/ampl_2 + 1, 0))
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

def custom_function(func, x, idx_channel, chan_centers):
    act_fcn = func(x-chan_centers[idx_channel])
    return act_fcn

# -------------------------------------------------------
# Testing the mono-dimensional (1D) linear activation function
# -------------------------------------------------------
def plot_fuzzify(params):
    import matplotlib.pyplot as plt

    if params['functions'] != 'Rectangular' and params['functions'] != 'Triangular':
        if isinstance(params['names'], list):
            n_func = len(params['names'])
            for func in params['functions']:
                try:
                    exec(func, globals())
                except Exception as e:
                    print(f"An error occurred: {e}")
        else:
            n_func = 1
            try:
                exec(params['functions'], globals())
            except Exception as e:
                print(f"An error occurred: {e}")

    # Array of the independent variable
    x_test = np.linspace(params['centers'][0] - 2, params['centers'][-1] + 2, num=1000) 
    # Array of the channel centers
    chan_centers = np.array(params['centers'])
    # Plot the activation functions
    fig = plt.figure(figsize=(10,5))
    fig.suptitle('Activation functions')
    ax = plt.subplot()
    plt.grid(True)
    for i in range(len(chan_centers)):
        ax.axvline(x=chan_centers[i], color='r', linestyle='--')
        if params['functions'] == 'Triangular':
            activ_fun = triangular(x_test, i, chan_centers)
        elif params['functions'] == 'Rectangular':
            activ_fun = rectangular(x_test, i, chan_centers)
        else:
            if isinstance(params['names'], list):
                if i >= n_func:
                    func_idx = i - round(n_func * (i // n_func))
                else:
                    func_idx = i
                function_to_call = globals()[params['names'][func_idx]]
            else:
                function_to_call = globals()[params['names']]
            activ_fun = custom_function(function_to_call, x_test, i, chan_centers)

        #activ_fun = custom_function(fun3,x_test, i, chan_centers)
        ax.plot(x_test,activ_fun,linewidth=3,label='Channel '+str(i+1))
    ax.legend()
    return fig
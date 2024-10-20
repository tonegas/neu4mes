from neu4mes.fuzzify import triangular, rectangular, custom_function

import matplotlib.pyplot as plt
import io
import numpy as np
import os

import torch


def generate_training_report(train_loss, val_loss, y_true, y_pred, output_file='training_report.pdf'):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
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
    x_test = torch.from_numpy(x_test)
    # Array of the channel centers
    chan_centers = np.array(params['centers'])
    chan_centers = torch.from_numpy(chan_centers)
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

def import_onnx(self, onnx_path, data):
    import onnxruntime
    # Load the ONNX model
    session = onnxruntime.InferenceSession(onnx_path)
    # Get input and output names
    input_names = [item.name for item in session.get_inputs()]
    output_names = [item.name for item in session.get_outputs()]
    #input_name = session.get_inputs()#[0].name
    #output_name = session.get_outputs()[0].name

    print('input_name: ', input_names)
    print('output_name: ', output_names)

    # Run inference
    result = session.run([output_names], {input_names: data})
    # Print the result
    print(result)

def ExportReport(self, data, train_loss, val_loss):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    file_name = "report.pdf"
    # Combine the folder path and file name to form the complete file path
    file_path = os.path.join(self.folder_path, file_name)

    # Create PDF
    c = canvas.Canvas(file_path, pagesize=letter)
    width, height = letter

    with torch.inference_mode():
        out, minimize_out = self.model(data)

    for key, value in self.model_def['Minimizers'].items():
        # Create loss plot
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss[key], label='train loss')
        if val_loss:
            plt.plot(val_loss[key], label='validation loss')
        plt.title(f'{key} Error Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        loss_plot_buffer = io.BytesIO()
        plt.savefig(loss_plot_buffer, format='png')
        loss_plot_buffer.seek(0)
        plt.close()

        # Add loss plot
        c.drawString(50, height - 20, f'{key} Report')
        c.drawImage(ImageReader(loss_plot_buffer), 70, height - 270, width=500, height=250)

        # Convert tensors to numpy arrays
        name_a, name_b = value['A'].name, value['B'].name
        if isinstance(minimize_out[value['A'].name], torch.Tensor):
            y_pred = minimize_out[value['A'].name].squeeze().squeeze().detach().cpu().numpy()
        if isinstance(minimize_out[value['B'].name], torch.Tensor):
            y_true = minimize_out[value['B'].name].squeeze().squeeze().detach().cpu().numpy()
        # Create the scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        # Plot the perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        # Customize the plot
        plt.title(f"Predicted({name_a}) vs Real Values({name_b})")
        # Add a text box with correlation coefficient
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes,
                 verticalalignment='top')
        pred_real_plot_buffer = io.BytesIO()
        plt.savefig(pred_real_plot_buffer, format='png')
        pred_real_plot_buffer.seek(0)
        plt.close()

        # Add predicted vs real values plot
        c.drawImage(ImageReader(pred_real_plot_buffer), 70, height - 520, width=500, height=250)

        # Create the scatter plot
        plt.figure(figsize=(10, 6))
        plt.plot(y_pred, label=name_a)
        plt.plot(y_true, label=name_b)
        # Customize the plot
        plt.title(f"{key}: Predicted({name_a}) vs Real Values({name_b})")
        plt.xlabel("Samples")
        plt.ylabel("Values")
        plt.legend()
        plot_buffer = io.BytesIO()
        plt.savefig(plot_buffer, format='png')
        plot_buffer.seek(0)
        plt.close()

        # Add predicted vs real values plot
        c.drawImage(ImageReader(plot_buffer), 70, height - 770, width=500, height=250)
        c.showPage()

    for name, params in self.model_def['Functions'].items():
        if 'Fuzzify' in name:
            fig = plot_fuzzify(params=params)
            fuzzy_buffer = io.BytesIO()
            fig.savefig(fuzzy_buffer, format='png')
            fuzzy_buffer.seek(0)

            c.drawString(100, height - 50, f"fuzzy function : {name}")
            c.drawImage(ImageReader(fuzzy_buffer), 50, height - 350, width=500, height=250)

            c.showPage()

    c.save()
    self.visualizer.warning(f"Training report saved as {file_name}")


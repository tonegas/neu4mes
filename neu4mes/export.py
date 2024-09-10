import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
import numpy as np
import os

import torch
from torch.fx import symbolic_trace

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

def model_to_python(model_def, model, folder_path):
    ## create the python file path
    file_name = 'tracer_model.py'
    file_path = os.path.join(folder_path, file_name)
    # Get the symbolic tracer
    with torch.no_grad():
        trace = symbolic_trace(model)
    attributes = [line for line in trace.code.split() if 'self.' in line]
    saved_functions = []

    with open(file_path, 'w') as file:
        file.write("import torch.nn as nn\n")
        file.write("import torch\n\n")

        for name in model_def['Functions'].keys():
            if 'Fuzzify' in name:
                if 'slicing' not in saved_functions:
                    file.write("def neu4mes_fuzzify_slicing(res, i, x):\n")
                    file.write("    res[:, :, i:i+1] = x\n\n")
                    saved_functions.append('slicing')
                
                function_name = model_def['Functions'][name]['names']
                function_code = model_def['Functions'][name]['functions']
                if isinstance(function_code, list):
                    for i, fun_code in enumerate(function_code):
                        if fun_code != 'Rectangular' and fun_code != 'Triangular':
                            if function_name[i] not in saved_functions:
                                fun_code = fun_code.replace(f'def {function_name[i]}', f'def neu4mes_fuzzify_{function_name[i]}')
                                file.write(fun_code)
                                file.write("\n")
                                saved_functions.append(function_name[i])
                else:
                    if (function_name != 'Rectangular') and (function_name != 'Triangular') and (function_name not in saved_functions):
                        function_code = function_code.replace(f'def {function_name}', f'def neu4mes_fuzzify_{function_name}')
                        file.write(function_code)
                        file.write("\n")
                        saved_functions.append(function_name)
                

            elif 'ParamFun' in name:
                function_name = model_def['Functions'][name]['name']
                #torch.fx.wrap(self.model_def['Functions'][name]['name'])
                if function_name not in saved_functions:
                    code = model_def['Functions'][name]['code']
                    code = code.replace(f'def {function_name}', f'def neu4mes_parametricfunction_{function_name}')
                    file.write(code)
                    file.write("\n")
                    saved_functions.append(function_name)

        file.write("class TracerModel(torch.nn.Module):\n")
        file.write("    def __init__(self):\n")
        file.write("        super().__init__()\n")
        file.write("        self.all_parameters = {}\n")
        for attr in attributes:
            if 'constant' in attr:
                file.write(f"        {attr} = torch.tensor({getattr(trace, attr.replace('self.', ''))})\n")
            elif 'relation_forward' in attr:
                key = attr.split('.')[2]
                if 'Fir' in key or 'Linear' in key:
                    param = model_def['Relations'][key][2] if 'weights' in attr.split('.')[3] else model_def['Relations'][key][3]
                    value = model.all_parameters[param].data.squeeze(0) if 'Linear' in key else model.all_parameters[param].data
                    file.write(f"        self.all_parameters[\"{param}\"] = torch.nn.Parameter(torch.{value}, requires_grad=True)\n")
            elif 'all_parameters' in attr:
                key = attr.split('.')[-1]
                file.write(f"        self.all_parameters[\"{key}\"] = torch.nn.Parameter(torch.{model.all_parameters[key].data}, requires_grad=True)\n")
        
        file.write("        self.all_parameters = torch.nn.ParameterDict(self.all_parameters)\n")
        for line in trace.code.split("\n")[len(saved_functions)+1:]:
            if 'self.relation_forward' in line:
                attribute = line.split()[-1]
                relation = attribute.split('.')[2]
                relation_type = attribute.split('.')[3]
                param = model_def['Relations'][relation][2] if 'weights' == relation_type else model_def['Relations'][relation][3]
                new_attribute = f'self.all_parameters.{param}'
                file.write(f"    {line.replace(attribute, new_attribute)}\n")
            else:
                file.write(f"    {line}\n")
    return file_path

def model_to_python_onnx(model_def, tracer_path):
    # Define the mapping dictionary
    trace_mapping = {}
    forward = 'def forward(self,'
    for key in model_def['Inputs'].keys():
        value = f'kwargs[\'{key}\']'
        trace_mapping[value] = key
        forward = forward + f' {key},'
    forward = forward + '):'
    # Open and read the file
    with open(tracer_path, 'r') as file:
        file_content = file.read()
    # Replace the forward header
    file_content = file_content.replace('def forward(self, kwargs):', forward)
    # Perform the substitution
    for key, value in trace_mapping.items():
        file_content = file_content.replace(key, value)
    # Write the modified content back to a new file
    onnx_path = tracer_path.replace('.py','_onnx.py')
    with open(onnx_path, 'w') as file:
        file.write(file_content)
    return onnx_path

def model_to_onnx(model, model_def, input_n_samples, file_path):
    dummy_inputs = []
    input_names = []
    for key, item in model_def['Inputs'].items():
        input_names.append(key)
        window_size = input_n_samples[key]
        dummy_inputs.append(torch.randn(size=(1, window_size, item['dim'])))
    output_names = [name for name in model_def['Outputs'].keys()]
    dummy_inputs = tuple(dummy_inputs)

    onnx_path = file_path.replace('.py','.onnx')
    torch.onnx.export(
                model,                            # The model to be exported
                dummy_inputs,                          # Tuple of inputs to match the forward signature
                onnx_path,                             # File path to save the ONNX model
                export_params=True,                    # Store the trained parameters in the model file
                opset_version=12,                      # ONNX version to export to (you can use 11 or higher)
                do_constant_folding=True,              # Optimize constant folding for inference
                input_names=input_names,               # Name each input as they will appear in ONNX
                output_names=output_names,             # Name the output
                #dynamic_axes={
                #                'input1': {0: 'batch_size'},       # Dynamic batch size for input1
                #                'input2': {0: 'batch_size'},       # Dynamic batch size for input2
                #                'output': {0: 'batch_size'}        # Dynamic batch size for the output
                #            }
                )
    return onnx_path
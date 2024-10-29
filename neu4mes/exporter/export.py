import sys, os, torch, importlib
from collections import OrderedDict

from torch.fx import symbolic_trace

from pprint import PrettyPrinter

class JsonPrettyPrinter(PrettyPrinter):
    def _format(self, object, *args):
        if isinstance(object, str):
            width = self._width
            self._width = sys.maxsize
            try:
                super()._format(object.replace('\'','_"_'), *args)
            finally:
                self._width = width
        else:
            super()._format(object, *args)

def save_model(model, model_path):
    # Export the dictionary as a JSON file
    with open(model_path, 'w') as json_file:
        # json.dump(self.model_def, json_file, indent=4)
        json_file.write(JsonPrettyPrinter().pformat(model)
                        .replace('\'', '\"')
                        .replace('_"_', '\'')
                        .replace('None', 'null')
                        .replace('False', 'false')
                        .replace('True', 'true'))
        # json_file.write(JsonPrettyPrinter().pformat(model).replace('None','null'))
        # data = json.dumps(self.model_def)
        # json_file.write(pformat(data).replace('\\\\n', '\\n').replace('\'', '').replace('(','').replace(')',''))
        # json_file.write(pformat(data).replace('\'', '\"'))

def load_model(model_path):
    import json
    with open(model_path, 'r', encoding='UTF-8') as file:
        model_def = json.load(file)
    return model_def

def export_python_model(model_def, model, model_path):
    # Get the symbolic tracer
    with torch.no_grad():
        trace = symbolic_trace(model)
    attributes = sorted(set([line for line in trace.code.split() if 'self.' in line]))
    saved_functions = []

    with open(model_path, 'w') as file:
        #file.write("import torch.nn as nn\n")
        file.write("import torch\n\n")

        for name in model_def['Functions'].keys():
            if 'Fuzzify' in name:
                if 'slicing' not in saved_functions:
                    #file.write("@torch.fx.wrap\n")
                    file.write("def neu4mes_fuzzify_slicing(res, i, x):\n")
                    file.write("    res[:, :, i:i+1] = x\n\n")
                    saved_functions.append('slicing')

                function_name = model_def['Functions'][name]['names']
                function_code = model_def['Functions'][name]['functions']
                if isinstance(function_code, list):
                    for i, fun_code in enumerate(function_code):
                        if fun_code != 'Rectangular' and fun_code != 'Triangular':
                            if function_name[i] not in saved_functions:
                                fun_code = fun_code.replace(f'def {function_name[i]}',
                                                            f'def neu4mes_fuzzify_{function_name[i]}')
                                #file.write("@torch.fx.wrap\n")
                                file.write(fun_code)
                                file.write("\n")
                                saved_functions.append(function_name[i])
                else:
                    if (function_name != 'Rectangular') and (function_name != 'Triangular') and (
                            function_name not in saved_functions):
                        function_code = function_code.replace(f'def {function_name}',
                                                              f'def neu4mes_fuzzify_{function_name}')
                        #file.write("@torch.fx.wrap\n")
                        file.write(function_code)
                        file.write("\n")
                        saved_functions.append(function_name)


            elif 'ParamFun' in name:
                function_name = model_def['Functions'][name]['name']
                # torch.fx.wrap(self.model_def['Functions'][name]['name'])
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
        file.write("        self.all_constants = {}\n")
        for attr in attributes:
            if 'all_constant' in attr:
                key = attr.split('.')[-1]
                file.write(
                    f"        self.all_constants[\"{key}\"] = torch.tensor({model.all_constants[key].tolist()})\n")
                #file.write(f"        {attr} = torch.tensor({getattr(trace, attr.replace('self.', ''))})\n")
            elif 'relation_forward' in attr:
                key = attr.split('.')[2]
                if 'Fir' in key or 'Linear' in key:
                    if 'weights' in attr.split('.')[3]:
                        param = model_def['Relations'][key][2]
                        value = model.all_parameters[param] #.squeeze(0) if 'Linear' in key else model.all_parameters[param]
                        file.write(
                            f"        self.all_parameters[\"{param}\"] = torch.nn.Parameter(torch.tensor({value.tolist()}), requires_grad=True)\n")
                    elif 'bias' in attr.split('.')[3]:
                        param = model_def['Relations'][key][3]
                        # value = model.all_parameters[param].data.squeeze(0) if 'Linear' in key else model.all_parameters[param].data
                        # value = model.all_parameters[param].data
                        file.write(
                            f"        self.all_parameters[\"{param}\"] = torch.nn.Parameter(torch.tensor({model.all_parameters[param].tolist()}), requires_grad=True)\n")
                    elif 'dropout' in attr.split('.')[3]:
                        param = model_def['Relations'][key][4]
                        file.write(f"        self.{key} = torch.nn.Dropout(p={param})\n")
                    # param = model_def['Relations'][key][2] if 'weights' in attr.split('.')[3] else model_def['Relations'][key][3]
                    # value = model.all_parameters[param].data.squeeze(0) if 'Linear' in key else model.all_parameters[param].data
                    # file.write(f"        self.all_parameters[\"{param}\"] = torch.nn.Parameter(torch.{value}, requires_grad=True)\n")
            elif 'all_parameters' in attr:
                key = attr.split('.')[-1]
                file.write(
                    f"        self.all_parameters[\"{key}\"] = torch.nn.Parameter(torch.tensor({model.all_parameters[key].tolist()}), requires_grad=True)\n")

        file.write("        self.all_parameters = torch.nn.ParameterDict(self.all_parameters)\n")
        file.write("        self.all_constants = torch.nn.ParameterDict(self.all_constants)\n")
        file.write("    def init_states(self, state_model, connect = {}, reset_states = False):\n")
        file.write("        pass\n")
        file.write("    def reset_connect_variables(self, connect, values = None, only = True):\n")
        file.write("        pass\n")
        file.write("    def reset_states(self, values = None, only = True):\n")
        file.write("        pass\n")

        for line in trace.code.split("\n")[len(saved_functions) + 1:]:
            if 'self.relation_forward' in line:
                if 'dropout' in line:
                    attribute = line.split()[0]
                    layer = attribute.split('_')[2].capitalize()
                    old_line = f"self.relation_forward.{layer}.dropout"
                    new_line = f"self.{layer}"
                    file.write(f"    {line.replace(old_line, new_line)}\n")
                else:
                    attribute = line.split()[-1]
                    relation = attribute.split('.')[2]
                    relation_type = attribute.split('.')[3]
                    param = model_def['Relations'][relation][2] if 'weights' == relation_type else \
                    model_def['Relations'][relation][3]
                    new_attribute = f'self.all_parameters.{param}'
                    file.write(f"    {line.replace(attribute, new_attribute)}\n")
            else:
                file.write(f"    {line}\n")

def export_pythononnx_model(input_order, outputs_order, model_path, model_onnx_path):
    # Define the mapping dictionary input
    trace_mapping_input = {}
    forward = 'def forward(self,'
    for i, key in enumerate(input_order):
        value = f'kwargs[\'{key}\']'
        trace_mapping_input[value] = key
        forward = forward + f' {key}' + (',' if i < len(input_order) - 1 else '')
    forward = forward + '):'
    # Define the mapping dictionary output
    outputs = '        return ('
    for i, key in enumerate(outputs_order):
        outputs += f'outputs[0][\'{key}\']' + (',' if i < len(outputs_order) - 1 else ')')

    # Open and read the file
    with open(model_path, 'r') as file:
        file_content = file.read()
    # Replace the forward header
    file_content = file_content.replace('def forward(self, kwargs):', forward)
    # Perform the substitution
    for key, value in trace_mapping_input.items():
        file_content = file_content.replace(key, value)
    # Write the modified content back to a new file
    # Replace the return statement
    # Trova l'ultima occorrenza di 'return'
    last_return_index = file_content.rfind('return')
    # Se 'return' è trovato, sostituiscilo con 'outputs ='
    if last_return_index != -1:
        file_content = file_content[:last_return_index] + 'outputs =' + file_content[last_return_index + len('return'):]
    file_content += outputs
    with open(model_onnx_path, 'w') as file:
        file.write(file_content)

def import_python_model(name, model_folder):
    sys.path.insert(0, model_folder)
    module_name = os.path.basename(name)
    if module_name in sys.modules:
        # Reload the module if it is already loaded
        module = importlib.reload(sys.modules[module_name])
    else:
        # Import the module if it is not loaded
        module = importlib.import_module(module_name)
    return module.TracerModel()

def export_onnx_model(model_def, model, input_order, output_order, model_path):
    dummy_inputs = []
    input_names = []
    dynamic_axes = {}
    for key in input_order:
        input_names.append(key)
        window_size = model_def['Inputs'][key]['ntot']
        dummy_inputs.append(torch.randn(size=(1, window_size, model_def['Inputs'][key]['dim'])))
        dynamic_axes[key] = {0: 'batch_size'}
    output_names = output_order
    dummy_inputs = tuple(dummy_inputs)

    torch.onnx.export(
        model,                            # The model to be exported
        dummy_inputs,                          # Tuple of inputs to match the forward signature
        model_path,                             # File path to save the ONNX model
        export_params = True,                    # Store the trained parameters in the model file
        opset_version = 12,                      # ONNX version to export to (you can use 11 or higher)
        do_constant_folding=True,              # Optimize constant folding for inference
        input_names = input_names,               # Name each input as they will appear in ONNX
        output_names = output_names,             # Name the output
        dynamic_axes = dynamic_axes
    )

def import_onnx_model(name, model_folder):
    import onnxruntime as ort
    model_path = os.path.join(model_folder, name + '.onnx')
    return ort.InferenceSession(model_path)

    # import numpy as np
    # ort_sess = ort.InferenceSession("./TODO/mauro/net.onnx")
    # prova = {'Vo': np.array([[20.0]], dtype=np.float32),
    #          'VL': np.array([[[50.0], [50.0], [50.2], [30.1], [15.0]]], dtype=np.float32)}
    # # ortvalue = ort.OrtValue.ortvalue_from_numpy(prova)
    # outputs = ort_sess.run(None, prova)
    # # Print Result
    # print(outputs)
    #
    # session = onnxruntime.InferenceSession(onnx_path)
    # # Get input and output names
    # input_names = [item.name for item in session.get_inputs()]
    # output_names = [item.name for item in session.get_outputs()]
    # # input_name = session.get_inputs()#[0].name
    # # output_name = session.get_outputs()[0].name
    #
    # print('input_name: ', input_names)
    # print('output_name: ', output_names)
    #
    # # Run inference
    # result = session.run([output_names], {input_names: data})
    # # Print the result
    # print(result)
    #
    # import onnx


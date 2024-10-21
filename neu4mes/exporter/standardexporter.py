import os, sys, torch
from datetime import datetime

from neu4mes.exporter.exporter import Exporter
from neu4mes.exporter.export import save_model, load_model, export_python_model, export_pythononnx_model, export_onnx_model, import_python_model, import_onnx_model
from neu4mes.utils import check

class StandardExporter(Exporter):
    def __init__(self, workspace=None, save_history=False):
        # Export parameters
        if workspace is not None:
            self.workspace = workspace
            os.makedirs(self.workspace, exist_ok=True)
            if save_history:
                self.folder = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                self.workspace_folder = os.path.join(self.workspace, self.folder)
            else:
                self.workspace_folder = self.workspace
            os.makedirs(self.workspace_folder, exist_ok=True)

    def getWorkspace(self):
        return self.workspace_folder

    def saveTorchModel(self, name = 'net', model_folder = None): #TODO, model = None)
        check(self.n4m.neuralized == True, RuntimeError, 'The model is not neuralized yet!')
        file_name = name + ".pt"
        model_path = os.path.join(self.workspace_folder, file_name) if model_folder is None else os.path.join(model_folder,file_name)
        torch.save(self.n4m.model.state_dict(), model_path)
        self.n4m.visualizer.saveModel('Torch Model', model_path)

    def loadTorchModel(self, name = 'net', model_folder = None): #TODO, model = None):
        check(self.n4m.neuralized == True, RuntimeError, 'The model is not neuralized yet!')
        file_name = name + ".pt"
        model_path = os.path.join(self.workspace_folder, file_name) if model_folder is None else os.path.join(model_folder,file_name)
        self.n4m.model.load_state_dict(torch.load(model_path))
        self.n4m.visualizer.loadModel('Torch Model',model_path)

    def saveModel(self, model_def, name = 'net', model_folder = None):
        # Combine the folder path and file name to form the complete file path
        model_folder = self.workspace_folder if model_folder is None else model_folder
        # Specify the JSON file name
        file_name = name + ".json"
        # Combine the folder path and file name to form the complete file path
        model_path = os.path.join(model_folder, file_name)
        save_model(model_def, model_path)
        self.n4m.visualizer.saveModel('JSON Model', model_path)

    def loadModel(self, name = 'net', model_folder = None):
        # Combine the folder path and file name to form the complete file path
        model_folder = self.workspace_folder if model_folder is None else model_folder
        model_def = None
        try:
            file_name = name + ".json"
            model_path = os.path.join(model_folder, file_name)
            model_def = load_model(model_path)
            self.n4m.visualizer.loadModel('JSON Model', model_path)
        except Exception as e:
            self.n4m.visualizer.warning(f"The file {model_path} it is not found or not conformed.\n Error: {e}")
        return model_def

    def exportPythonModel(self, name = 'net', model_folder = None):
        check(self.n4m.traced == False, RuntimeError, 'The model is traced and cannot be exported to Python.\n Run neuralizeModel() to recreate a standard model.')
        check(self.n4m.neuralized == True, RuntimeError, 'The model is not neuralized yet!')
        file_name = name + ".py"
        model_path = os.path.join(self.workspace_folder, file_name) if model_folder is None else os.path.join(model_folder, file_name)
        ## Export to python file
        export_python_model(self.n4m.model_def, self.n4m.model, model_path)
        self.n4m.visualizer.exportModel('Python Torch Model', model_path)

    def importPythonModel(self, name = 'net', model_folder = None):
        try:
            model_folder = self.workspace_folder if model_folder is None else model_folder
            model = import_python_model(name, model_folder)
            self.n4m.visualizer.importModel('Python Torch Model', os.path.join(model_folder,name+'.py'))
        except Exception as e:
            self.n4m.visualizer.warning(f"The module {name} it is not found in the folder {model_folder}.\nError: {e}")
        return model

    def exportONNX(self, inputs_order, outputs_order, name = 'net', model_folder = None):
        check(self.n4m.traced == False, RuntimeError, 'The model is traced and cannot be exported to ONNX.\n Run neuralizeModel() to recreate a standard model.')
        check(self.n4m.neuralized == True, RuntimeError, 'The model is not neuralized yet!')
        check(set(inputs_order) == set(self.n4m.model_def['Inputs'].keys()), ValueError,
              f'The inputs are not the same as the model inputs ({self.n4m.model_def["Inputs"].keys()}).')
        check(set(outputs_order) == set(self.n4m.model_def['Outputs'].keys()), ValueError,
              f'The outputs are not the same as the model outputs ({self.n4m.model_def["Outputs"].keys()}).')
        file_name = name + ".py"
        model_folder = self.workspace_folder if model_folder is None else model_folder
        model_folder = os.path.join(model_folder, 'onnx')
        os.makedirs(model_folder, exist_ok=True)
        model_path = os.path.join(model_folder, file_name)
        onnx_python_model_path = model_path.replace('.py', '_onnx.py')
        onnx_model_path = model_path.replace('.py', '.onnx')
        ## Export to python file (onnx compatible)
        export_python_model(self.n4m.model_def, self.n4m.model, model_path)
        self.n4m.visualizer.exportModel('Python Torch Model', model_path)
        export_pythononnx_model(inputs_order, outputs_order, model_path, onnx_python_model_path)
        self.n4m.visualizer.exportModel('Python Onnx Torch Model', onnx_python_model_path)
        ## Export to onnx file (onnx compatible)
        model = import_python_model(file_name.replace('.py', '_onnx'), model_folder)
        export_onnx_model(inputs_order, outputs_order, self.n4m.model_def, model, self.n4m.input_n_samples, onnx_model_path)
        self.n4m.visualizer.exportModel('Onnx Model', onnx_model_path)

    def importONNX(self, name = 'net', model_folder = None):
        try:
            model_folder = self.workspace_folder if model_folder is None else model_folder
            model = import_onnx_model(name, model_folder)
            self.n4m.visualizer.importModel('Onnx Model', os.path.join(model_folder,name+'.py'))
        except Exception as e:
            self.n4m.visualizer.warning(f"The module {name} it is not found in the folder {model_folder}.\nError: {e}")
        return model

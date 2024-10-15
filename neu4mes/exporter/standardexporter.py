import os, sys, torch
from datetime import datetime

from neu4mes.exporter.exporter import Exporter
from neu4mes.exporter.export import save_model, load_model, export_python_model, export_pythononnx_model, export_onnx_model
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
        check(self.n4m.neuralized == True, RuntimeError, 'The model is not neuralized yet!')
        file_name = name + ".py"
        model_path = os.path.join(self.workspace_folder, file_name) if model_folder is None else os.path.join(model_folder, file_name)
        ## Export to python file
        export_python_model(self.n4m.model_def, self.n4m.model, model_path)
        self.n4m.visualizer.exportModel('Python Torch Model', model_path)

    def importPythonModel(self, name = 'net', model_folder = None):
        try:
            model_folder = self.workspace_folder if model_folder is None else model_folder
            sys.path.insert(0, model_folder)
            module_name = os.path.basename(name)
            module = __import__(module_name)
            model = module.TracerModel()
            self.n4m.visualizer.importModel('Python Torch Model', os.path.join(model_folder,module_name+'.py'))
        except Exception as e:
            self.n4m.visualizer.warning(f"The module {name} it is not found in the folder {module_name}.\nError: {e}")
        return model

    def exportOnnxModel(self, name = 'net', model_folder = None):
        check(self.n4m.neuralized == True, RuntimeError, 'The model is not neuralized yet!')
        file_name = name + ".py"
        model_path = os.path.join(self.workspace_folder, file_name) if model_folder is None else os.path.join(model_folder, file_name)
        onnx_python_model_path = model_path.replace('.py', '.onnx.py')
        onnx_model_path = model_path.replace('.onnx.py', '.onnx')
        ## Export to python file (onnx compatible)
        export_python_model(self.n4m.model_def, self.n4m.model, model_path)
        export_pythononnx_model(self.n4m.model_def, model_path, onnx_python_model_path)
        self.n4m.visualizer.exportModel('Python Onnx Torch Model', onnx_python_model_path)
        ## Export to onnx file (onnx compatible)
        export_onnx_model(self.n4m.model_def, self.n4m.model, self.n4m.input_n_samples, onnx_model_path)
        self.n4m.visualizer.exportModel('Onnx Model', onnx_python_model_path)


        # ## Export to python file
        # python_path = model_to_python(self.model_def, self.model, folder_path=self.folder_path)
        # ## Export to python file (onnx compatible)
        # python_onnx_path = model_to_python_onnx(self.model_def, tracer_path=python_path)
        # ## Export to onnx file
        # self.importTracer(python_onnx_path)
        # self.model.eval()
        # onnx_path = model_to_onnx(self.model, self.model_def, self.input_n_samples, python_path)
        #
        # self.visualizer.warning(f"The pytorch model has been exported to {self.folder}.")
        # return python_path, python_onnx_path, onnx_path

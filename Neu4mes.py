import Linear
import Sum
import Relation

import os, os.path
from pprint import pp, pprint
import numpy as np


from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import tensorflow.keras.layers #import Layer, Dense, Add, Lambda, RNN
from tensorflow.python.training.tracking import data_structures

def rmse(y_true, y_pred):
    # Root mean squared error (rmse) for regression
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

class Neu4mes:
    def __init__(self, model_def = 0):
        self.model_def = model_def
        #Keras structs
        self.relation_types = {
            'Linear': Linear.Linear,
            'Sum': Sum.Sum
        }
        self.input_time_window = {}
        self.input_n_samples = {}
        self.max_n_samples = 0
        self.inputs = {}
        self.relations = {}
        self.output_relation = {}
        self.outputs = {}
        self.model = 0                      #Keras model
        #data structs
        self.input_data = {}
        self.inout_data_time_window = {}
        self.inout_asarray = {}
        #training
        self.batch_size = 128
        self.inout_4train = {}
        self.inout_4validation = {}
        self.learning_rate = 0.001
        self.num_of_epochs = 200

    def addModel(self, model_def):
        if type(model_def) is Relation.Relation:
            self.model_def = model_def.json
        elif type(model_def) is dict:
            self.model_def = model_def

    def neuralizeModel(self):
        reletions = self.model_def['Relations']
        for outel in self.model_def['Outputs']:
            relel = reletions.get(outel)
            for reltype, relvalue in relel.items():
                relaction = self.relation_types.get(reltype)
                if relaction:
                    relaction().setInput(self, relvalue)
                else:
                    print("Relation not defined")
                    
        for key,val in self.model_def['Inputs'].items():
            time_window = self.input_time_window[key]
            self.input_n_samples[key] = int(time_window/self.model_def['SampleTime'])
            if self.input_n_samples[key] > self.max_n_samples:
                self.max_n_samples = self.input_n_samples[key]
            self.inputs[key] = tensorflow.keras.layers.Input(shape = (self.input_n_samples[key], ), batch_size = None, name = val['Name'])
        
        for outel in self.model_def['Outputs']:
            relel = reletions.get(outel)
            self.output_relation[outel] = []
            for reltype, relvalue in relel.items():
                relaction = self.relation_types.get(reltype)
                if relaction:
                    relaction().createElem(self,relvalue,outel)
                else:
                    print("Relation not defined")           
            self.outputs[outel] = tensorflow.keras.layers.Add(name = outel)([self.relations[o+outel] for o in self.output_relation[outel]])
        
        #print([val for key,val in self.inputs.items()])
        #print([val for key,val in self.outputs.items()])
        self.model = Model(inputs = [val for key,val in self.inputs.items()], outputs=[val for key,val in self.outputs.items()])
        print(self.model.summary())

    def loadData(self, format, folder = './data', skiplines = 0):
        path, dirs, files = next(os.walk(folder))
        file_count = len(files)
        
        for key in format+list(self.output_relation.keys()):
            self.inout_data_time_window[key] = []
                
        for file in files:
            # Read data file
            for data in format: 
                self.input_data[(file,data)] = []

            all_lines = open(folder+file, 'r')
            lines = all_lines.readlines()[skiplines:] # skip first lines to avoid NaNs

            for line in range(0, len(lines)):
                splitline = lines[line].rstrip("\n").split(";")
                for idx, key in enumerate(format):
                    self.input_data[(file,key)].append(float(splitline[idx]))     

            if 'time' in format:
                for i in range(0, len(self.input_data[(file,'time')])-self.max_n_samples):
                    self.inout_data_time_window['time'].append(self.input_data[(file,'time')][i+self.max_n_samples-1])

            for key in self.input_n_samples.keys():
                for i in range(0, len(self.input_data[(file,key)])-self.max_n_samples):
                    if self.input_n_samples[key] == 1:
                        self.inout_data_time_window[key].append(self.input_data[(file,key)][i+self.max_n_samples-1])
                    else:
                        self.inout_data_time_window[key].append(self.input_data[(file,key)][i+self.max_n_samples-self.input_n_samples[key]:i+self.max_n_samples])
        
            for key in self.output_relation.keys():
                used_key = key
                elem_key = key.split('_')
                if len(elem_key) > 1 and elem_key[1]== 'z':
                    used_key = elem_key[0]
                for i in range(0, len(self.input_data[(file,used_key)])-self.max_n_samples):
                    self.inout_data_time_window[key].append(self.input_data[(file,used_key)][i+self.max_n_samples])
        
        for key,data in self.inout_data_time_window.items():
            self.inout_asarray[key]  = np.asarray(data)

    def trainModel(self, validation_percentage = 0):
        # Divide train and test samples
        num_of_sample = len(list(self.inout_asarray.values())[0])
        validation = round(validation_percentage*num_of_sample/100)
        train  = num_of_sample-validation

        if train < self.batch_size or validation < self.batch_size:
            batch = 1
        else:
            # Samples must be multiplier of batch
            train = int(train/self.batch_size) * self.batch_size
            validation  = num_of_sample-train
            validation = int(validation/self.batch_size) * self.batch_size

        for key,data in self.inout_asarray.items():
            if len(data.shape) == 1:
                self.inout_4train[key] = data[0:train]
                self.inout_4validation[key]  = data[train:train+validation]
            else:
                self.inout_4train[key] = data[0:train,:]
                self.inout_4validation[key]  = data[train:train+validation,:]                

        #print('Samples: ' + str(train+validation) + '/' + str(num_of_sample) + ' (' + str(train) + ' train + ' + str(validation) + ' validation)')
        #print('Batch: ' + str(self.batch_size))
        
        # Configure model for training
        self.opt = optimizers.Adam(learning_rate=self.learning_rate) #optimizers.Adam(learning_rate=l_rate) #optimizers.RMSprop(learning_rate=lrate, rho=0.4)
        self.model.compile(optimizer=self.opt, loss='mean_squared_error', metrics=[rmse])

        # Train model
        #print('[Fitting]')
        print(len([self.inout_4train[key] for key in self.model_def['Output'].keys()]))

        self.fit = self.model.fit([self.inout_4train[key] for key in self.model_def['Input'].keys()],
                            [self.inout_4train[key] for key in self.model_def['Output'].keys()],
                            epochs=self.num_of_epochs, batch_size=self.batch_size, verbose=1)

    def controlDefinition(control):
        pass

    def neuralizeControl():
        pass

    def trainControl(data):
        pass

    def exportModel(parmas):
        pass

    def exportControl(params):
        pass



import Neu4mes 
from Neu4mes.Output import Output
from Neu4mes.NeuObj import NeuObj, merge

from pprint import pprint 
import tensorflow.keras.layers
import tensorflow.keras.models
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import string

import random
def rand(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def rmse(y_true, y_pred):
    # Root mean squared error (rmse) for regression
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

class Neu4mes:
    def __init__(self, model_def = 0):
        self.model_used =  NeuObj().json
        if type(model_def) is Output:
            self.model_def = model_def.json
        elif type(model_def) is dict:
            self.model_def = self.model_def
        else:
            self.model_def = NeuObj().json

        self.elem = 0
        self.input_time_window = {}
        self.input_n_samples = {}
        self.max_n_samples = 0
        self.inputs = {}
        self.inputs_for_model = {}
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
        if type(model_def) is Output:
            self.model_def = merge(self.model_def, model_def.json)
        elif type(model_def) is dict:
            self.model_def = merge(self.model_def, model_def) 
        pprint(self.model_def)

    def neuralizeModel(self, sample_time = 0):
        if sample_time:
            self.model_def["SampleTime"] = sample_time
        relations = self.model_def['Relations']
        for outel in self.model_def['Outputs']:
            relel = relations.get(outel)
            for reltype, relvalue in relel.items():
                self.setInput(relvalue, outel)

        for key,val in self.model_def['Inputs'].items():
            time_window = self.input_time_window[key]
            input_n_sample_aux = int(time_window/self.model_def['SampleTime'])

            if input_n_sample_aux > self.max_n_samples:
                self.max_n_samples = input_n_sample_aux
            
            if self.input_n_samples.get(key):
                if input_n_sample_aux > self.input_n_samples[key]:
                    self.input_n_samples[key] = input_n_sample_aux
            else:
                self.input_n_samples[key] = input_n_sample_aux
            
            if key not in self.model_used['Inputs']:
                if 'Discrete' in val:
                    (self.inputs_for_model[key],self.inputs[key]) = self.discreteInput(key, self.input_n_samples[key], val['Discrete'])
                else: 
                    (self.inputs_for_model[key],self.inputs[key]) = self.input(key, self.input_n_samples[key])
                self.model_used['Inputs'][key]=val

        for outel in self.model_def['Outputs']:
            relel = relations.get(outel)
            self.output_relation[outel] = []
            for reltype, relvalue in relel.items():
                relation = getattr(self,reltype)
                if relation:
                    self.outputs[outel] = self.createElem(relation,relvalue,outel)
                else:
                    print("Relation not defined") 
            if outel not in self.model_used['Outputs']:
                self.model_used['Outputs'][outel]=self.model_def['Outputs'][outel]   
                self.model_used['Relations'][outel]=relel        
                #self.relations[outel]=self.outputs[outel]

        #pprint(self.model_used)
        # print([(key,val) for key,val in self.inputs_for_model.items()])
        # print([(key,val) for key,val in self.outputs.items()])
        # print([(key,val) for key,val in self.relations.items()])
        self.model = tensorflow.keras.models.Model(inputs = [val for key,val in self.inputs_for_model.items()], outputs=[val for key,val in self.outputs.items()])
        print(self.model.summary())

    def setInput(self, relvalue, outel):
        for el in relvalue:
            if type(el) is tuple:
                if el[0] in self.model_def['Inputs']:
                    time_window = self.input_time_window.get(el[0])
                    if time_window is not None:
                        if self.input_time_window[el[0]] < el[1]:
                            self.input_time_window[el[0]] = el[1]
                    else:
                        self.input_time_window[el[0]] = el[1]
                else:
                    raise Exception("A window on internal signal is not supported!")
            else: 
                if el in self.model_def['Inputs']:
                    time_window = self.input_time_window.get(el)
                    if time_window is None:
                        self.input_time_window[el] = self.model_def['SampleTime']
                else:
                    relel = self.model_def['Relations'].get((outel,el))
                    if relel is None:
                        relel = self.model_def['Relations'].get(el)
                        if relel is None:
                            raise Exception("Graph is not completed!")
                    for reltype, relvalue in relel.items():
                        self.setInput(relvalue, outel)

    def createRelation(self,relation,el,outel):
        relel = self.model_def['Relations'].get((outel,el))
        if relel is None:
            relel = self.model_def['Relations'].get(el)
            if relel is None:
                raise Exception("Graph is not completed!")
        for new_reltype, new_relvalue in relel.items():
            new_relation = getattr(self,new_reltype)
            if new_relation:
                if el not in self.model_used['Relations']:
                    self.relations[el] = self.createElem(new_relation, new_relvalue, outel)
                    self.model_used['Relations'][el]=relel
                return self.relations[el]
            else:
                print("Relation not defined")    

    def createElem(self, relation, relvalue, outel):
        self.elem = self.elem + 1
        if len(relvalue) == 1:
            el = relvalue[0]
            if type(el) is tuple:
                if el[0] in self.model_def['Inputs']:
                    samples = int(el[1]/self.model_def['SampleTime'])
                    if (el[0],samples) not in self.inputs:
                        self.inputs[(el[0],samples)] = self.part(el[0],self.inputs[el[0]],samples)
                    return relation(outel[:2]+'_'+el[0][:2]+str(self.elem), self.inputs[(el[0],samples)])                
                else:
                    print("Tuple is defined only for Input")   
            else:
                if el in self.model_def['Inputs']:
                    if (el,1) not in self.inputs:
                        self.inputs[(el,1)] = self.part(el,self.inputs[el],1)
                    return relation(outel[:2]+'_'+el[:2]+str(self.elem), self.inputs[(el,1)])
                else:
                    input = self.createRelation(relation, el, outel)
                    return relation(outel[:2]+'_'+el[:2]+str(self.elem), input)
        else:
            inputs = []
            name = outel[:2]
            for idx, el in enumerate(relvalue):
                if type(el) is tuple:
                    if el[0] in self.model_def['Inputs']:
                        samples = int(el[1]/self.model_def['SampleTime'])
                        if (el[0],samples) not in self.inputs:
                            self.inputs[(el[0],samples)] = self.part(el[0],self.inputs[el[0]],samples)
                        inputs.append(self.inputs[(el[0],samples)])
                        name = name +'_'+el[0][:2]
                    else:
                        print("Tuple is defined only for Input")  
                else:
                    if el in self.model_def['Inputs']:
                        if (el[0],1) not in self.inputs:
                            self.inputs[(el[0],1)] = self.part(el,self.inputs[el],1)
                        inputs.append(self.inputs[(el[0],1)])
                        name = name +'_'+ el[:2]
                    else:
                        inputs.append(self.createRelation(relation, el, outel))
                        name = name +'_'+ el[:2]
            
            return relation(name+str(self.elem), inputs)

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
                elem_key = key.split('__')
                if len(elem_key) > 1 and elem_key[1]== '-z1':
                    used_key = elem_key[0]
                for i in range(0, len(self.input_data[(file,used_key)])-self.max_n_samples):
                    self.inout_data_time_window[key].append(self.input_data[(file,used_key)][i+self.max_n_samples])
        
        for key,data in self.inout_data_time_window.items():
            self.inout_asarray[key]  = np.asarray(data)

    def trainModel(self, validation_percentage = 0, show_results = False):
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
        self.opt = optimizers.Adam(learning_rate = self.learning_rate) #optimizers.Adam(learning_rate=l_rate) #optimizers.RMSprop(learning_rate=lrate, rho=0.4)
        self.model.compile(optimizer = self.opt, loss = 'mean_squared_error', metrics=[rmse])

        # Train model
        #print('[Fitting]')
        #print(len([self.inout_4train[key] for key in self.model_def['Outputs'].keys()]))

        self.fit = self.model.fit([self.inout_4train[key] for key in self.model_def['Inputs'].keys()],
                                  [self.inout_4train[key] for key in self.model_def['Outputs'].keys()],
                                  epochs = self.num_of_epochs, batch_size = self.batch_size, verbose=1)

        if show_results:
            # Prediction on validation samples
            #print('[Prediction]')
            self.prediction = self.model.predict([self.inout_4validation[key] for key in self.model_def['Inputs'].keys()]) #, callbacks=[NoiseCallback()])
            self.prediction = np.array(self.prediction)
            if len(self.prediction.shape) == 2:
                self.prediction = np.expand_dims(self.prediction, axis=0)

            key = list(self.model_def['Outputs'].keys())

            # Rmse
            pred_rmse = []
            for i in range(len(key)):
                pred_rmse.append( np.sqrt(np.mean(np.square(self.prediction[i].flatten() - self.inout_4validation[key[i]].flatten()))) )
            pred_rmse = np.array(pred_rmse)

            # Square error   
            se = []
            for i in range(len(key)):
                se.append( np.square(self.prediction[i].flatten() - self.inout_4validation[key[i]].flatten()) )
            se = np.array(se)
            self.max_se_idxs = np.argmax(se, axis=1)

            # Plot
            self.fig, self.ax = plt.subplots(self.prediction.shape[0], 2, gridspec_kw={'width_ratios': [5, 1]})
            if len(self.ax.shape) == 1:
                self.ax = np.expand_dims(self.ax, axis=0)
            #plotsamples = self.prediction.shape[1]
            plotsamples = 200
            for i in range(0, self.prediction.shape[0]):
                # Data
                self.ax[i,0].plot(self.prediction[i].flatten(), linestyle='dashed')
                self.ax[i,0].plot(self.inout_4validation[key[i]].flatten())
                self.ax[i,0].grid('on')
                self.ax[i,0].set_xlim((self.max_se_idxs[i]-plotsamples, self.max_se_idxs[i]+plotsamples))
                self.ax[i,0].vlines(self.max_se_idxs[i], self.prediction[i][self.max_se_idxs[i]], self.inout_4validation[key[i]][self.max_se_idxs[i]], colors='r', linestyles='dashed')
                self.ax[i,0].legend(['predicted', 'validation'], prop={'family':'serif'})
                self.ax[i,0].set_title(key[i], family='serif')
                # Statitics
                self.ax[i,1].axis("off")
                self.ax[i,1].invert_yaxis()
                text = "Rmse: {:3.4f}".format(pred_rmse[i])
                self.ax[i,1].text(0, 0, text, family='serif', verticalalignment='top')
            #plt.subplots_adjust(left=0.125, bottom=0.25) # adjust the main plot to make room for the sliders
            #axpos = plt.axes([0.125, 0.15, 0.35, 0.02], facecolor='lightgoldenrodyellow')
            #self.spos = Slider(ax=axpos, label='', valmin=self.max_se_idxs[i]-plotsamples/10, valmax=self.max_se_idxs[i]+plotsamples/10, valinit=self.max_se_idxs[i], valstep=1)
            self.fig.tight_layout()
            plt.show()
       
    #def __updateSlider(val, i):
    #    pos = self.spos.val
    #    plotsamples = self.prediction.shape[1]
    #    for i in range(self.prediction.shape[0]):
    #        self.ax[i].axis([pos-plotsamples/10,pos+plotsamples/10,1])
    #    self.fig.canvas.draw_idle()

    def controlDefinition(control):
        pass

    def neuralizeControl():
        pass

    def trainControl(data):
        pass

    def exportModel(params):
        pass

    def exportControl(params):
        pass



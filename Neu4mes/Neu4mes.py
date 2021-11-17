from pprint import pprint 
import tensorflow.keras.layers
import tensorflow.keras.models
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import string

import neu4mes 
import random

def rand(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def rmse(y_true, y_pred):
    # Root mean squared error (rmse) for regression
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

class RNNCell(tensorflow.keras.layers.Layer):
    def __init__(self, model, input_is_state, states_size, inputs_size, **kwargs):
        super(RNNCell, self).__init__(**kwargs)
        self.model = model
        self.input_is_state = input_is_state
        self.inputs_size = inputs_size
        self.state_size = states_size

    def call(self, inputs, states):

        # print('-------------------------------------')
        # print(states)
        # print(len(states))
        # print(self.state_size)
        # print(inputs)
        # print(self.input_is_state)
        # print('-------------------------------------')
        
        network_inputs, inputs_idx, states_idx = [], 0 ,0
        for is_state in self.input_is_state:
            if is_state:
                network_inputs.append(states[states_idx])
                states_idx += 1
            else:
                network_inputs.append(inputs[inputs_idx])
                inputs_idx += 1
        
        output = self.model(network_inputs)
        if type(output) is not list:
            output = [output] 
        # print('-------------------------------------')
        # print(output)
        # print('-------------------------------------')
        
        new_states = []
        for idx in range(len(states)):
            new_states.append(tf.concat([states[idx][:,1:self.state_size[idx]], output[idx]], axis=1))

        out = K.concatenate(output)
        return out, (new_states)

    def get_config(self):
        pass

class Neu4mes:
    def __init__(self, model_def = 0):
        self.model_used =  neu4mes.NeuObj().json
        if type(model_def) is neu4mes.Output:
            self.model_def = model_def.json
        elif type(model_def) is dict:
            self.model_def = self.model_def
        else:
            self.model_def = neu4mes.NeuObj().json

        self.elem = 0
        self.idx_of_rows = [0]
        self.output_keys = []
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
        self.learning_rate = 0.0005
        self.num_of_epochs = 200
        #rnn
        self.rnn_model = None
        self.rnn_window = None
        self.inputs_for_rnn_model = {}
        self.inout_rnn_asarray = {}
        self.inout_rnn_data_time_window = {}
        self.rnn_inputs = {}
        self.init_rnn_state = {}
        self.learning_rate_rnn = self.learning_rate/10000
        self.num_of_epochs_rnn = 50
        self.rnn_opt = None
        self.inout_rnn_4train = {}
        self.inout_rnn_4validation = {}

    def addModel(self, model_def):
        if type(model_def) is neu4mes.Output:
            self.model_def = neu4mes.merge(self.model_def, model_def.json)
        elif type(model_def) is dict:
            self.model_def = neu4mes.merge(self.model_def, model_def) 
        #pprint(self.model_def)

    def neuralizeModel(self, sample_time = 0, prediction_window = None):
        if prediction_window is not None:
            self.rnn_window = round(prediction_window/sample_time)

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
                    (self.inputs_for_rnn_model[key],self.rnn_inputs[key]) = self.inputRNN(key, self.rnn_window, self.input_n_samples[key])

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
        print([(key,val) for key,val in self.inputs_for_model.items()])
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
                # print(outel[:2]+'_'+el[0][:2]+str(self.elem))
                if el[0] in self.model_def['Inputs']:
                    samples = int(el[1]/self.model_def['SampleTime'])
                    if (el[0],samples) not in self.inputs:
                        self.inputs[(el[0],samples)] = self.part(el[0],self.inputs[el[0]],samples)
                    return relation(outel[:2]+'_'+el[0][:2]+str(self.elem), self.inputs[(el[0],samples)])                
                else:
                    print("Tuple is defined only for Input")   
            else:
                # print(outel[:2]+'_'+el[:2]+str(self.elem))
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

            # print(name+str(self.elem))
            return relation(name+str(self.elem), inputs)

    def loadData(self, format, folder = './data', skiplines = 0, delimiters=['\t',';']):
        path, dirs, files = next(os.walk(folder))
        file_count = len(files)

        for idx,key in enumerate(self.output_relation.keys()):
            self.output_keys.append(key)
            elem_key = key.split('__')
            if len(elem_key) > 1 and elem_key[1]== '-z1':
                self.output_keys[idx] = elem_key[0]
            else:
                raise("Operation not implemeted yet!")

        for key in format+list(self.output_relation.keys()):
            self.inout_data_time_window[key] = []

        if self.rnn_window:
            for key in format+list(self.output_relation.keys()):
                self.inout_rnn_data_time_window[key] = []

        for file in files:
            # Read data file
            for data in format: 
                self.input_data[(file,data)] = []

            all_lines = open(folder+file, 'r')
            lines = all_lines.readlines()[skiplines:] # skip first lines to avoid NaNs

            for line in range(0, len(lines)):
                delimiter_string = '|'.join(delimiters)
                splitline = re.split(delimiter_string,lines[line].rstrip("\n"))
                for idx, key in enumerate(format):
                    try:
                        self.input_data[(file,key)].append(float(splitline[idx]))
                    except ValueError:
                        self.input_data[(file,key)].append(splitline[idx])  

            if self.rnn_window:
                if 'time' in format:
                    for i in range(0, len(self.input_data[(file,'time')])-self.max_n_samples-self.rnn_window):
                        inout_rnn = []
                        for j in range(i, i+self.rnn_window):
                            inout_rnn.append(self.input_data[(file,'time')][j:j+self.max_n_samples])

                        self.inout_rnn_data_time_window['time'].append(inout_rnn)
                
                for key in self.input_n_samples.keys():
                    for i in range(0, len(self.input_data[(file,key)])-self.max_n_samples-self.rnn_window):
                        inout_rnn = []
                        for j in range(i, i+self.rnn_window):
                            if self.input_n_samples[key] == 1:
                                inout_rnn.append(self.input_data[(file,key)][j+self.max_n_samples-1])
                            else:
                                inout_rnn.append(self.input_data[(file,key)][j+self.max_n_samples-self.input_n_samples[key]:j+self.max_n_samples])
                        self.inout_rnn_data_time_window[key].append(inout_rnn)

                for idx,key in enumerate(self.output_relation.keys()):
                    for i in range(0, len(self.input_data[(file,self.output_keys[idx])])-self.max_n_samples-self.rnn_window):
                        inout_rnn = []
                        for j in range(i, i+self.rnn_window):
                            inout_rnn.append(self.input_data[(file,self.output_keys[idx])][j+self.max_n_samples])
                        self.inout_rnn_data_time_window[key].append(inout_rnn)

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

            self.idx_of_rows.append(len(self.inout_data_time_window['time']))

        if self.rnn_window:
            for key,data in self.inout_rnn_data_time_window.items():
                self.inout_rnn_asarray[key]  = np.asarray(data)

        for key,data in self.inout_data_time_window.items():
            self.inout_asarray[key]  = np.asarray(data)

    def trainRecurrentModel(self, states, validation_percentage = 0, show_results = False):
        state_keys = [key.signal_name for key in states if type(key) is neu4mes.Output]
        states_size = [self.input_n_samples[key] for key in self.model_def['Inputs'].keys() if key in state_keys]
        inputs_size = [self.input_n_samples[key] for key in self.model_def['Inputs'].keys() if key not in state_keys]
        state_vector = [1 if key in state_keys else 0 for key in self.model_def['Inputs'].keys()]
        rnn_cell = RNNCell(self.model, state_vector, states_size, inputs_size)
        # print(self.inputs_for_rnn_model)
        # print(state_keys)
        # print(states_size)
        # print(inputs_size)

        for key in state_keys:
            self.init_rnn_state[key] = tensorflow.keras.layers.Lambda(lambda x: x[:,0,:], name=key+'_init_state')(self.rnn_inputs[key])
        # print(self.init_rnn_state)

        initial_state_rnn = [self.init_rnn_state[key] for key in state_keys]
        # print(initial_state_rnn)
        inputs = [self.rnn_inputs[key] for key in self.model_def['Inputs'].keys() if key not in state_keys]
        # print(inputs)
        out_x_rnn = tensorflow.keras.layers.RNN(rnn_cell, return_sequences=True, stateful=False, unroll=True, name='rnn')(tuple(inputs), initial_state=initial_state_rnn)
        # splited_out = tensorflow.keras.layers.Lambda(lambda tensor: tf.split(tensor, num_or_size_splits=len(states_size), axis = 2))(out_x_rnn)
        splited_out = []
        for idx in range(len(self.model_def['Outputs'])):
            splited_out.append(out_x_rnn[:,:,idx])

        self.rnn_model = tensorflow.keras.models.Model(inputs=[val for key,val in self.inputs_for_rnn_model.items()], outputs=splited_out)
        print(self.rnn_model.summary())

        self.rnn_opt = optimizers.Adam(learning_rate = self.learning_rate_rnn)
        self.rnn_model.compile(optimizer = self.opt, loss = 'mean_squared_error', metrics=[rmse])
        self.rnn_model.set_weights(self.net_weights)

        # Divide train and test samples
        num_of_sample = len(list(self.inout_rnn_asarray.values())[0])
        validation = round(validation_percentage*num_of_sample/100)
        train  = num_of_sample-validation

        if train < self.batch_size or validation < self.batch_size:
            batch = 1
        else:
            # Samples must be multiplier of batch
            train = int(train/self.batch_size) * self.batch_size
            validation  = num_of_sample-train
            validation = int(validation/self.batch_size) * self.batch_size

        for key,data in self.inout_rnn_asarray.items():
            if len(data.shape) == 1:
                self.inout_rnn_4train[key] = data[0:train]
                self.inout_rnn_4validation[key]  = data[train:train+validation]
            else:
                self.inout_rnn_4train[key] = data[0:train,:]
                self.inout_rnn_4validation[key]  = data[train:train+validation,:]

        self.fit = self.rnn_model.fit([self.inout_rnn_4train[key] for key in self.model_def['Inputs'].keys()],
                                        [self.inout_rnn_4train[key] for key in self.model_def['Outputs'].keys()],
                                        epochs = self.num_of_epochs_rnn, batch_size = self.batch_size, verbose=1)

        # print(self.model.get_weights())             
        # print(self.rnn_model.get_weights())  
        first_idx_validation = next(x[0] for x in enumerate(self.idx_of_rows) if x[1] > self.train_idx)
        # print(first_idx_validation)
        if show_results:
            print('[Prediction]')
            self.output_rnn = []
            for i in range(first_idx_validation, len(self.idx_of_rows)-1):
                first_idx = self.idx_of_rows[i]-self.train_idx
                last_idx = self.idx_of_rows[i+1]-self.train_idx
                # print(last_idx-first_idx)
                # print(len(self.inout_4validation[key][first_idx:-1]))
                # print(first_idx)
                # print(last_idx)
                # print(self.idx_of_rows[i]-self.train_idx)
                input = [np.expand_dims(self.inout_4validation[key][first_idx],axis = 0) for key in self.model_def['Inputs'].keys()]
                # print([self.idx_of_rows[i]+1-self.train_idx,self.idx_of_rows[i+1]-self.train_idx])
                for t in range(first_idx+1,last_idx+1): 
                    # print(t)
                    self.rnn_prediction = np.array(self.model(input)) #, callbacks=[NoiseCallback()])
                    # print(self.rnn_prediction)
                    if len(self.rnn_prediction.shape) == 2:
                        self.rnn_prediction = np.expand_dims(self.rnn_prediction, axis=0)

                    if t != last_idx:
                        for idx, key in enumerate(self.model_def['Inputs'].keys()):
                            # print('------------------------------')
                            if key in state_keys:
                                idx_out = self.output_keys.index(key)
                                input[idx] = np.append(input[idx][:,1:],self.rnn_prediction[idx_out], axis = 1)
                                
                                # print(self.rnn_prediction[idx_out])
                                # print(input[idx])
                                
                            else:
                                input[idx] = np.expand_dims(self.inout_4validation[key][t], axis = 0)  
                        # print('------------------------------')                
                    
                    self.output_rnn.append(self.rnn_prediction)
            
            key = list(self.model_def['Outputs'].keys())
            # print(state_keys)
            # print(self.output_rnn)
            self.output_rnn = np.asarray(self.output_rnn)
            # print(self.output_rnn.shape)
            # print(len(self.output_keys))
            self.output_rnn = np.transpose(self.output_rnn.reshape((-1,len(self.output_keys))))
            # print(self.output_rnn.shape)
            # print(self.output_rnn[0].shape)
            #print(self.output_rnn)
            # print(self.inout_4validation[key[0]].shape)
            # print(self.inout_4validation[key[0]][self.idx_of_rows[first_idx_validation]-self.train_idx:].shape)
            # print(self.inout_4validation[key[0]][self.idx_of_rows[first_idx_validation]-self.train_idx-1:-1].flatten().shape)

            

            # Plot
            self.fig, self.ax = plt.subplots(2*len(key), 2, gridspec_kw={'width_ratios': [5, 1], 'height_ratios': [2, 1]*len(key)})
            if len(self.ax.shape) == 1:
                self.ax = np.expand_dims(self.ax, axis=0)
            #plotsamples = self.prediction.shape[1]
            plotsamples = 200
            for i in range(0, len(key)):
                # Zoomed validation data
                self.ax[2*i,0].plot(self.output_rnn[i].flatten(), linestyle='dashed')
                self.ax[2*i,0].plot(self.inout_4validation[key[i]][self.idx_of_rows[first_idx_validation]-self.train_idx:].flatten())
                self.ax[2*i,0].grid('on')
                # self.ax[2*i,0].set_xlim((self.max_se_idxs[i]-plotsamples, self.max_se_idxs[i]+plotsamples))
                # self.ax[2*i,0].vlines(self.max_se_idxs[i], self.output_rnn[i][self.max_se_idxs[i]], self.inout_4validation[key[i]][self.max_se_idxs[i]], colors='r', linestyles='dashed')
                self.ax[2*i,0].legend(['predicted', 'validation'], prop={'family':'serif'})
                self.ax[2*i,0].set_title(key[i], family='serif')
                # Statitics
                self.ax[2*i,1].axis("off")
                self.ax[2*i,1].invert_yaxis()
                # text = "Rmse: {:3.4f}".format(pred_rmse[i])
                # self.ax[2*i,1].text(0, 0, text, family='serif', verticalalignment='top')
                # Validation data
                self.ax[2*i+1,0].plot(self.output_rnn[i].flatten(), linestyle='dashed')
                self.ax[2*i+1,0].plot(self.inout_4validation[key[i]][self.idx_of_rows[first_idx_validation]-self.train_idx:-1].flatten())
                self.ax[2*i+1,0].grid('on')
                self.ax[2*i+1,0].legend(['predicted', 'validation'], prop={'family':'serif'})
                self.ax[2*i+1,0].set_title(key[i], family='serif')
                # Empty
                self.ax[2*i+1,1].axis("off")
            self.fig.tight_layout()
            plt.show()


    def trainModel(self, states = None, validation_percentage = 0, show_results = False):
        # Divide train and test samples
        num_of_sample = len(list(self.inout_asarray.values())[0])
        validation = round(validation_percentage*num_of_sample/100)
        self.train_idx  = num_of_sample-validation

        if self.train_idx < self.batch_size or validation < self.batch_size:
            batch = 1
        else:
            # Samples must be multiplier of batch
            train = int(self.train_idx/self.batch_size) * self.batch_size
            validation  = num_of_sample-train
            validation = int(validation/self.batch_size) * self.batch_size

        for key,data in self.inout_asarray.items():
            if len(data.shape) == 1:
                self.inout_4train[key] = data[0:self.train_idx]
                self.inout_4validation[key]  = data[self.train_idx:self.train_idx+validation]
            else:
                self.inout_4train[key] = data[0:self.train_idx,:]
                self.inout_4validation[key]  = data[self.train_idx:self.train_idx+validation,:]                

        #print('Samples: ' + str(train+validation) + '/' + str(num_of_sample) + ' (' + str(train) + ' train + ' + str(validation) + ' validation)')
        #print('Batch: ' + str(self.batch_size))
        
        # Configure model for training
        self.opt = optimizers.Adam(learning_rate = self.learning_rate) #optimizers.Adam(learning_rate=l_rate) #optimizers.RMSprop(learning_rate=lrate, rho=0.4)
        self.model.compile(optimizer = self.opt, loss = 'mean_squared_error', metrics=[rmse])

        # print(self.inout_4train['x'].shape)
        # Train model
        #print('[Fitting]')
        #print(len([self.inout_4train[key] for key in self.model_def['Outputs'].keys()]))

        self.fit = self.model.fit([self.inout_4train[key] for key in self.model_def['Inputs'].keys()],
                                  [self.inout_4train[key] for key in self.model_def['Outputs'].keys()],
                                  epochs = self.num_of_epochs, batch_size = self.batch_size, verbose=1)
        
        self.net_weights = self.model.get_weights()

        if show_results:
            # Prediction on validation samples
            #print('[Prediction]')
            self.prediction = self.model([self.inout_4validation[key] for key in self.model_def['Inputs'].keys()]) #, callbacks=[NoiseCallback()])
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
            self.fig, self.ax = plt.subplots(2*self.prediction.shape[0], 2,
                                             gridspec_kw={'width_ratios': [5, 1], 'height_ratios': [2, 1]*self.prediction.shape[0]})
            if len(self.ax.shape) == 1:
                self.ax = np.expand_dims(self.ax, axis=0)
            #plotsamples = self.prediction.shape[1]
            plotsamples = 200
            for i in range(0, self.prediction.shape[0]):
                # Zoomed validation data
                self.ax[2*i,0].plot(self.prediction[i].flatten(), linestyle='dashed')
                self.ax[2*i,0].plot(self.inout_4validation[key[i]].flatten())
                self.ax[2*i,0].grid('on')
                self.ax[2*i,0].set_xlim((self.max_se_idxs[i]-plotsamples, self.max_se_idxs[i]+plotsamples))
                self.ax[2*i,0].vlines(self.max_se_idxs[i], self.prediction[i][self.max_se_idxs[i]], self.inout_4validation[key[i]][self.max_se_idxs[i]], colors='r', linestyles='dashed')
                self.ax[2*i,0].legend(['predicted', 'validation'], prop={'family':'serif'})
                self.ax[2*i,0].set_title(key[i], family='serif')
                # Statitics
                self.ax[2*i,1].axis("off")
                self.ax[2*i,1].invert_yaxis()
                text = "Rmse: {:3.4f}".format(pred_rmse[i])
                self.ax[2*i,1].text(0, 0, text, family='serif', verticalalignment='top')
                # Validation data
                self.ax[2*i+1,0].plot(self.prediction[i].flatten(), linestyle='dashed')
                self.ax[2*i+1,0].plot(self.inout_4validation[key[i]].flatten())
                self.ax[2*i+1,0].grid('on')
                self.ax[2*i+1,0].legend(['predicted', 'validation'], prop={'family':'serif'})
                self.ax[2*i+1,0].set_title(key[i], family='serif')
                # Empty
                self.ax[2*i+1,1].axis("off")
            self.fig.tight_layout()
            plt.show()
       
        if states is not None:
            self.trainRecurrentModel(states, validation_percentage, show_results)
    #def __updateSlider(val, i):
    #    pos = self.spos.val
    #    plotsamples = self.prediction.shape[1]
    #    for i in range(self.prediction.shape[0]):
    #        self.ax[i].axis([pos-plotsamples/10,pos+plotsamples/10,1])
    #    self.fig.canvas.draw_idle()

    # def controlDefinition(control):
    #     pass

    # def neuralizeControl():
    #     pass

    # def trainControl(data):
    #     pass

    # def exportModel(params):
    #     pass

    # def exportControl(params):
    #     pass



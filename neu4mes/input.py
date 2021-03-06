import neu4mes
import tensorflow.keras.layers
import tensorflow as tf
import numpy as np

class Input(neu4mes.NeuObj):
    def __init__(self,name,values = None):
        super().__init__()
        self.name = name
        self.json['Inputs'][self.name] = {}
        if values:
            self.values = values
            self.json['Inputs'][self.name] = {
                'Discrete' : values
            }

    def tw(self, tw, offset = None):
        if offset is not None:
            return self, tw, offset
        return self, tw
    
    def z(self, advance):
        if advance > 0:
            return self, '__+z'+str(advance)
        else:
            return self, '__-z'+str(-advance)

    def s(self, derivate):
        if derivate > 0:
            return self, '__+s'+str(derivate)
        else:
            return self, '__-s'+str(-derivate)

class ControlInput(Input):
    def __init__(self,name,values = None):
        super().__init__(name,values)

def createDiscreteInput(Neu4mes, name, size, types):
    input = tensorflow.keras.layers.Input(shape = (size, ), batch_size = None, name = name, dtype='int32')
    return (input,tensorflow.keras.layers.Lambda(lambda x: tf.one_hot(x[:,0]-types[0], len(set(np.asarray(types)))))(input))

def createDiscreteInputRNN(Neu4mes, name, window, size, types):
    input = tensorflow.keras.layers.Input(shape = (window, size, ), batch_size = None, name = name, dtype='int32')
    return (input,tensorflow.keras.layers.Lambda(lambda x: tf.one_hot(x[:,:,0]-types[0], len(set(np.asarray(types)))))(input))

def createInput(Neu4mes, name, size):
    input = tensorflow.keras.layers.Input(shape = (size, ), batch_size = None, name = name)
    return (input,input)

def createInputRNN(Neu4mes, name, window, size):
    input = tensorflow.keras.layers.Input(shape = (window, size, ), batch_size = None, name = name)
    return (input,input)

def createPart(Neu4mes, name, input, backward, forward = 0, offset = None):
    if Neu4mes.input_n_samples[name] != backward+forward:
        crop_back = Neu4mes.input_ns_backward[name]-backward
        crop_front = Neu4mes.input_ns_forward[name]-forward
        if offset is not None:
            step1 = tensorflow.keras.layers.Reshape((Neu4mes.input_n_samples[name],-1))(input-input[:,Neu4mes.input_ns_backward[name]-offset-1])
        else:
            step1 = tensorflow.keras.layers.Reshape((Neu4mes.input_n_samples[name],-1))(input) 
        return tensorflow.keras.layers.Reshape((backward+forward,))(
                tensorflow.keras.layers.Cropping1D(cropping=(crop_back, crop_front))(step1))
    else:
        return input

setattr(neu4mes.Neu4mes, 'discreteInput', createDiscreteInput)
setattr(neu4mes.Neu4mes, 'discreteInputRNN', createDiscreteInputRNN)
setattr(neu4mes.Neu4mes, 'input', createInput)
setattr(neu4mes.Neu4mes, 'inputRNN', createInputRNN)
setattr(neu4mes.Neu4mes, 'part', createPart)

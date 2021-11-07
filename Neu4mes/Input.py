import Neu4mes
import tensorflow.keras.layers
import tensorflow as tf
import numpy as np

class Input(Neu4mes.NeuObj.NeuObj):
    def __init__(self,name,values = None):
        super().__init__()
        self.name = name
        self.max_tw = 0
        self.json['Inputs'][self.name] = {}
        if values:
            self.values = values
            self.json['Inputs'][self.name] = {
                'Discrete' : values
            }


    def tw(self, seconds):
        if self.max_tw < seconds:
            self.max_tw = seconds
        return self, seconds 
    
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

def createDiscreteInput(Neu4mes, name, size, types):
    input = tensorflow.keras.layers.Input(shape = (size, ), batch_size = None, name = name, dtype='int32')
    return (input,tensorflow.keras.layers.Lambda(lambda x: tf.one_hot(x[:,0], len(set(np.asarray(types)))))(input))

def createInput(Neu4mes, name, size):
    input = tensorflow.keras.layers.Input(shape = (size, ), batch_size = None, name = name)
    return (input,input)

def createPart(Neu4mes, name, input, size):
    if Neu4mes.input_n_samples[name] != size:
        crop_velue = Neu4mes.input_n_samples[name]-size
        return tensorflow.keras.layers.Reshape((size,))(
            tensorflow.keras.layers.Cropping1D(cropping=(0, crop_velue))(
                tensorflow.keras.layers.Reshape((Neu4mes.input_n_samples[name],-1))(input)))
    else:
        return input

setattr(Neu4mes.Neu4mes, 'discreteInput', createDiscreteInput)
setattr(Neu4mes.Neu4mes, 'input', createInput)
setattr(Neu4mes.Neu4mes, 'part', createPart)
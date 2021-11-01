import Neu4mes
import tensorflow.keras.layers

class Input(Neu4mes.NeuObj.NeuObj):
    def __init__(self,name):
        super().__init__()
        self.name = name
        self.max_tw = 0
        self.json['Inputs'][self.name] = {}

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

def createInput(Neu4mes, name, size):
    return tensorflow.keras.layers.Input(shape = (size, ), batch_size = None, name = name)

def createPart(Neu4mes, name, input, size):
    if Neu4mes.input_n_samples[name] != size:
        crop_velue = Neu4mes.input_n_samples[name]-size
        return tensorflow.keras.layers.Reshape((size,))(
            tensorflow.keras.layers.Cropping1D(cropping=(0, crop_velue))(
                tensorflow.keras.layers.Reshape((Neu4mes.input_n_samples[name],-1))(input)))
    else:
        return input

setattr(Neu4mes.Neu4mes, 'input', createInput)
setattr(Neu4mes.Neu4mes, 'part', createPart)
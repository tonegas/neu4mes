import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())


import logging

import time
from neu4mes import *
from neu4mes.neu4mes import log
from neu4mes.visualizer import StandardVisualizer

# Create neural model
x = Input('x')
F = Input('Fs')
F1= Input('Fa')
F2 = Input('F2')
F3 = Input('F3')
F4 = Input('F4')
F5 = Input('F5')
F6 = Input('F6')


#log.setLevel(logging.WARNING)

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)
log.disable()
log.enable()
Fir(x.tw(2))+Fir(F)
x_z = Output('x_k', Fir(x.tw(2))+Fir(F)+Fir(F1)+Fir(F2)+Fir(F3))
log.disable()

# Add the neural model to the neu4mes structure and neuralization of the model
mass_spring_damper = Neu4mes(verbose = True)
mass_spring_damper.addModel(x_z)
mass_spring_damper.minimizeError('min', x.z(-1), x_z, 'mse')
mass_spring_damper.neuralizeModel(0.05)


# Data load
data_struct = ['time','x','x_s','F']
data_folder = './examples/datasets/mass-spring-damper/data/'
mass_spring_damper.loadData(data_struct)

# Neural network train
start = time.time()
mass_spring_damper.trainModel(test_percentage = 10)
end = time.time()
print(end - start)

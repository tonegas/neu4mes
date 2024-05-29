import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())


import logging

import time
from neu4mes import *
from neu4mes.visualizer import StandardVisualizer

# Create neural model
x = Input('x')
F = Input('F')



#log.setLevel(logging.WARNING)

#log = logging.getLogger("mass-spring-damper")
#logging.getLogger("neu4mes.relation").setLevel(logging.DEBUG)
#logging.getLogger("neu4mes.neu4mes").setLevel(logging.DEBUG)
#logging.getLogger("neu4mes.output").setLevel(logging.DEBUG)
#log.setLevel(logging.DEBUG)
#log.enable()
#Fir(x.tw(2))
#log.titlejson('PROVA',{'Gas':'gas'})
#log.debug('GASTONE')
#log.info('GASTONE')
#log.warning('GASTONE')
#log.error('GASTONE')
#log.disable()

 #+Fir(F))

#x_z = Output('x_k', Fir(x)+Fir(F))


# Add the neural model to the neu4mes structure and neuralization of the model
mass_spring_damper = Neu4mes(verbose = 1)
#mass_spring_damper.addModel(x_z)
mass_spring_damper.minimizeError('position', x.z(-1), Fir(x)+Fir(F), 'mse')
mass_spring_damper.minimizeError('velocity', x.z(-1), Fir(x.tw(2))+Fir(F), 'mse')
#mass_spring_damper.minimizeError('acc',Fir(x.tw(2))+Fir(F.tw(1)), Fir(x.tw(2))+Fir(F), 'mse')
#log.disable()
#log.enable()
mass_spring_damper.neuralizeModel()
#log.disable()
#log.debug('GASTONE')
# run funzione check
#mass_spring_damper({'x':[4,2,3,4,4]})
# Warning
#mass_spring_damper({'x':[4,2,3,4,9],'F':[2]})

# Data load
data_struct = ['time','x','x_s','F']
data_folder = './examples/datasets/mass-spring-damper/data/'
mass_spring_damper.loadData(data_folder, data_struct)

# Neural network train
start = time.time()
mass_spring_damper.trainModel(test_percentage = 10, training_params={'num_of_epochs': 2})
end = time.time()
#print(end - start)

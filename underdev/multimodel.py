import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
from neu4mes.visualizer import MPLVisulizer
from pprint import pprint

#Ho due modelli connessi in serie per una variabile e poi ogni modello ha dei propri ingressi e delle proprie uscite

a = Input('a')
b_target = Input('b_t')
b = Output('b',Linear(W='tone')(Fir(parameter='gas')(a.tw(0.5))))

model = Neu4mes()
model.addModel(b)
model.minimizeError('b_min',b,b_target.last())
model.neuralizeModel(0.1)

c = Input('c')
b_in = Input('b_in')
d_target = Input('d_t')
d = Output('d',Fir(c.tw(0.5))+Fir(b_in.tw(0.2)))

model.addModel('d_model', d)
model.minimizeError('d_min',d,d_target.last())
model.neuralizeModel(0.1)
model.trainModel(model = 'd_model', connect={'b_in':'b'})


# Casi possibili
# 1) Caso del controllore voglio che i modelli siano entità indipendenti
# 2) Caso dei modelli complessi costruiti e trainati a pezzi voglio che alla fine siamo un modello solo
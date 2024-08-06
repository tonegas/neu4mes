import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
from neu4mes.visualizer import MPLVisulizer
from pprint import pprint

data_a = np.arange(1,1001, dtype=np.float32)
data_b_t = np.arange(3,1003, dtype=np.float32)

data_c = np.arange(2,1002, dtype=np.float32)
data_b_in = np.arange(5,1005, dtype=np.float32)
data_d_t = np.arange(4,1004, dtype=np.float32)

dataset = {'a': data_a, 'b_t': data_b_t, 'c':data_c, 'b_in': data_b_in, 'd_t':data_d_t }

#Ho due modelli connessi in serie per una variabile e poi ogni modello ha dei propri ingressi e delle proprie uscite
# L'uscita del modello b è connesso al all'ingresso del modello d
# Modello b
a = Input('a')
b_t = Input('b_t')
b = Output('b',Linear(W='condiviso')(a.last())+Linear(W='A')(Fir(parameter='B')(a.tw(0.5))))

model = Neu4mes()
model.addModel('b_model',b)
model.addMinimize('b_min',b,b_t.last())
model.neuralizeModel(0.1)
model.loadData('dataset', dataset)

# Faccio il training solo di b_model
model.trainModel()

# Modello d
c = Input('c')
b_in = Input('b_in')
d_t = Input('d_t')
d = Output('d',Linear(W='condiviso')(c.last())+Fir(parameter='C')(c.tw(0.5))+Fir(parameter='D')(b_in.tw(0.2)))

model.addModel('d_model', d)
model.addMinimize('d_min',d,d_t.last())
model.neuralizeModel(0.1)

# Faccio il traning solo dei parametri di d_model
# Le minimize sono tutte attive
# Equivalenti
model.trainModel(model = 'd_model', connect={'b_in':'b'})
model.trainModel(connect={'b_in':'b'}, lr_gain = {'condiviso':0,'A':0,'B':0,'C':1,'D':1})

# Faccio il training di tutte e due modelli
# Le minimize sono tutte attive
# Equivalenti
model.trainModel(model = ['d_model','b_model'], connect={'b_in':'b'})
model.trainModel(connect={'b_in':'b'})

# Faccio il traning solo dei parametri di b_model
# Le minimize sono tutte attive
# Equivalenti
model.trainModel(model = 'b_model', connect={'b_in':'b'})
model.trainModel(connect={'b_in':'b'}, lr_gain = {'condiviso':0,'A':1,'B':1,'C':0,'D':0})

# Faccio il traning solo dei parametri di b_model
# Ma disattivo la minimize d_min perché anche quella genera un errore che modifica b_model
# Aggiungiamo anche una funzione per scegliere il gain delle funzioni minimize
# Equivalenti
model.trainModel(model = 'b_model', connect={'b_in':'b'}, minimize_gain={'d_min':0})
model.trainModel(connect={'b_in':'b'}, lr_gain = {'condiviso':0,'A':1,'B':1,'C':0,'D':0}, minimize_gain={'d_min':0}

# Serve una funzione per rimuovere un modello e rimuovere una minimize
# Questa funzione ci ragioniamo
model.removeModel(['d_model'])


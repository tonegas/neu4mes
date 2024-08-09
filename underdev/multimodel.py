import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
from neu4mes.visualizer import MPLVisulizer
from pprint import pprint

def linear_function(x, k1, k2):
    return x*k1 + k2

data_a = np.arange(1,1001, dtype=np.float32)
data_b_t = linear_function(data_a, 2, 3)

data_c = np.arange(1,1001, dtype=np.float32)
data_b_in = np.arange(5,1005, dtype=np.float32)
data_d_t = linear_function(data_c, 5, 1)

dataset = {'a': data_a, 'b_t': data_b_t, 'c':data_c, 'b_in': data_b_in, 'd_t':data_d_t }
#dataset = {'a': data_a, 'b_t': data_b_t, 'c':data_c, 'd_t':data_d_t }

#Ho due modelli connessi in serie per una variabile e poi ogni modello ha dei propri ingressi e delle proprie uscite
# L'uscita del modello b è connesso al all'ingresso del modello d
# Modello b
a = Input('a')
b_t = Input('b_t')
b = Output('b',Linear(W='condiviso')(a.last())+Linear(W='A')(Fir(parameter='B')(a.tw(0.5))))

model = Neu4mes(seed=42)
model.addModel('b_model',b)
model.addMinimize('b_min',b,b_t.last())
model.neuralizeModel(0.1)

#model.loadData('dataset', dataset)
# Faccio il training solo di b_model
#model.trainModel()

# Modello d
c = Input('c')
b_in = Input('b_in')
d_t = Input('d_t')
d = Output('d',Linear(W='condiviso')(c.last())+Fir(parameter='C')(c.tw(0.5))+Fir(parameter='D')(b_in.tw(0.2)))

model.addModel('d_model', d)
model.addMinimize('d_min',d,d_t.last())
model.neuralizeModel(0.1)
model.loadData('dataset', dataset)

params = {'num_of_epochs': 50, 
          'train_batch_size': 32, 
          'val_batch_size':32, 
          'test_batch_size':1, 
          'learning_rate':0.01}
'''
print('#### TRAINING 1 ####')
## training dei parametri di tutti i modelli
model.trainModel(training_params=params)
print('#### TRAINING 2 ####')
## training dei parametri del modello d_model e b_model (equivalente a training 1 se passo tutti i modelli)
model.trainModel(models=['b_model','d_model'],training_params=params)
print('#### TRAINING 3 ####')
## training dei parametri del modello d_model
model.trainModel(models='d_model', training_params=params)
print('#### TRAINING 4 ####')
## training dei parametri del modello d_model e di A con gain 2*lr
model.trainModel(models='d_model', training_params=params, lr_gain = {'A':2})
print('#### TRAINING 5 ####')
## training dei parametri del modello b_model e di A con gain 2*lr e condiviso con gain 5*lr
model.trainModel(models='b_model', training_params=params, lr_gain = {'A':2, 'condiviso':5})
'''

# Faccio il traning solo dei parametri di d_model
# Le minimize sono tutte attive
# Equivalenti
print('### BEFORE TRAIN ###')
print(model.model.relation_forward['Linear18'].lin.weight)
print(model.model.relation_forward['Linear8'].lin.weight)
print(model.model.relation_forward['Linear4'].lin.weight)
print('# PARAMETERS #')
print(model.model.all_parameters['A'])
print(model.model.all_parameters['B'])
print(model.model.all_parameters['C'])
print(model.model.all_parameters['D'])
print(model.model.all_parameters['condiviso'])
model.trainModel(models='d_model', training_params=params, splits=[100,0,0], connect={'b_in':'b'}, lr_gain={'condiviso':1, 'A':1, 'B':1, 'C':1, 'D':1})
#model.trainModel(connect={'b_in':'b'}, lr_gain = {'condiviso':0,'A':0,'B':0,'C':1,'D':1})
print('### AFTER TRAIN ###')
print(model.model.relation_forward['Linear18'].lin.weight)
print(model.model.relation_forward['Linear8'].lin.weight)
print(model.model.relation_forward['Linear4'].lin.weight)
print('# PARAMETERS #')
print(model.model.all_parameters['A'])
print(model.model.all_parameters['B'])
print(model.model.all_parameters['C'])
print(model.model.all_parameters['D'])
print(model.model.all_parameters['condiviso'])
print('### MODEL PARAMETERS ###')
for param in model.model.parameters():
    print(type(param), param.size())
print('### NUMBER OF PARAMETERS ###')
print(sum(p.numel() for p in model.model.parameters()))
'''
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
model.trainModel(connect={'b_in':'b'}, lr_gain = {'condiviso':0,'A':1,'B':1,'C':0,'D':0}, minimize_gain={'d_min':0})

# Serve una funzione per rimuovere un modello e rimuovere una minimize
model.removeModel('d_model')
model.removeMinimize('d_min')
# TODO evitare di ricreare i pesi ma utilizzare quelli già inizializzati
model.neuralizeModel(0.1)

'''

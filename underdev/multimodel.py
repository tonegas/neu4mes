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
d = Output('d',Linear(W='condiviso')(c.last())+Fir(parameter='C')(c.tw(0.5))+Fir(parameter='D')(b_in.tw(0.3)))

model.addModel('d_model', d)
model.addMinimize('d_min',d,d_t.last())
model.neuralizeModel(0.1)
model.loadData('dataset', dataset)

params = {'num_of_epochs': 15, 
          'train_batch_size': 32, 
          'val_batch_size': 32, 
          'test_batch_size':1, 
          'learning_rate':0.1}

def print_parameters():
    for name, value in model.model.all_parameters.items():
        print(f'parameter {name}: {value.data}')

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
print_parameters()
model.trainModel(models='b_model', training_params=params, lr_gain = {'A':2, 'condiviso':5})
print_parameters()

print('#### TRAINING 6 ####')
# Faccio il traning solo dei parametri di b_model
# Ma disattivo la minimize d_min perché anche quella genera un errore che modifica b_model
# Aggiungiamo anche una funzione per scegliere il gain delle funzioni minimize
model.trainModel(models = 'b_model', training_params=params, minimize_gain={'d_min':0.5, 'b_min':2.0})

print('#### TRAINING 7 ####')
## training dei parametri del modello d_model con connect tra ouput b e input b_in for 4 sample horizon
model.loadData('dataset', dataset)   ## b_in is initialized with the dataset since it has a bigger window dimension
model.trainModel(models='d_model', training_params=params, prediction_samples=4, connect={'b_in':'b'})

print('#### TRAINING 8 ####')
## training dei parametri del modello d_model con connect tra ouput b e input b_in (b and b_in have the same time window)
dataset = {'a': data_a, 'b_t': data_b_t, 'c':data_c, 'd_t':data_d_t } ## remove b_in from dataset we don't need it anymore
model.loadData('dataset', dataset)  ## b_in is initialized with zeros for the first b_in.tw samples
model.trainModel(models='d_model', training_params=params, prediction_samples=4, connect={'b_in':'b'})


'''
# Serve una funzione per rimuovere un modello e rimuovere una minimize
model.removeModel('d_model')
model.removeMinimize('d_min')
# TODO evitare di ricreare i pesi ma utilizzare quelli già inizializzati
model.neuralizeModel(0.1)


print('### NUMBER OF PARAMETERS ###')
print(sum(p.numel() for p in model.model.parameters()))
'''

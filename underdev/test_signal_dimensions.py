import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *

test = Neu4mes(visualizer=TextVisualizer(verbose=5))
vx = Input('vx', dimensions = 3)
def fun(x,P):
    print(f'x:{x.shape}')
    print(f'P:{P.shape}')
    import torch
    return x*P

parfun = ParamFun(fun)
out1 = Output('fun1',parfun(vx))

parfun2 = ParamFun(fun)
out2 = Output('fun2',parfun2(vx.tw([-2,0])))

test.addModel([out1,out2])
test.neuralizeModel(1)
# Test uscita
#Tempo     -2,-1,0,1,2,3
input   = [[-2,-1,0],[1,2,3]]
pprint(test({'vx':input}))

#
# # Neuralizzazione
# test.neuralizeModel(1)
# # Test uscita
# #Tempo     -2,-1,0,1,2,3
# input   = [-2,-1,0,1,2,3]
# pprint(test({'x':input}))
#
# input   = [-2,-1,0,1,2,3,4,5,6,7]
# pprint(test({'x':input}))
#
# # Neuralizzazione
# test.neuralizeModel(1)
# # Test uscita
# # Tempo         -3        -2          -1       0
# input   = [[-4,-3,-2],[-3,-2,-1],[-2,-1,0],[1,2,3]] #So che l'ingresso ha dimensione 3 e questi sono due input temporali
# pprint(test({'vx':input}))
# input   = [[-4,-3,-2],[-3,-2,-1],[-2,-1,0],[1,2,3],[0,0,0],[1,1,1]] #So che l'ingresso ha dimensione 3 e questi sono due input temporali
# pprint(test({'vx':input}))

# Funzioni

# Aritmetiche elementwise, Activation, Trigonomotriche
# le dimensioni e le funestre temporali rimangono invariate, per gli
# operatori binari devono essere uguali

# Fir
# Input scalare la dimensione temporale va a 1,
# Input vettoriali non ammessi, si potrebbe fare che vengono costruiti un numero di filtri fir pari alla dimensione del vettore
# I pesi devono essere condivisi o no?

# Fully Connected
# La dimensione temporale deve essere 1
# La dimensione di uscita è definita dall'utente
# Se la dimensione temporale non è 1 si potrebbe fare che vengono costruiti un numero di reti pari alla dimnesione temporale
# I pesi devono essere condivisi o no?

# Parametric Function
# Un input la dimensione sia temporale che non resta invariata a meno che non sia ridefinita in uscita
# Se ci sono più input la funzione ritorna un errore  se non si definiscono le dimensioni

# Part, Select, TimePart, TimeSelect
# Part seleziona una parte dell'input, la dimensione è la dimensione della parte, l'input deve essere un vettore Se c'è una componente temporale questa rimane invariata
# Select la dimensione diventa 1, l'input deve essere un vettore Se c'è una componente temporale questa rimane invariata
# TimePart viene selezionata una finestra temporale (funziona come timewindow tw) La dimensione rimane invariata
# TimeSelect viene selezionato un indice specifico? Forse è meglio sampleSelect

# Fuzzy
# La dimensione temporale resta invariata, l'input deve essere scalare.
# Input vettoriali non ammessi, si potrebbe fare che vengono fuzzificate tutte le dimensioni

# Local Model


'''data_struct = ['x1','P[5]','out'] # Questo rappresenta un vettore di 5 elementi
data_folder = 'DATA/'
test.loadData(data_struct, skiplines=4)
test.removeData(x < 5 & x > -5) # Rimuovi dal dataset tutti i dati dove x è compresa tra -5 e 5

test.trainModel(test_percentage = 50,  show_results = True)'''

# Ci sono tre problemi
# 1. come rappresento nel json ritardo anticipo derivata integrale finestre temporali finestre in sample
# 2. come rappresento il minimize error ? Non lo voglio rappresentare dentro il modello mi faccio un dizionario che seleziona una relazione
# però potrebbe succedere che tale relazione non è in uscita
# posso avere un modello senza uscite?!
# 3. come gestire gli output

# Devo capire se l'operazione che faccio merita avere una relazione oppure no
# Derivata x.s(1)
import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *

x = Input('x')
F = Input('F')
x_z = Output('x_k', Fir(x.tw(2))+Fir(F))
x_zz = Output('x_k2', Fir(x.tw(1))+Fir(F.tw(3)))
example1 = Neu4mes(verbose = True)
#example1.addModel(out)
example1.minimizeError(x_zz,x_z)
#example1.minimizeError(o.sw([-1,2]),out)
example1.neuralizeModel(0.05)
#print(example1({'x':[1,2,3,4,5],'x_s':[1,2,3]}))

data_struct = ['time','x','x_s','F']
data_folder = './examples/datasets/mass-spring-damper/data/'
example1.loadData(data_struct, folder = data_folder)
example1.trainModel(test_percentage = 10, show_results = True)



'''
#assert(example1({'x':[3]}),{'xtw1':[3]} )

#out2 = Output('xtw2',Fir(x.tw(0.1))+Fir(x.tw(1)))
example1.addModel(out)
example1.neuralizeModel(0.1)

#example1({'x':[1,2,3,4,5,6,7,8,9,10]})

#Time window
t = [-5,-4,-3,-2,-1,0,1,2,3,4]
x = Input('x') # samplerate = 1
x.tw(1)      #-> [0]
x.tw([-1,0]) #-> [0] il secondo è escluso
x.tw([-3,0]) #-> [-2, -1, 0]
x.tw([-3,1]) #-> [-2  -1  0 1]
x.tw([-0,1]) #-> [1] l'indice zero e basta

x.z(-1)      #-> [1]  l'indice zero
x.z(0)       #-> [0] l

x.sw()

'''
'''





x.tw(1) -> ('x',1) -> {'x':{'tw':1}}
x.tw([1,2]) -> {'x':{'tw':[1,2]}}
x.z(-1) -> {'x':{'z':-1}}
x.z([-1,-5]) -> {'x':{'z':[-1,-5]}}


{'x':{'z':-1}}

minimizeError(x.z(-1),dsdsds)
'Outputs': {'out': 'Fir160'},

('in1', 0.1) -> {'in1':{"tw":0.1}}

'Relations': {'Fir160': ['Fir', [('in1', 0.1)], 'PFir75']},

# Derivata del segnale
x.s(1) ->
# Integrale del segnale
x.s(-1) ->



{
 'Functions': {},
 'Inputs': {'in1': {'dim': 1}},
 'Outputs': {'out': 'Fir160'},
 'Parameters': {'PFir75': {'dim': 3, 'tw': 0.1}},
 'Relations': {'Fir160': ['Fir', [{'in1':{"tw":0.1}}], 'PFir75']},
 'SampleTime': 0.01
 }

('in1', 0.1) -> {'in1':{"tw":0.1}}

{'x':{'z':-1}}
'''
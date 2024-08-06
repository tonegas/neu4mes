from neu4mes import *
## Reti ricorrenti
# Ci sono due casi in cui ha senso una rete ricorrente:
# 1. Training ricorrente ho una rete che normalmente funziona come rete feedforward ma voglio verificarne e migliorarne la stabilità
# e quindi eseguo un training dove una o più variabili che erano calcolate in uscita sono utilizzate in ingresso.
# esempio la predizione della posizione della massa futura che dipende dalla posizione della massa corrente.
# Normalmente la posizione della massa corrente la recupero da un sensore ma nel caso voglia verificare la capacità della
# rete di predirre il futuro, metto in ingresso l'uscita della rete per un certo orizzonte di predizione.
# 2. Ho una rete che di natura è ricorrente quindi lei stima degli stati e questi sono utilizzati in loop.
# Può succedere che questi stati siano anche input in fase di training nel senso che possa leggerli per una prima fase di training
# non ricorrente.

# Caso 1
x = Input('x')
F = Input('F')
x_k1 = Fir(x.tw(0.5))+F
est_x_k1 = Output('xk1',x_k1)

mass_spring_damper = Neu4mes(verbose = True)
mass_spring_damper.addModel(est_x_k1)
mass_spring_damper.addMinimize(x.z(-1), x_k1, )
mass_spring_damper.neuralizeModel(0.1)
data_struct = ['time','x','x_s','F']
data_folder = './examples/datasets/mass-spring-damper/data/'
mass_spring_damper.loadData(data_struct)

# Training non ricorrente
mass_spring_damper.trainModel(test_percentage = 10, show_results = True)

# Training ricorrente
# bisogna passare alla variabile close_loop un dizionario che indica per ogni variabile di input una variabile di output
# Le dimensioni di ingresso ed uscita devono essere le medesime.
# La finestra temporale di 'x' è riempita inizialmente con i dati presi dal file e poi successivemente è riempieta utilizzando
# l'uscita 'xk1' per un orizzonte temporale di 1 secondo.
mass_spring_damper.trainModel(test_percentage = 10, show_results = True, close_loop = {'x':'xk1'}, prediction_horizon = 1, step = 1 )


# Caso 2
x = State('x') # Questa è una variabile che si considera come stato
F = Input('F')
x_k1 = Fir(x.tw(0.5))+F
x.update(x_k1) # Con questa funzione collego uno stream a uno stato
est_x_k1 = Output('xk1',x_k1)

mass_spring_damper = Neu4mes(verbose = True)
mass_spring_damper.addModel(est_x_k1)
mass_spring_damper.neuralizeModel(0.05)

# Dopo che è stato chiamato neuralizeModel
# Esempio di utilizzo
mass_spring_damper({'F':5}) # La posizione x viene cosiderata un vettore di 0 Uscita un numero fondamentalmente casuale, si aggiorna lo stato 'x'
mass_spring_damper({'F':5,'x':[1,2,3,4,5,6,7,8,9,10]}) # La posizione x viene passata dall'esterno lo stato è resettato uscita un numero fondamentalmente casuale
for i in range(10):
    mass_spring_damper({'F':i}) # Qui l'uscità sarà sempre diversa perche si aggiorna lo stato 'x' ad ogni volta esempio {'xk1':0.323241}
mass_spring_damper.clearState() # Funzione che ripulisce lo stato

# L'operazione
x.z(-1) # Operazione non ammissibile su uno stato non può accedere al futuro
# Quindi per leggere un valore dal dataset posso utilizzare una nuova variabile input
x_true = Input('x')
mass_spring_damper.addMinimize(x_true.z(-1), x_k1, )

# Caricamento dei dati in questo caso è presente 'x' come input
data_struct = ['time','x','x_s','F'] # La x la uso sia per inizializzare lo stato sia per inizializzare x_true
data_folder = './examples/datasets/mass-spring-damper/data/'
mass_spring_damper.loadData(data_struct)

# Training non ricorrente in questo caso leggo x sia come stato della rete che come input x_true
mass_spring_damper.trainModel(test_percentage = 10, show_results = True)

# Training ricorrente in questo caso inizializzo lo stato x con i valori letti dal dataset e
# poi uso lo stato in modo ricorrente per 1 sec comunque leggo x dal dataset per l'uscita x_true
mass_spring_damper.trainModel(test_percentage = 10, show_results = True, prediction_horizon = 1)

# Mettiamo il caso che voglio inizializzare lo stato a zero basterà che nessuna colonna del dataset sia uguale alla variabile di stato
x_true = Input('xy')
data_struct = ['time','xy','x_s','F']
mass_spring_damper.addMinimize(x_true.z(-1), x_k1, )
mass_spring_damper.loadData(data_struct)

# Training non ricorrente ma in questo caso la variabile di stato è settata a 0 in pratica si usa sempre una finestra di zero
# la variabile xy (cioè x_true) è letta da file
mass_spring_damper.trainModel(test_percentage = 10, show_results = True)

# Training ricorrente in questo caso inizializzo lo stato x con 0 e poi uso lo stato in modo ricorrente per 1 sec
# la variabile xy (cioè x_true) è letta da file
mass_spring_damper.trainModel(test_percentage = 10, show_results = True, prediction_horizon = 1)


#mass_spring_damper.minimizeError(dasdas)
#mass_spring_damper.addModel('out',out)
#mass_spring_damper.addModel('contr',outcontr)
#mass_spring_damper.trainModel(models=['out'])
#mass_spring_damper.trainModel(models=['contr'],connect={'out.in':'contr.out'},horizon=10,step=1,parameters={'par':{'multipliers':0}})
#mass_spring_damper.trainModel(models=['contr'],connect={'out.in':'contr.out'},horizon=10,step=1,parameters={'par':{'multipliers':3}})

{'Functions': {},
 'Inputs': {'F': {'dim': 1, 'sw': [-1, 0], 'tw': [0, 0]},
            'dx': {'dim': 1, 'sw': [0, 1], 'tw': [0, 0]},
            'x': {'dim': 1, 'sw': [0, 1], 'tw': [-0.2, 0]}},
 'States': {'x': {'dim': 1, 'sw': [0, 1], 'tw': [-0.2, 0], 'update':'Add17'},
            'xy': {'dim': 1, 'sw': [0, 1], 'tw': [-0.2, 0], 'update':'Add9'}},
 'Outputs': {'dx[k+1]': 'Fir18', 'x[k+1]': 'Add9'},
 'Parameters': {'PFir3': {'dim': 1, 'tw': 0.2},
                'PFir4': {'dim': 1, 'sw': 1},
                'PFir5': {'dim': 1, 'tw': 0.2},
                'PFir6': {'dim': 1, 'sw': 1},
                'PFir7': {'dim': 1, 'sw': 1}},
 'Relations': {'Add17': ['Add', ['Fir13', 'Fir16']],
               'Add9': ['Add', ['Fir5', 'Fir8']],
               'Fir13': ['Fir', ['TimePart12'], 'PFir5'],
               'Fir16': ['Fir', ['SamplePart15'], 'PFir6'],
               'Fir18': ['Fir', ['Add17'], 'PFir7'],
               'Fir5': ['Fir', ['TimePart4'], 'PFir3'],
               'Fir8': ['Fir', ['SamplePart7'], 'PFir4'],
               'SamplePart15': ['SamplePart', ['F'], [-1, 0]],
               'SamplePart21': ['SamplePart', ['x'], [0, 1]],
               'SamplePart23': ['SamplePart', ['dx'], [0, 1]],
               'SamplePart7': ['SamplePart', ['F'], [-1, 0]],
               'TimePart12': ['TimePart', ['x'], [-0.2, 0]],
               'TimePart4': ['TimePart', ['x'], [-0.2, 0]]},
 'SampleTime': 0.05}






# Caso 2
x = State('x_state') # Questa è una variabile che si considera come stato
F = Input('F')
x_k1 = Fir(x.tw(0.5))+F
x.update(x_k1) # Con questa funzione collego uno stream a uno stato
est_x_k1 = Output('xk1',x_k1)

mass_spring_damper = Neu4mes(verbose = True)
mass_spring_damper.addModel(est_x_k1)
mass_spring_damper.neuralizeModel(0.05)

mass_spring_damper.addMinimize('error', x_k1, F.next())

# Caricamento dei dati in questo caso è presente 'x' come input
data_struct = ['time',('x','x_state'),'x_s','F'] # La x la uso sia per inizializzare lo stato
data_folder = './examples/datasets/mass-spring-damper/data/'
mass_spring_damper.loadData(data_struct)

# Training non ricorrente in questo caso leggo x come stato 
mass_spring_damper.trainModel(test_percentage = 10, show_results = True)

# Training ricorrente in questo caso inizializzo lo stato x con i valori letti dal dataset e
# poi uso lo stato in modo ricorrente per 1 sec
mass_spring_damper.trainModel(test_percentage = 10, show_results = True, prediction_horizon = 1)


{'Functions': {},
 'Inputs': {'F': {'dim': 1, 'sw': [-1, 0], 'tw': [0, 0]},
            'dx': {'dim': 1, 'sw': [0, 1], 'tw': [0, 0]},
            'x': {'dim': 1, 'sw': [0, 1], 'tw': [-0.2, 0]},
            'x_state': {'dim': 1, 'sw': [0, 1], 'tw': [-0.2, 0]}},
 'States': {'x_state': {'dim': 1, 'sw': [0, 1], 'tw': [-0.2, 0], 'update':'Add17'},
            'y': {'dim': 1, 'sw': [0, 1], 'tw': [-0.2, 0], 'update':'Add9'}},
 'Outputs': {'dx[k+1]': 'Fir18', 'x[k+1]': 'Add9'},
 'Parameters': {'PFir3': {'dim': 1, 'tw': 0.2},
                'PFir4': {'dim': 1, 'sw': 1},
                'PFir5': {'dim': 1, 'tw': 0.2},
                'PFir6': {'dim': 1, 'sw': 1},
                'PFir7': {'dim': 1, 'sw': 1}},
 'Relations': {'Add17': ['Add', ['Fir13', 'Fir16']],
               'Add9': ['Add', ['Fir5', 'Fir8']],
               'Fir13': ['Fir', ['TimePart12'], 'PFir5'],
               'Fir16': ['Fir', ['SamplePart15'], 'PFir6'],
               'Fir18': ['Fir', ['Add17'], 'PFir7'],
               'Fir5': ['Fir', ['TimePart4'], 'PFir3'],
               'Fir8': ['Fir', ['SamplePart7'], 'PFir4'],
               'SamplePart15': ['SamplePart', ['F'], [-1, 0]],
               'SamplePart21': ['SamplePart', ['x'], [0, 1]],
               'SamplePart23': ['SamplePart', ['dx'], [0, 1]],
               'SamplePart7': ['SamplePart', ['F'], [-1, 0]],
               'TimePart12': ['TimePart', ['x'], [-0.2, 0]],
               'TimePart4': ['TimePart', ['x'], [-0.2, 0]]},
 'SampleTime': 0.05}
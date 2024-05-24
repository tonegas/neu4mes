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
mass_spring_damper.minimizeError(x.z(-1),x_k1)
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
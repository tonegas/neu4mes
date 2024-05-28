import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

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
#x_k2 = Fir(x.tw(0.5))+Fir(F.tw(0.5))
est_x_k1 = Output('xk1',x_k1)  ## TODO: should work without Output

mass_spring_damper = Neu4mes(verbose = 2)
mass_spring_damper.addModel(est_x_k1) ## TODO: should work without addModel
mass_spring_damper.minimizeError('out',x.z(-1),x_k1)
#mass_spring_damper.minimizeError('out2', x_k1, x_k2)
mass_spring_damper.neuralizeModel(0.1)

## build custom dataset
data_x = np.asarray(range(1000), dtype=np.float32)
data_F = np.asarray([i*5 for i in range(1000)], dtype=np.float32)
dataset = {'x': data_x, 'F': data_F}

## Load Data
mass_spring_damper.loadData(source=dataset)

# Training non ricorrente
#mass_spring_damper.trainModel(test_percentage = 10)

# Training ricorrente
# bisogna passare alla variabile close_loop un dizionario che indica per ogni variabile di input una variabile di output
# Le dimensioni di ingresso ed uscita devono essere le medesime.
# La finestra temporale di 'x' è riempita inizialmente con i dati presi dal file e poi successivemente è riempieta utilizzando
# l'uscita 'xk1' per un orizzonte temporale di 1 secondo.
params = {'train_batch_size':8, 'test_batch_size':4}
mass_spring_damper.trainRecurrentModel(close_loop = {'x':'xk1'}, 
                                       prediction_horizon = 1, 
                                       step = 1, 
                                       test_percentage = 10, 
                                       training_params=params)

## Example prediction after training
print('EXAMPLE 1')
random_sample = mass_spring_damper.get_random_samples(window=2)
print('random sample: ',random_sample)
results = mass_spring_damper(random_sample, sampled=True)
print('prediction: ', results['xk1'])
print('label: ', results['out_x'])

print('EXAMPLE 2')
sample = {'x':[1.0,2.0,3.0,4.0,5.0,6.0], 'F':[25.0]}
print('sample: ', sample)
results = mass_spring_damper(sample)
print('prediction: ', results['xk1'])
print('label: ', results['out_x'])

print('EXAMPLE 3')
sample = {'x':[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0], 'F':[25.0,30.0,35.0,40.0,45.0]}
print('sample: ', sample)
results = mass_spring_damper(sample)
print('prediction: ', results['xk1'])
print('label: ', results['out_x'])
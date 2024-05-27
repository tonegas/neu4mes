import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
import torch
torch.manual_seed(1)

# Cose da sistemare per far andare questo file
# 1. Modificare il discorso degli Output
# 2. Creare la function call per la rete
# 3. Aggiungere la funzione minimizeError per gestire il training
x = Input('x')
F = Input('F')
x_k1 = Fir(x.tw(0.5))+Fir(F)

# La funzione Output prende due parametri il primo è un etichetta e il secondo è uno Stream
est_x_k1 = Output('xk1',x_k1)

# Dopo che chi neuralizzato
example1 = Neu4mes(verbose = False)
example1.addModel(est_x_k1)
example1.neuralizeModel(0.05)

# Posso fare queste chiamate  (X: 10 samples and F: 1 samples)
print('EXAMPLE 1')
results = example1(inputs={'F':[9],'x':[3,4,5,6,7,8,9,10,11,12]}) # 1 window -> 1 output
for output, result in results.items():
    print(f'prediction for {output}: {result}')

print('EXAMPLE 2')
results = example1(inputs={'F':[5,2],'x':[1,2,3,4,5,6,7,8,9,10,11]}) # 2 window -> 2 output
for output, result in results.items():
    print(f'prediction for {output}: {result}')

print('EXAMPLE 3')
results = example1(inputs={'F':[5,4,5],'x':[1,2,3,4,5,6,7,8,9,10]}) # 1 window (x = 10) -> 1 output
for output, result in results.items():
    print(f'prediction for {output}: {result}')

print('EXAMPLE 4')
results = example1(inputs={'F':[5,3,4,1],'x':[1,2,3,4,5,6,7,8,9,10]}) # 1 window (x = 10) -> 1 output
for output, result in results.items():
    print(f'prediction for {output}: {result}')

print('EXAMPLE 5')
results = example1(inputs={'F':[5,2],'x':[1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,5]}) # 2 window (F = 2) -> 2 output
for output, result in results.items():
    print(f'prediction for {output}: {result}')

print('EXAMPLE 6')
results = example1(inputs={'F':[5,2,4,5,1,7,8,9,10],'x':[1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,5]}) # 9 window (F = 9 ; x = 20) -> 9 output
for output, result in results.items():
    print(f'prediction for {output}: {result}')

## Now we try to use the random sampler
print('EXAMPLE 7')
theta = Input('theta')
T     = Input('torque')
out = Output('theta_s', Fir(theta)+Fir(T))
example2 = Neu4mes(verbose = False)
example2.addModel(out)
example2.neuralizeModel(0.05)
# Data load
data_struct = ['time','theta','theta_s','','','torque']
data_folder = './examples/datasets/pendulum/data/'
example2.loadData(source=data_folder, format=data_struct)
sample = example2.get_random_samples(window=3)
print('random sample: ', sample)
results = example2(inputs=sample) # 3 sample -> 3 output
for output, result in results.items():
    print(f'prediction for {output}: {result}')


#print('EXAMPLE 8') ##WORK IN PROGRESS
# La funzione prende in ingresso due Stream
# Adesso facciamo che funziona come prima e non gestisce due reti poi faremo anche la cosa che gestisce due reti
example2.minimizeError('out', x.z(-1),x_k1)
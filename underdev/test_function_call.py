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
example1 = Neu4mes(verbose = True)
example1.addModel(est_x_k1)
example1.neuralizeModel(0.05)

# Posso fare queste chiamate
results = example1(inputs={'F':[9],'x':[3,4,5,6,7,8,9,10,11,12]}) # x ed F sono passate alla funzione
for output, result in results.items():
    print(f'prediction for {output}: {result}')






results = example1(inputs={'F':[5,2],'x':[1,2,3,4,5,6,7,8,9,10,11]}) # x ed F sono passate alla funzione
for output, result in results.items():
    print(f'prediction for {output}: {result}')
results = example1(inputs={'F':[5,4,5],'x':[1,2,3,4,5,6,7,8,9,10]}) # x ed F sono passate alla funzione
for output, result in results.items():
    print(f'prediction for {output}: {result}')
results = example1(inputs={'F':[5,3,4,1],'x':[1,2,3,4,5,6,7,8,9,10]}) # x ed F sono passate alla funzione
for output, result in results.items():
    print(f'prediction for {output}: {result}')
results = example1(inputs={'F':[[5],[2]],'x':[[1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,5]]}) # x ed F sono passate alla funzione
for output, result in results.items():
    print(f'prediction for {output}: {result}')
# results = example1(inputs={'F':[[5],[2],[5]],'x':[[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,5]]}) # x ed F sono passate alla funzione
# for output, result in results.items():
    # print(f'prediction for {output}: {result}')
# il ritorno dovrebbe essere una cosa del genere
# {'xk1': 3.231} adesso è un numero casuale dopo il traning sarà un numero sensato
for output, result in results.items():
    print(f'prediction for {output}: {result}')

# La funzione prende in ingresso due Stream
# Adesso facciamo che funziona come prima e non gestisce due reti poi faremo anche la cosa che gestisce due reti
example1.minimizeError(x.z(-1),x_k1)
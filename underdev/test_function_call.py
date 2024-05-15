from neu4mes import *
# Cose da sistemare per far andare questo file
# 1. Modificare il discorso degli Output
# 2. Creare la function call per la rete
# 3. Aggiungere la funzione minimizeError per gestire il training
x = Input('x')
F = Input('F')
x_k1 = Fir(x.tw(0.5))+F

# La funzione Output prende due parametri il primo è un etichetta e il secondo è uno Stream
est_x_k1 = Output('xk1',x_k1)

# Dopo che chi neuralizzato
example1 = Neu4mes(verbose = True)
example1.addModel(est_x_k1)
example1.neuralizeModel(0.05)

# Posso fare queste chiamate
example1({'F':5,'x':[1,2,3,4,5,6,7,8,9,10]}) # x ed F sono passate alla funzione
# il ritorno dovrebbe essere una cosa del genere
# {'xk1': 3.231} adesso è un numero casuale dopo il traning sarà un numero sensato

# La funzione prende in ingresso due Stream
# Adesso facciamo che funziona come prima e non gestisce due reti poi faremo anche la cosa che gestisce due reti
example1.minimizeError(x.z(-1),x_k1)
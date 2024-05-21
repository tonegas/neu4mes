import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *

x = Input('x')
F = Input('F')

print("------------------------EXAMPLE 1------------------------")
# Example 1 Parametric Function Basic
# Questa funzione ha due parametri p1 e p2 di dimensione 1 e due ingressi K1 e K2
# La dimensione di uscita viene definita dall'utente
# se non è specificata ci si aspetta che l'uscita sia 1 se poi non è 1 la funzione da errore
def myFun(K1,K2,p1,p2):
    return p1*K1+p2*np.sin(K2)

parfun = ParamFun(myFun,1) # definisco una funzione scalare basata su myFun
gianni = Fir()
out = Output('x.z(-1)',gianni(parfun(x,F)) + gianni(x))
example = Neu4mes(verbose = True)
example.addModel(out)
example.neuralizeModel(0.05)
#

print("------------------------EXAMPLE 2------------------------")
# Example 2
# definisco l'uscita della funzione che adesso è 4 se poi la funzione non esce con dimensione 4 do errore
# anche in questo caso ho due parametri p1 e p2
# nella funzione c'è un prodotto tra un vettore ed uno scalare e poi una somma con uno scalare
# la dimensione dei parametri p1 e p2 è 1
def myFun(K1,K2,p1,p2):
    import torch
    return torch.tensor([p1,p1,p1,p1])*K1+p2*np.sin(K2)
parfun = ParamFun(myFun, output_dimension = 4) # definisco una funzione scalare basata su myFun
out = Output('out',parfun(x,F))
example = Neu4mes(verbose = True)
example.addModel(out)
example.neuralizeModel(0.05)
#

print("------------------------EXAMPLE 3------------------------")
#Example 3
# Questo caso definisco la dimensione specifica dei parametri
# il primo p1 è un vettore colonna da 4 righe
# La dimensione di uscita della funzione è 1
# in Questo caso faccio un prodotto scalare tra il vettore e p1 che è un vettore [4,1]
# La dimensione temporale dell'uscita non è definita ma dipende dell'input
# Nella prima chiamata parfun(x.tw(1),F.tw(1)) l'uscita temporale è una finestra di 1 sec
# Nella seconda chiamata parfun(x,F) è un uscita istantanea
def myFun(K1,K2,p1):
    import torch
    return torch.tensor([K1,2*K1,3*K1,4*K1])*p1+np.sin(K2)
parfun = ParamFun(myFun,1, parameters_dimensions = {'p1':[4,1]})
out = Output('out',Fir(parfun(x.tw(1),F.tw(1)))+parfun(x,F))
example = Neu4mes(verbose = True)
example.addModel(out)
example.neuralizeModel(0.05)
#

print("------------------------EXAMPLE 4------------------------")
# Example 4
# Questo caso creo un parametro che passo alla funzione parametrica
# La funzione parametrica prende un parametro di dimensione 4
# La funzione ha in ingresso tre input i primi due sono input ed il secondo un parametro K
# La funzione crea un tensore esegue un prodotto scalare tra l'input 1 e p1 (che di fatto è K Parametro)
K = Parameter('k', dimensions =  4) # dovrebbe essere uguale all'esempio 3
parfun = ParamFun(myFun, output_dimension = 1, parameters = [K] )
out = Output(x.z(-1),parfun(x,F))
example = Neu4mes(verbose = True)
example.addModel(out)
example.neuralizeModel(0.05)
#



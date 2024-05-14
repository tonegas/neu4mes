from neu4mes import *

x = Input('x')
F = Input('F')
activationA = Fuzzify(2,[0,1],functions='Triangular')(x)
activationB = Fuzzify(2,[0,1],functions='Triangular')(F)

print("------------------------EXAMPLE 1------------------------")
# Example 1
# Singola funzione di attivazione e solo funzione in input
loc = LocalModel(input_function = lambda : Fir)(x.tw(1),activationA)
out = Output(x.z(-1),loc)
example = Neu4mes(verbose = True)
example.addModel(out)
example.neuralizeModel(0.05)
#

print("------------------------EXAMPLE 2------------------------")
# Example 2
# Due funzioni di attivazione e quindi 4 filtri di ingresso
loc = LocalModel(input_function = Fir)(x.tw(1),(activationA,activationB))
out = Output(x.z(-1),loc)
example = Neu4mes(verbose = True)
example.addModel(out)
example.neuralizeModel(0.05)
#

print("------------------------EXAMPLE 3------------------------")
# Example 3
# Modello locale con una funzione di attivazione e una funzione parametrica di uscita
def myFun(in1,p1):
    return in1*p1
loc = LocalModel(output_function = lambda:ParamFun(myFun))(x,activationA)
out = Output(x.z(-1),loc)
example = Neu4mes(verbose = True)
example.addModel(out)
example.neuralizeModel(0.05)
#

print("------------------------EXAMPLE 4------------------------")
# Example 4
# Nuova funzione di attivazione con uscita una finestra temporale di 1 secondo
# Il modello locale ha una funzione di ingresso myFun e una funzione di uscita un filtro Fir che esce con dimensione 1
# Nella funzione di uscita non essendoci la lamda il filtro Fir condivide i pesi in uscita
activationA = Fuzzify(2,[0,1],functions='Triangular')(x.tw(1))
def myFun(in1,p1,p2):
    return p1*in1+p2*np.sin(in1)
loc = LocalModel(input_function = lambda:ParamFun(myFun), output_function = Fir())(x.tw(1),activationA)
out = Output(x.z(-1),loc)
example = Neu4mes(verbose = True)
example.addModel(out)
example.neuralizeModel(0.05)
#

# TODO Manca la parte con input_relation_matrix per la funzione parametrica
#loc = LocalModel(input_function = lambda:ParamFun(myFun), input_relation_matrix=[[1,1][1,1]] )(x.tw(1),(activationA,activationB))
#out = Output(x.z(-1),loc)
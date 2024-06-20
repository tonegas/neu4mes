import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *

x = Input('x')
F = Input('F')
activationA = Fuzzify(2,[0,1],functions='Triangular')(x.last())
activationB = Fuzzify(2,[0,1],functions='Triangular')(F.last())

print("------------------------EXAMPLE 1------------------------")
# Example 1
# Singola funzione di attivazione e solo funzione in input
loc = LocalModel(input_function = lambda : Fir)(x.tw(1),activationA)
out = Output('out',loc)
example = Neu4mes()
example.addModel(out)
example.neuralizeModel(0.25)
print(example({'x':[2,3,3,5,0]}))
#

print("------------------------EXAMPLE 2------------------------")
# Example 2
# Due funzioni di attivazione e quindi 4 filtri di ingresso
loc = LocalModel(input_function = Fir)(x.tw(1),(activationA,activationB))
out = Output('out',loc)
example = Neu4mes()
example.addModel(out)
example.neuralizeModel(0.5)
print(example({'x':[2,3,3,5]}))
#

print("------------------------EXAMPLE 3------------------------")
# Example 3
# Modello locale con una funzione di attivazione e una funzione parametrica di uscita
def myFun(in1,p1):
    return in1*p1
loc = LocalModel(output_function = lambda:ParamFun(myFun))(x.last(),activationA)
out = Output('out',loc)
example = Neu4mes()
example.addModel(out)
example.neuralizeModel(0.05)
print(example({'x':[2,3,3]}))
#

print("------------------------EXAMPLE 4------------------------")
# Example 4
# Nuova funzione di attivazione con uscita una finestra temporale di 1 secondo
# Il modello locale ha una funzione di ingresso myFun e una funzione di uscita un filtro Fir che esce con dimensione 1
# Nella funzione di uscita non essendoci la lamda il filtro Fir condivide i pesi in uscita

activationA = Fuzzify(2,[0,1],functions='Triangular')(x.tw(1))
activationB = Fuzzify(2,[0,1],functions='Triangular')(F.tw(1))
def myFun(in1,p1,p2):
    return p1*in1+p2

loc = LocalModel(input_function = lambda:ParamFun(myFun), output_function = Fir(3))(x.tw(1),(activationA,activationB))
# parfun_1 = ParamFun(myFun)(x.tw(1))
# parfun_2 = ParamFun(myFun)(x.tw(1))
# out_in_1 = Output('parfun1', parfun_1)
# out_in_2 = Output('parfun2', parfun_2)
# act = Output('fuzzy',activationA)
# act_sel1 = Select(activationA,0)
# act_sel2 = Select(activationA,1)
# out_act_sel1 = Output('fuzzy_sel0',Select(activationA,0))
# out_act_sel2 = Output('fuzzy_sel1',Select(activationA,1))
# mul1 =  parfun_1*act_sel1
# mul2 =  parfun_2*act_sel2
# out_mul1 = Output('mul1',mul1)
# out_mul2 = Output('mul2',mul2)
# # fir1 =  Fir(mul1)
# # fir2 =  Fir(mul2)
# out_mul1 = Output('mul1',mul1)
# out_mul2 = Output('mul2',mul2)
# sum = mul1+mul2
# out = Output('out', sum)
out = Output('out', loc)
example = Neu4mes()
# example.addModel([out_in_1,out_in_2,out_act_sel1,act,out_act_sel2,out_mul1,out_mul2,out])
example.addModel([out])
example.neuralizeModel(0.5)
print(example({'x':[2,3]}))
#

print("------------------------EXAMPLE 5------------------------")
# Example 5
# Il modello locale non ha una funzione di ingresso ma ha una funzione di uscita un filtro Fir che esce con dimensione 1
# Nella funzione di uscita non essendoci la lamda il filtro Fir condivide i pesi in uscita

loc = LocalModel(output_function = lambda:Fir(2))(x.tw(1),activationB)
out = Output('out', loc)
example = Neu4mes()
example.addModel([out])
example.neuralizeModel(0.5)
print(example({'x':[2,3]}))
#

print("------------------------EXAMPLE 5------------------------")
# Example 5
# Il modello locale non ha una funzione di ingresso ma ha una funzione di uscita un filtro Fir che esce con dimensione 1
# Nella funzione di uscita non essendoci la lamda il filtro Fir condivide i pesi in uscita

def fun(x,p):
    return x*p

loc = LocalModel(output_function = lambda:ParamFun(myFun))(x.tw(1),activationA)
out = Output('out', loc)
example = Neu4mes()
example.addModel([out])
example.neuralizeModel(0.5)
print(example({'x':[2,3]}))
#



# TODO Manca la parte con input_relation_matrix per la funzione parametrica
#loc = LocalModel(input_function = lambda:ParamFun(myFun), input_relation_matrix=[[1,1][1,1]] )(x.tw(1),(activationA,activationB))
#out = Output(x.z(-1),loc)
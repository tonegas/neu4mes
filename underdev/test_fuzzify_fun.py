from neu4mes import *

x = Input('x')
F = Input('F')

print("------------------------EXAMPLE 1------------------------")
# Example 1
# Ho creato una funzione fuzzificatrice con 5 membership function in un intervallo [1,5] della variabile di ingresso
fuz = Fuzzify(5,[1,5])
out = Output(x.z(-1),fuz(x))
example = Neu4mes(verbose = True)
example.addModel(out)
example.neuralizeModel(0.05)
#

print("------------------------EXAMPLE 2------------------------")
# Example 2
# Ho creato una funzione fuzzificatrice con 5 membership function in un intervallo [1,5] della variabile di ingresso
# e funzione di attivazione triangolare
fuz = Fuzzify(5,[1,5], functions = 'Triangular')
out = Output(x.z(-1),fuz(x))
example = Neu4mes(verbose = True)
example.addModel(out)
example.neuralizeModel(0.05)
#

print("------------------------EXAMPLE 3------------------------")
# Example 3
# Crea 6 membership functions dividendo l'intervallo da 1, 6 con funzioni rettangolari
# i centri sono in [1,2,3,4,5,6] le funzioni sono larghe 1 tranne la prima e l'ultima
# [-inf, 1.5] [1.5,2.5] [2.5,3.5] [3.5,4.5] [4.5,5.5] [5.5.inf]
fuz = Fuzzify(6,[1,6], functions = 'Rettangular')
out = Output(x.z(-1),fuz(x))
example = Neu4mes(verbose = True)
example.addModel(out)
example.neuralizeModel(0.05)
#

print("------------------------EXAMPLE 4------------------------")
# Example 4
# Crea 10 membership functions dividendo l'intervallo da -5, 5 con funzioni custom
# i centri sono in [-5,-4,-3,-2,-1,0,1,2,3,4,5]
def fun(x):
    return np.tanh(x)
fuz = Fuzzify(output_dimension = 11, range = [-5,5], functions = fun)
out = Output(x.z(-1),fuz(x))
example = Neu4mes(verbose = True)
example.addModel(out)
example.neuralizeModel(0.05)
#

print("------------------------EXAMPLE 5------------------------")
# Example 5
# Crea 2 membership function custom che si posizionano in -1 e 5
def fun1(x):
    return np.sin(x)
def fun2(x):
    return np.cos(x)
fuz = Fuzzify(2,range=[-1,5],functions=[fun1,fun2]) # Crea 2 memebership function custom
out = Output(x.z(-1),fuz(x))
example = Neu4mes(verbose = True)
example.addModel(out)
example.neuralizeModel(0.05)
#

print("------------------------EXAMPLE 6------------------------")
# Example 6
# Crea 4 membership function custom che si posizionano in [-1,0,3,5]
def fun1(x):
    return np.sin(x)
def fun2(x):
    return np.cos(x)
fuz = Fuzzify(4,centers=[-1,0,3,5],functions=[fun1,fun2,fun1,fun2])
out = Output(x.z(-1),fuz(x.tw(1))+fuz(F.tw(1)))
example = Neu4mes(verbose = True)
example.addModel(out)
example.neuralizeModel(0.05)
#

#TODO fare delle membership funtion usando le funzioni parametriche
# def myFun(in1,p1):
#    return np.sin(in1-p1)
# parfun = ParamFun(myFun,1)
# fuz = Fuzzify(3,range=[-2,2],functions=[parfun,parfun,parfun])
# out = Output(x.z(-1),fuz(x))
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
# Single activation function and only one input
# The input function is a Fir
loc = LocalModel(input_function = lambda : Fir)(x.tw(1),activationA)
out = Output('out',loc)
example = Neu4mes()
example.addModel('out',out)
example.neuralizeModel(0.25)
# The output is 2 samples
print(example({'x':[-1,0,1,2,0]}))
print(example({'x':[[-1,0,1,2],[0,1,2,0]]}, sampled=True))
#

print("------------------------EXAMPLE 2------------------------")
# Example 2
# Single activation function and only one input
# The input function is a single Fir with the shared weight
loc = LocalModel(input_function = Fir())(x.tw(1),activationA)
out = Output('out',loc)
example = Neu4mes()
example.addModel('out',out)
example.neuralizeModel(0.25)
# The output is 2 samples
print(example({'x':[-1,0,1,2,0]}))
print(example({'x':[[-1,0,1,2],[0,1,2,0]]}, sampled=True))
#

print("------------------------EXAMPLE 3------------------------")
# Example 3
# Two activation functions so 2*2 = 4 input functions, use tuple to join the activation functions
# There are 4 Fir filters
loc = LocalModel(input_function = Fir)(x.tw(1),(activationA,activationB))
out = Output('out',loc)
example = Neu4mes()
example.addModel('out',out)
example.neuralizeModel(0.5)
# The output is 3 samples
print(example({'x':[-1,0,1,2],'F':[-1,0,1]}))
#

print("------------------------EXAMPLE 4------------------------")
# Example 4
# Single activation function with ParamFun for output
# The learning parameters are two
def myFun(in1,p1):
    return in1*p1
loc = LocalModel(output_function = lambda:ParamFun(myFun))(x.last(),activationA)
out = Output('out',loc)
example = Neu4mes()
example.addModel('out',out)
example.neuralizeModel(0.05)
# The output is 2 samples
print(example({'x':[2,3]}))
#

print("------------------------EXAMPLE 5------------------------")
# Example 5
# Single activation function with ParamFun for input and Fir for output
# There are 2 parameters for each ParamFun and a parameter for the Fir
# Total number of parameter are 3*2 = 6
def myFun(in1,p1,p2):
    return p1*in1+p2

loc = LocalModel(input_function = lambda:ParamFun(myFun), output_function = lambda:Fir)(x.last(),activationA)
out = Output('out', loc)
example = Neu4mes()
example.addModel('out',out)
example.neuralizeModel(0.5)
# The output is 2 samples
print(example({'x':[2,3]}))
#

print("------------------------EXAMPLE 6------------------------")
# Example 6
# New activation function with a time window of 1 sec
# Two activation functions with ParamFun for input and Fir for output
# The output function is a Fir with shared weights
# The ParamFun have 2 parameters
# There are 2 activiation function with 2 member function, the total numer of parameter is 2*4+1

activationA = Fuzzify(2,[0,1],functions='Triangular')(x.tw(1))
activationB = Fuzzify(2,[0,1],functions='Triangular')(F.tw(1))
def myFun(in1,p1,p2):
    return p1*in1+p2

loc = LocalModel(input_function = lambda:ParamFun(myFun), output_function = Fir(3))(x.tw(1),(activationA,activationB))
out = Output('out', loc)
example = Neu4mes()
example.addModel('out',out)
example.neuralizeModel(0.5)
# Single sample but with dimension 3 due to the dimension of the output of the Fir
print(example({'x':[2,3]}))
#

print("------------------------EXAMPLE 7------------------------")
# Example 7
# Two activation functions with input_function_gen for input and output_fun_gen for output
# It is set pass_indexes to True means the input and output functions are call with the first parametrer is a vector of indexes
# There are 2 activiation functions with 2 member function, the total numer of parameter is (2+2)*4 = 16 but the parametric function use some shared parameters
# so the total number of parameters are 8

def myFun(in1,p1,p2):
    return p1*in1+p2

p1_0 = Parameter('p1_0',values=[[1]])
p1_1 = Parameter('p1_1',values=[[2]])
p2_0 = Parameter('p2_0',values=[[2]])
p2_1 = Parameter('p2_1',values=[[3]])
def input_function_gen(idx_list):
    if idx_list == [0,0]:
        p1, p2 = p1_0, p2_0
    if idx_list == [0,1]:
        p1, p2 = p1_0, p2_1
    if idx_list == [1,0]:
        p1, p2 = p1_1, p2_0
    if idx_list == [1, 1]:
        p1, p2 = p1_1, p2_1
    return ParamFun(myFun,parameters=[p1,p2])
def output_function_gen(idx_list):
    pfir = Parameter('pfir_'+str(idx_list),tw=1,dimensions=2,values=[[1+idx_list[0],2+idx_list[1]],[3+idx_list[0],4+idx_list[1]]])
    return Fir(2,parameter=pfir)

loc = LocalModel(input_function = input_function_gen, output_function = output_function_gen, pass_indexes = True)(x.tw(1),(activationA,activationB))
#Example of the structure of the local model
pfir00 = Parameter('N_pfir_[0, 0]',tw=1,dimensions=2,values=[[1,2],[3,4]])
pfir01 = Parameter('N_pfir_[0, 1]',tw=1,dimensions=2,values=[[1,3],[3,5]])
pfir10 = Parameter('N_pfir_[1, 0]',tw=1,dimensions=2,values=[[2,2],[4,4]])
pfir11 = Parameter('N_pfir_[1, 1]',tw=1,dimensions=2,values=[[2,3],[4,5]])
parfun_00 = ParamFun(myFun,parameters=[p1_0,p2_0])(x.tw(1))
parfun_01 = ParamFun(myFun,parameters=[p1_0,p2_1])(x.tw(1))
parfun_10 = ParamFun(myFun,parameters=[p1_1,p2_0])(x.tw(1))
parfun_11 = ParamFun(myFun,parameters=[p1_1,p2_1])(x.tw(1))
out_in_00 = Output('parfun00', parfun_00)
out_in_01 = Output('parfun01', parfun_01)
out_in_10 = Output('parfun10', parfun_10)
out_in_11 = Output('parfun11', parfun_11)
actA = Output('fuzzyA',activationA)
actB = Output('fuzzyB',activationB)
act_selA0 = Select(activationA,0)
act_selA1 = Select(activationA,1)
act_selB0 = Select(activationB,0)
act_selB1 = Select(activationB,1)
out_act_selA0 = Output('fuzzy_selA0',act_selA0)
out_act_selA1 = Output('fuzzy_selA1',act_selA1)
out_act_selB0 = Output('fuzzy_selB0',act_selB0)
out_act_selB1 = Output('fuzzy_selB1',act_selB1)
mul00 =  parfun_00*act_selA0*act_selB0
mul01 =  parfun_01*act_selA0*act_selB1
mul10 =  parfun_10*act_selA1*act_selB0
mul11 =  parfun_11*act_selA1*act_selB1
out_mul00 = Output('mul00',mul00)
out_mul01 = Output('mul01',mul01)
out_mul10 = Output('mul10',mul10)
out_mul11 = Output('mul11',mul11)
fir00 =  Fir(2,parameter=pfir00)(mul00)
fir01 =  Fir(2,parameter=pfir01)(mul01)
fir10 =  Fir(2,parameter=pfir10)(mul10)
fir11 =  Fir(2,parameter=pfir11)(mul11)
out_fir00 = Output('fir00',fir00)
out_fir01 = Output('fir01',fir01)
out_fir10 = Output('fir10',fir10)
out_fir11 = Output('fir11',fir11)
sum = fir00+fir01+fir10+fir11
out_sum = Output('out_sum', sum)
out = Output('out', loc)
example = Neu4mes()
example.addModel('all_out',[out_in_00,out_in_01,out_in_10,out_in_11,
                  out_act_selA0,out_act_selA1,out_act_selB0,out_act_selB1,
                  out_mul00,out_mul01,out_mul10,out_mul11,
                  out_fir00,out_fir01,out_fir10,out_fir11,
                  out_sum])
example.addModel('out', out)
example.neuralizeModel(0.5)
# Three semples with a dimensions 2
pprint(example({'x':[0,1,-2,3],'F':[-2,2,1,5]}))
#
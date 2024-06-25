import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *

inin = Input('in').last()
par = Parameter('par',sw=1)
inin4 = Input('in',dimensions=4).last()
par4 = Parameter('par4',dimensions=4,sw=1)
add = inin+par+5.2
sub = inin-par-5.2
mul = inin*par*5.2
div = inin/par/5.2
pow = inin**par**5.2
sin = Sin(5.2)
cos = Cos(par)+Cos(5.2)
tan = Tan(par)+Tan(5.2)
relu = Relu(par)+Relu(5.2)
tanh = Tanh(par)+Tanh(5.2)

add4 = inin4+par4+5.2
sub4 = inin4-par4-5.2
mul4 = inin4*par4*5.2
div4 = inin4/par4/5.2
pow4 = inin4**par4**5.2
sin4 = Sin(par4)+Sin(5.2)
cos4 = Cos(par4)+Cos(5.2)
tan4 = Tan(par4)+Tan(5.2)
relu4 = Relu(par4)+Relu(5.2)
tanh4 = Tanh(par4)+Tanh(5.2)
out = Output('out',add+sub+mul+div+pow+Linear(add4+sub4+mul4+div4+pow4)+sin+cos+tan+relu+tanh+Linear(sin4+cos4+tan4+relu4+tanh4))
test = Neu4mes()
test.addModel(out)
test.neuralizeModel()

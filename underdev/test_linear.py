import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *

x = Input('x',dimensions=3)
p = Parameter('p',dimensions=(5,3))
p2 = Parameter('p2',dimensions=(5,5))
out = Linear(Tanh(Linear(W=p2,b=True)(Tanh(Linear(W=p)(x.tw(2))))))

out_out = Output('out',out)

n = Neu4mes()
n.addModel(out_out)
n.neuralizeModel(0.5)
print(n({'x':[[5,3,2],[5,3,2],[5,3,2],[5,3,2]]}))

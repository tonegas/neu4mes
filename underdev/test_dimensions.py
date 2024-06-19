import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
import torch

in1 = Input('in1',dimensions=3)
inFir = Input('in2')
# TODO se rimuovo questo togliendo questo modello dalla lista non funziona più

# Finestre nel tempo
out1 = Output('x.tw(1)', in1.tw(1, offset=0))
#out2 = Output('x.tw([-1,0])', in1.tw([-1, 0], offset=0))
out3 = Output('x.tw([1,3])', in1.tw([1, 3], offset=2))
#out4 = Output('x.tw([-3,-2])', in1.tw([-3, -2], offset=-2))

# Finesatre nei samples
#out5 = Output('x.sw([-1,0])', in1.sw([-1, 0], offset=0))
#out6 = Output('x.sw([-3,1])', in1.sw([-3, 1], offset=-2))
#out7 = Output('x.sw([0,1])', in1.sw([0, 1], offset=1))
#out8 = Output('x.sw([-3, 3])', in1.sw([-3, 3], offset=3))
#out9 = Output('x.sw([-3, 3])-2', in1.sw([-3, 3], offset=0))

test = Neu4mes()
#test.addModel([out0, out1, out2, out3, out4, out5, out6, out7, out8, out9])
test.addModel([out1, out3])

test.neuralizeModel(1)

# Single input
# Time                  -2,         -1,      0,      1,      2,       3 # zero represent the last passed instant
#results = test({'in1': [[-2,3,4],[-1,2,2],[0,0,0],[1,2,3],[2,7,3],[3,3,3]],'in2':[1,2,3,4,5,6]})
results = test({'in1': [[-2,3,4],[-1,2,2],[0,0,0],[1,2,3],[2,3,4]],'in2':[1,2,3,4]})
print('results: ', results)
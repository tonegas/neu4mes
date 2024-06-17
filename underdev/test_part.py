import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *

in1 = Input('in1')

rel1 = Fir(in1.sw(1))
rel2 = Fir(in1.sw([-2,2]))
rel3 = Fir(in1.sw([-5,-2],offset=-3))
rel4 = Fir(SamplePart(in1.sw([-5,-2],offset=-3),0,2))
rel5 = Fir(in1.tw(1))
rel6 = Fir(in1.tw([-2,2]))
rel7 = Fir(in1.tw([-5,-2],offset=-3))
rel8 = Fir(TimePart(in1.tw([-5,-2],offset=-3),0,2))
# La differenza tra TimePart e InputTimePart (stessa cosa vale anche per Sample)
# è che InputTimePart gli indici sono assoluti cioè che se te scrivi tw(-5,-3) vuoi quegli istanti temporali il -5 ed il -4
# Mentre TimePart se te hai un segnale che nel tempo dura 3 campioni del tipo s=[0,1,2] se te vuoi selezioneare
# il primo ed il secondo scriverai TimePart(s,0,2) cioò è gli indici non hanno una valenza temporale specifica.
# In realtà InputTimePart non può essere chiamata direttamente ma solo attraverso il metodo tw.
out = Output('out', rel1+rel2+rel3+rel4+rel5+rel6+rel7+rel8)

# input1 = Input('in1')
# output = Input('out')
# rel1 = Linear(input1.tw(0.05))
# fun = Output(output.z(-1),rel1)

test = Neu4mes()
test.addModel(out)

test.minimizeError('pos', in1.sw([2,3]), out, 'mse')
test.minimizeError('vel', in1.sw([2,3]), rel1, 'mse')

test.neuralizeModel(0.5)

## build custom dataset
data_x = np.asarray(range(1000), dtype=np.float32)
dataset = {'in1': data_x}

test.loadData(source=dataset)

print('BEFORE TRAINING')
sample = {'in1':[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]} ## 2 samples 
results = test(sample, sampled=False)
print('results: ', results)

test.trainModel(test_percentage=10, training_params={'train_batch_size': 8, 'test_batch_size':8})

print('AFTER TRAINING')
results = test(sample, sampled=False)
print('results: ', results)
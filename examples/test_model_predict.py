import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

# Objective of this exmaple
# This example show how you can use your model for inference

from neu4mes import *
import torch
torch.manual_seed(1)

# Consider a simple model for the mass spring damper
x = Input('x')
F = Input('F')
# Last represents the last time instant of a variable
f_last = F.last()
# tw represents the last 0.5 second of a variable
x_win = x.tw(0.5)
# Next represents the next instant value of the variable
next_x = x.next()
# Prediction model
px = Parameter('px', tw=0.5, values=[[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.]])
pf = Parameter('pf', sw=1, values=[[1.]])
model_next_x = Fir(parameter=px)(x_win)+Fir(parameter=pf)(f_last)

# Define one or multiple Outputs to be added to the model
out_model_next_x = Output('model_next_x',model_next_x)
out_x_win = Output('x_win',x_win)
out_f_last = Output('f_last',f_last)
out_next_x = Output('next_x',next_x)

# Create the model and add the outputs
example1 = Neu4mes()
example1.addModel([out_model_next_x,out_x_win,out_f_last,out_next_x])
# Choose the Samplingrate and neuralize the model
example1.neuralizeModel(0.05)

print("------------------------EXAMPLE 1------------------------")
# Call your model with the minimum number of inputs
# 'x' = [1,2,3,4,5,6,7,8,9,10,11] the first 10 sample are in the past and the last sample is the next
# The output model_next_x contain 1 samples
results = example1(inputs={'F':[[9]],'x':[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11]]})
for output, result in results.items():
    print(f'Prediction for {output}: {result}')


print("------------------------EXAMPLE 2------------------------")
# In this case the samples are 12 for x and 2 for F, this means that the network will run for two sample time.
# The first time the input will be F = 5 x = [1,2,3,4,5,6,7,8,9,10] x_next = 11
# The second time the input will be F = 2 x = [2,3,4,5,6,7,8,9,10,11] x_next = 12
# The output model_next_x contain 2 samples
# The output x_win contains the two window used as inputs
results = example1(inputs={'F':[[5],[2]],'x':[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12]]})
for output, result in results.items():
    print(f'prediction for {output}: {result}')

print("------------------------EXAMPLE 3------------------------")
# In this case the samples are 13 for x and 3 for F, this means that the network will run for tree sample time.
results = example1(inputs={'F':[[5],[4],[9]],'x':[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13]]}) # 1 window (x = 10) -> 1 output
for output, result in results.items():
    print(f'prediction for {output}: {result}')

print("------------------------EXAMPLE 4------------------------")
# In this example there is 4 sample for the F and only 12 for the x so the minimum number of sample will be generated.
results = example1(inputs={'F':[[5],[3],[4],[1]],'x':[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12]]})
for output, result in results.items():
    print(f'prediction for {output}: {result}')

print("------------------------EXAMPLE 5------------------------")
# Using the option sampled the input must be created to reflect the need of the network.
# In particular the network needs a sample for F and 11 samples for x
# This way permits to define each sample window independently
results = example1(inputs={'F':[[5],[2]],'x':[[1,2,3,4,5,6,7,8,9,10,11],[12,13,14,15,16,17,18,19,20,21,22]]}, sampled=True)
for output, result in results.items():
    print(f'prediction for {output}: {result}')
    


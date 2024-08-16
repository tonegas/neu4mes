import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
from neu4mes.visualizer import MPLVisulizer
from pprint import pprint

def linear_function(x, k1, k2):
    return x*k1 + k2

##
## 1 * 1 = 1
## (2 - 1)^2 = 1
data_a = np.array([1.0,1.0], dtype=np.float32)
data_b = np.array([2.0,2.0], dtype=np.float32)

dataset = {'a': data_a, 'b_t': data_b}

a = Input('a')
b_t = Input('b_t')
p = Parameter('p', dimensions=(1,1), values=[[[1.0]]])  ## why (1,1)???
b = Output('b',Linear(W=p)(a.last()))

model = Neu4mes(seed=42)
model.addModel('b_model',b)
model.addMinimize('b_min',b,b_t.last())

model.neuralizeModel(0.1)

model.loadData('dataset', dataset)

params = {'num_of_epochs': 1, 
          'train_batch_size': 1, 
          'learning_rate':0.1}

print('before train')
for param in model.model.parameters():
    print(type(param), param.data)
model.trainModel(splits=[100,0,0], training_params=params)
print('after train')
for param in model.model.parameters():
    print(type(param), param.data)
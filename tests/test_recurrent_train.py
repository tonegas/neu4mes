import unittest

import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())
from neu4mes import *

# Linear function
def linear_fun(x,a,b):
    return x*a+b

data_x = np.random.rand(500)*20-10
data_a = 2
data_b = -3
dataset = {'in1': data_x, 'out': linear_fun(data_x,data_a,data_b)}
data_folder = 'data/'

class Neu4mesTrainingTest(unittest.TestCase):
    def test_build_dataset_batch(self):
        
        input1 = Input('in1')
        out = Input('out')
        parfun = ParamFun(linear_fun)
        rel1 = Fir(parfun(input1.tw(0.05)))
        y = Output('y', rel1)


        test = Neu4mes()
        test.addModel(y)
        test.minimizeError('pos', out.z(-1), y)
        test.neuralizeModel(0.01)

        test.loadData(name='data',source=dataset)

        training_params = {}
        training_params['train_batch_size'] = 4
        training_params['test_batch_size'] = 4
        training_params['learning_rate'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainRecurrentModel(close_loop={'in1':'y'}, prediction_horizon=0.05, step=1, test_percentage=30, training_params = training_params)

        # 15 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample 10
        # 10 / 2 = 5 for training and test
        self.assertEqual(86,test.n_samples_train) ## ((500 - 5) * 0.7) // 4 = 86
        self.assertEqual(37,test.n_samples_test) ## ((500 - 5) * 0.3) // 4 = 37
        self.assertEqual(495,test.num_of_samples)
        self.assertEqual(4,test.train_batch_size)
        self.assertEqual(4,test.test_batch_size)
        self.assertEqual(5,test.num_of_epochs)
        self.assertEqual(0.1,test.learning_rate)

if __name__ == '__main__':
    unittest.main()
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

        test.loadData(source=dataset)

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

'''
    def test_build_dataset_batch2(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))

        test = Neu4mes()
        test.minimizeError('out', output.z(-1), rel1)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(source=data_folder, format=data_struct, skiplines=4)
        self.assertEqual((10,5),test.inout_asarray['in1'].shape)

        training_params = {}
        training_params['train_batch_size'] = 25
        training_params['test_batch_size'] = 25
        training_params['learning_rate'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(test_percentage=50,training_params = training_params)

        # 15 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample 10
        # 10 / 2 = 5 for training and test 
        # batch_size > 5 use batch_size = 1
        self.assertEqual(5,test.n_samples_train)
        self.assertEqual(5,test.n_samples_test)
        self.assertEqual(1,test.train_batch_size)
        self.assertEqual(1,test.test_batch_size)
        self.assertEqual(5,test.num_of_epochs)
        self.assertEqual(0.1,test.learning_rate)
    

    def test_build_dataset_batch3(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))

        test = Neu4mes()
        test.minimizeError('out', output.z(-1), rel1)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(source=data_folder, format=data_struct, skiplines=4)
        self.assertEqual((10,5),test.inout_asarray['in1'].shape)

        training_params = {}
        training_params['train_batch_size'] = 2
        training_params['test_batch_size'] = 2
        training_params['learning_rate'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(test_percentage=50, training_params = training_params)

        # 15 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample 10 
        # 5 / 2 = 2 for training
        # 5 / 2 = 2 for test
        # batch_size > 5 -> NO
        # num_of_training_sample must be multiple of batch_size
        # num_of_test_sample must be multiple of batch_size and at least 50%
        self.assertEqual(2,test.n_samples_train)
        self.assertEqual(2,test.n_samples_test)
        self.assertEqual(10,test.num_of_samples)
        self.assertEqual(2,test.train_batch_size)
        self.assertEqual(2,test.test_batch_size)
        self.assertEqual(5,test.num_of_epochs)
        self.assertEqual(0.1,test.learning_rate)
    
    def test_build_dataset_batch4(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))

        test = Neu4mes()
        test.minimizeError('out', output.z(-1), rel1)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(source=data_folder, format=data_struct, skiplines=4)
        self.assertEqual((10,5),test.inout_asarray['in1'].shape)

        training_params = {}
        training_params['train_batch_size'] = 2
        training_params['test_batch_size'] = 2
        training_params['learning_rate'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(test_percentage=10, training_params = training_params)

        # 15 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample 10
        # 10 / 100 * 10 = 1 for training and test 
        # batch_size > 1 -> YES
        # num_of_training_sample must be multiple of batch_size
        # num_of_test_sample must be multiple of batch_size and at least 10%
        self.assertEqual(4,test.n_samples_train)
        self.assertEqual(1,test.n_samples_test)
        self.assertEqual(10,test.num_of_samples)
        self.assertEqual(2,test.train_batch_size)
        self.assertEqual(1,test.test_batch_size)
        self.assertEqual(5,test.num_of_epochs)
        self.assertEqual(0.1,test.learning_rate)

    def test_build_dataset_from_code(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))

        test = Neu4mes()
        test.minimizeError('out', output.z(-1), rel1)
        test.neuralizeModel(0.01)

        x_size = 10
        data_x = np.random.rand(x_size)*20-10
        data_a = 2
        data_b = -3
        dataset = {'in1': data_x, 'out': data_x*data_a+data_b}

        test.loadData(source=dataset, skiplines=4)
        self.assertEqual((5,5),test.inout_asarray['in1'].shape)  ## 10 data - 6 tw = 4 sample | 0.05/0.01 = 5 in1

        training_params = {}
        training_params['train_batch_size'] = 2
        training_params['test_batch_size'] = 2
        training_params['learning_rate'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(test_percentage=10, training_params = training_params)

        # 10 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample (10 - 5 - 1) = 4
        # 4 / 100 * 10 = 1 for test and 3 for training
        # batch_size > 1 -> YES
        # num_of_training_sample must be multiple of batch_size
        # num_of_test_sample must be multiple of batch_size and at least 10%
        self.assertEqual(2,test.n_samples_train)
        self.assertEqual(1,test.n_samples_test)
        self.assertEqual(5,test.num_of_samples)
        self.assertEqual(2,test.train_batch_size)
        self.assertEqual(1,test.test_batch_size)
        self.assertEqual(5,test.num_of_epochs)
        self.assertEqual(0.1,test.learning_rate)
'''

if __name__ == '__main__':
    unittest.main()
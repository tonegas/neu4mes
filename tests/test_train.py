import unittest, logging
import numpy as np
from neu4mes import *

import os
is_travis = 'TRAVIS' in os.environ
if is_travis:
    data_folder = '/home/travis/build/tonegas/neu4mes/tests/data/'
else:
    data_folder = './tests/data/'

class Neu4mesTrainingTest(unittest.TestCase):
    def test_build_dataset_batch(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Linear(input1.tw(0.05))
        fun = Output(output.z(-1),rel1)

        test = Neu4mes()
        test.addModel(fun)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(data_struct, folder = data_folder, skiplines = 4)
        self.assertEqual((10,5),test.inout_asarray['in1'].shape)

        training_params = {}
        training_params['batch_size'] = 5
        training_params['learning_rate'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(training_params = training_params, test_percentage = 50)

        # 15 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample 10
        # 10 / 2 = 5 for training and test
        self.assertEqual(5,test.num_of_training_sample)
        self.assertEqual(5,test.num_of_test_sample)
        self.assertEqual(10,test.num_of_samples)
        self.assertEqual(5,test.batch_size)
        self.assertEqual(5,test.num_of_epochs)
        self.assertEqual(0.1,test.learning_rate)

    def test_build_dataset_batch2(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Linear(input1.tw(0.05))
        fun = Output(output.z(-1),rel1)

        test = Neu4mes()
        test.addModel(fun)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(data_struct, folder = data_folder, skiplines = 4)
        self.assertEqual((10,5),test.inout_asarray['in1'].shape)

        training_params = {}
        training_params['batch_size'] = 25
        training_params['learning_rate'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(training_params = training_params, test_percentage = 50)

        # 15 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample 10
        # 10 / 2 = 5 for training and test 
        # batch_size > 5 use batch_size = 1
        self.assertEqual(5,test.num_of_training_sample)
        self.assertEqual(5,test.num_of_test_sample)
        self.assertEqual(10,test.num_of_samples)
        self.assertEqual(1,test.batch_size)
        self.assertEqual(5,test.num_of_epochs)
        self.assertEqual(0.1,test.learning_rate)

    def test_build_dataset_batch3(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Linear(input1.tw(0.05))
        fun = Output(output.z(-1),rel1)

        test = Neu4mes()
        test.addModel(fun)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(data_struct, folder = data_folder, skiplines = 4)
        self.assertEqual((10,5),test.inout_asarray['in1'].shape)

        training_params = {}
        training_params['batch_size'] = 2
        training_params['learning_rate'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(training_params = training_params, test_percentage = 50)

        # 15 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample 10
        # 10 / 2 = 5 for training and test 
        # batch_size > 5 -> NO
        # num_of_training_sample must be multiple of batch_size
        # num_of_test_sample must be multiple of batch_size and at least 50%
        self.assertEqual(4,test.num_of_training_sample)
        self.assertEqual(6,test.num_of_test_sample)
        self.assertEqual(10,test.num_of_samples)
        self.assertEqual(2,test.batch_size)
        self.assertEqual(5,test.num_of_epochs)
        self.assertEqual(0.1,test.learning_rate)

    def test_build_dataset_batch3(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Linear(input1.tw(0.05))
        fun = Output(output.z(-1),rel1)

        test = Neu4mes()
        test.addModel(fun)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(data_struct, folder = data_folder, skiplines = 4)
        self.assertEqual((10,5),test.inout_asarray['in1'].shape)

        training_params = {}
        training_params['batch_size'] = 2
        training_params['learning_rate'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(training_params = training_params, test_percentage = 10)

        # 15 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample 10
        # 10 / 100 * 10 = 1 for training and test 
        # batch_size > 1 -> YES
        # num_of_training_sample must be multiple of batch_size
        # num_of_test_sample must be multiple of batch_size and at least 10%
        self.assertEqual(9,test.num_of_training_sample)
        self.assertEqual(1,test.num_of_test_sample)
        self.assertEqual(10,test.num_of_samples)
        self.assertEqual(1,test.batch_size)
        self.assertEqual(5,test.num_of_epochs)
        self.assertEqual(0.1,test.learning_rate)
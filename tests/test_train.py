import unittest

import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())
from neu4mes import *
data_folder = os.path.join(os.path.dirname(__file__), 'data/')

class Neu4mesTrainingTest(unittest.TestCase):
    def test_network_mass_spring_damper(self):
        x = Input('x')  # Position
        F = Input('F')  # Force

        # List the output of the model
        x_z = Output('x_z', Fir(x.tw(0.3)) + Fir(F))

        # Add the neural model to the neu4mes structure and neuralization of the model
        test = Neu4mes()
        test.addModel(x_z)
        test.minimizeError('next-pos', x.z(-1), x_z, 'mse')

        # Create the neural network
        test.neuralizeModel(sample_time=0.05)  # The sampling time depends to the dataset

        # Data load
        data_struct = ['x','F','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(source=data_folder, format=data_struct, skiplines=4)
        test.trainModel(test_percentage=30)


    def test_build_dataset_batch(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))

        test = Neu4mes(visualizer=None)
        test.minimizeError('out', output.z(-1), rel1)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(source=data_folder, format=data_struct, skiplines=4)
        self.assertEqual((10,5),test.inout_asarray['in1'].shape)

        training_params = {}
        training_params['train_batch_size'] = 1
        training_params['test_batch_size'] = 1
        training_params['learning_rate'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(test_percentage=30, training_params = training_params)

        # 15 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample 10
        # 10 / 2 = 5 for training and test
        self.assertEqual(7,test.n_samples_train)
        self.assertEqual(3,test.n_samples_test)
        self.assertEqual(10,test.num_of_samples)
        self.assertEqual(1,test.train_batch_size)
        self.assertEqual(1,test.test_batch_size)
        self.assertEqual(5,test.num_of_epochs)
        self.assertEqual(0.1,test.learning_rate)

    
    def test_build_dataset_batch2(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))

        test = Neu4mes(visualizer=None)
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

        test = Neu4mes(visualizer=None)
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

        test = Neu4mes(visualizer=None)
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

        test = Neu4mes(visualizer=None)
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
    

if __name__ == '__main__':
    unittest.main()
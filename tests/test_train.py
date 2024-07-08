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
        x_z = Output('x_z', Fir(x.tw(0.3)) + Fir(F.last()))

        # Add the neural model to the neu4mes structure and neuralization of the model
        test = Neu4mes(visualizer=None)
        test.addModel(x_z)
        test.minimizeError('next-pos', x.z(-1), x_z, 'mse')

        # Create the neural network
        test.neuralizeModel(sample_time=0.05)  # The sampling time depends to the dataset

        # Data load
        data_struct = ['x','F','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        test.trainModel(splits=[80,10,10])
    
    def test_build_dataset_batch(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))

        test = Neu4mes(visualizer=None)
        test.minimizeError('out', output.z(-1), rel1)
        test.neuralizeModel(0.01)

        data_struct = ['x','F','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10,5,1),test.data['dataset']['in1'].shape)

        training_params = {}
        training_params['train_batch_size'] = 1
        training_params['val_batch_size'] = 1
        training_params['test_batch_size'] = 1
        training_params['learning_rate'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(splits=[70,20,10],training_params = training_params)

        # 15 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample 10
        # 10 / 1 * 0.7 = 7 for training
        # 10 / 1 * 0.2 = 2 for validation
        # 10 / 1 * 0.1 = 1 for test
        self.assertEqual(7,test.n_samples_train)
        self.assertEqual(2,test.n_samples_val)
        self.assertEqual(1,test.n_samples_test)
        self.assertEqual(10,test.num_of_samples['dataset'])
        self.assertEqual(1,test.train_batch_size)
        self.assertEqual(1,test.val_batch_size)
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

        data_struct = ['x','F','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset',source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10,5,1),test.data['dataset']['in1'].shape)

        training_params = {}
        training_params['train_batch_size'] = 25
        training_params['val_batch_size'] = 25
        training_params['test_batch_size'] = 25
        training_params['learning_rate'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(splits=[50,0,50],training_params = training_params)

        # 15 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample 10 
        # batch_size > 5 use batch_size = 1
        # 10 / 1 * 0.5 = 5 for training
        # 10 / 1 * 0.0 = 0 for validation
        # 10 / 1 * 0.5 = 5 for test
        self.assertEqual(5,test.n_samples_train)
        self.assertEqual(0,test.n_samples_val)
        self.assertEqual(5,test.n_samples_test)
        self.assertEqual(1,test.train_batch_size)
        self.assertEqual(1,test.val_batch_size)
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

        data_struct = ['x','F','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10,5,1),test.data['dataset']['in1'].shape)

        training_params = {}
        training_params['train_batch_size'] = 2
        training_params['val_batch_size'] = 2
        training_params['test_batch_size'] = 2
        training_params['learning_rate'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(splits=[40,30,30], training_params = training_params)

        # 15 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample 10 
        # batch_size > 5 -> NO
        # num_of_training_sample must be multiple of batch_size
        # num_of_test_sample must be multiple of batch_size and at least 50%
        # 10 / 2 * 0.4 = 2 for training
        # 10 / 2 * 0.3 = 1 for validation
        # 10 / 2 * 0.3 = 1 for test
        self.assertEqual(2,test.n_samples_train)
        self.assertEqual(1,test.n_samples_val)
        self.assertEqual(1,test.n_samples_test)
        self.assertEqual(10,test.num_of_samples['dataset'])
        self.assertEqual(2,test.train_batch_size)
        self.assertEqual(2,test.val_batch_size)
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

        data_struct = ['x','F','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10,5,1),test.data['dataset']['in1'].shape)

        training_params = {}
        training_params['train_batch_size'] = 2
        training_params['val_batch_size'] = 2
        training_params['test_batch_size'] = 2
        training_params['learning_rate'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(splits=[80,10,10], training_params = training_params)

        # 15 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample 10
        # batch_size > 1 -> YES
        # num_of_training_sample must be multiple of batch_size
        # num_of_test_sample must be multiple of batch_size and at least 10%
        # 10 / 2 * 0.8 = 4 for training
        # 10 / 1 * 0.1 = 1 for validation
        # 10 / 1 * 0.1 = 1 for test
        self.assertEqual(4,test.n_samples_train)
        self.assertEqual(1,test.n_samples_val)
        self.assertEqual(1,test.n_samples_test)
        self.assertEqual(10,test.num_of_samples['dataset'])
        self.assertEqual(2,test.train_batch_size)
        self.assertEqual(1,test.val_batch_size)
        self.assertEqual(1,test.test_batch_size)
        self.assertEqual(5,test.num_of_epochs)
        self.assertEqual(0.1,test.learning_rate)
    
    def test_build_dataset_from_code(self):
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))

        test = Neu4mes(visualizer=None)
        test.minimizeError('out', output.next(), rel1)
        test.neuralizeModel(0.01)

        x_size = 20
        data_x = np.random.rand(x_size)*20-10
        data_a = 2
        data_b = -3
        dataset = {'in1': data_x, 'out': data_x*data_a+data_b}

        test.loadData(name='dataset', source=dataset, skiplines=0)
        self.assertEqual((15,5,1),test.data['dataset']['in1'].shape)  ## 20 data - 5 tw = 15 sample | 0.05/0.01 = 5 in1

        training_params = {}
        training_params['train_batch_size'] = 2
        training_params['val_batch_size'] = 2
        training_params['test_batch_size'] = 2
        training_params['learning_rate'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(splits=[80,20,0], training_params = training_params)

        # 20 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample (20 - 5 - 1) = 16
        # batch_size > 1 -> YES
        # num_of_training_sample must be multiple of batch_size
        # num_of_test_sample must be multiple of batch_size and at least 10%
        # 15 / 2 * 0.8 = 6 for training
        # 15 / 2 * 0.2 = 1 for validation
        # 15 / 2 * 0.0 = 0 for test
        self.assertEqual(6,test.n_samples_train)
        self.assertEqual(1,test.n_samples_val)
        self.assertEqual(0,test.n_samples_test)
        self.assertEqual(15,test.num_of_samples['dataset'])
        self.assertEqual(2,test.train_batch_size)
        self.assertEqual(2,test.val_batch_size)
        self.assertEqual(1,test.test_batch_size)
        self.assertEqual(5,test.num_of_epochs)
        self.assertEqual(0.1,test.learning_rate)
    
    def test_network_multi_dataset(self):
        train_folder = os.path.join(os.path.dirname(__file__), 'data/')
        val_folder = os.path.join(os.path.dirname(__file__), 'val_data/')
        test_folder = os.path.join(os.path.dirname(__file__), 'test_data/')

        x = Input('x')  # Position
        F = Input('F')  # Force

        # List the output of the model
        x_z = Output('x_z', Fir(x.tw(0.3)) + Fir(F.last()))

        # Add the neural model to the neu4mes structure and neuralization of the model
        test = Neu4mes(visualizer=None)
        test.addModel(x_z)
        test.minimizeError('next-pos', x.z(-1), x_z, 'mse')

        # Create the neural network
        test.neuralizeModel(sample_time=0.05)  # The sampling time depends to the dataset

        # Data load
        data_struct = ['x','F','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='train_dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        test.loadData(name='validation_dataset', source=val_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        test.loadData(name='test_dataset', source=test_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)

        training_params = {}
        training_params['train_batch_size'] = 3
        training_params['val_batch_size'] = 2
        training_params['test_batch_size'] = 1
        training_params['learning_rate'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(train_dataset='train_dataset', validation_dataset='validation_dataset', test_dataset='test_dataset', training_params=training_params)

        self.assertEqual(9,test.num_of_samples['train_dataset'])
        self.assertEqual(9,test.num_of_samples['validation_dataset'])
        self.assertEqual(9,test.num_of_samples['test_dataset'])
        self.assertEqual(3,test.train_batch_size)
        self.assertEqual(2,test.val_batch_size)
        self.assertEqual(1,test.test_batch_size)
        self.assertEqual(3,test.n_samples_train)
        self.assertEqual(4,test.n_samples_val)
        self.assertEqual(9,test.n_samples_test)
        self.assertEqual(5,test.num_of_epochs)
        self.assertEqual(0.1,test.learning_rate)
    
if __name__ == '__main__':
    unittest.main()